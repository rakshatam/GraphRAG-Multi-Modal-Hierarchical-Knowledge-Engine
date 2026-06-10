import os
import sys
import html
import json
import tempfile
import asyncio
import threading
import re
import gc  # Added for aggressive memory clearing
from typing import Tuple, List, Dict, Set

import torch
import networkx as nx
from networkx.algorithms import community
from pyvis.network import Network
import lancedb
import pyarrow as pa
from loguru import logger

# --- Force clear out hidden zombie memory from previous cell crashes ---
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- SOTA Search & Extraction Upgrades ---
from gliner import GLiNER
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

# --- UI & Framework Layers ---
import gradio as gr
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI

# --- Parser Engines ---
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
import whisper
from moviepy.editor import VideoFileClip 

# ==========================================
# 0. INITIALIZATION & STRUCTURED LOGGING
# ==========================================
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=False)
    logger.info("Multiprocessing sub-runtime safely initialized via 'spawn'.")
except RuntimeError:
    logger.warning("Multiprocessing context layout already fixed or configured.")

STORAGE_DIR = "./graphrag_sota_vault"
DB_PATH = os.path.join(STORAGE_DIR, "lancedb")
KG_PATH = os.path.join(STORAGE_DIR, "knowledge_graph.graphml")
CONTEXT_PATH = os.path.join(STORAGE_DIR, "knowledge_graph_context.json")
os.makedirs(STORAGE_DIR, exist_ok=True)

NGROK_TOKEN = os.environ.get("KAGGLE_SECRET_NGROK_TOKEN", "YOUR_NGROK_AUTH_TOKEN")

# ==========================================
# 1. DATABASE & HYBRID MEMORY REPOSITORIES
# ==========================================
db = lancedb.connect(DB_PATH)
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 384)), 
    pa.field("entity", pa.string())
])

if "knowledge_base" in db.table_names():
    logger.info("Existing LanceDB vectorized entity matrix identified.")
    tbl = db.open_table("knowledge_base")
else:
    logger.info("Initializing pristine LanceDB vectorized context structure schema...")
    tbl = db.create_table("knowledge_base", schema=schema)

state_lock = asyncio.Lock()
MODEL_INFERENCE_LOCK = threading.Lock()

KG = nx.DiGraph()
KG_UNDIRECTED = nx.Graph()
DISPLAY_KG = nx.DiGraph()
COMMUNITY_MAP: Dict[str, List[str]] = {}
COMMUNITY_LABELS: Dict[str, str] = {}
KNOWN_ENTITIES: Set[str] = set()
ENTITY_LOWER_TO_ORIGINAL: Dict[str, str] = {}
BM25_INDEX: BM25Okapi = None
BM25_CORPUS_MAP: List[str] = []
IS_INDEX_BUILT = False
MAX_CONTEXT_PER_NODE = 10

# ==========================================
# 2. STRATIFIED DUAL-GPU HARDWARE ALLOCATION
# ==========================================
device_count = torch.cuda.device_count()
logger.info(f"VRAM Interrogation Profile: Found {device_count} valid CUDA computation cores.")

if device_count >= 2:
    # Strict isolation strategy to guarantee zero VRAM spillover crashes
    whisper_device = "cuda:1"
    gliner_device = "cuda:1"
    llm_device_map = {"": 0}       # Enforces Qwen completely onto GPU 0
    embedder_device = "cuda:0"
    reranker_device = "cuda:0"
    logger.info("SOTA Resource Split: Models isolated across dual GPU engines to guarantee structural headrooms.")
else:
    whisper_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gliner_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    llm_device_map = "auto"
    embedder_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reranker_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Specialized Inference Configuration Profiles
logger.info("Loading GLiNER zero-shot name extraction cluster...")
gliner_model = GLiNER.from_pretrained("Urchade/gliner_medium-v2.1").to(gliner_device)

logger.info("Loading Speech-to-Text Whisper Cluster (Large-v3)...")
audio_model = whisper.load_model("large-v3", device=whisper_device)

logger.info("Initializing High-Throughput Transformers & Cross-Encoders...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=reranker_device)

logger.info("Deploying Optimized Causal Language Processing Grid (Qwen 2.5 3B Quantized)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# FIXED: Switched to 3B model for structural optimization within T4 limitations
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map=llm_device_map
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ==========================================
# 3. TEXT SYNTACTIC & MEDIA INGESTION LAYERS
# ==========================================
def parse_file(file_path: str) -> str:
    if os.path.getsize(file_path) > 200 * 1024 * 1024:
        return "ERROR: File exceeds maximum allowed size of 200MB."

    ext = file_path.lower().split('.')[-1]
    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == "docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext in ["xml", "html"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return BeautifulSoup(f.read(), "html.parser").get_text(separator=' ')
        elif ext in ["txt", "md"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == "csv":
            import csv
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return "\n".join([", ".join(row) for row in csv.reader(f)])
        elif ext in ["mp3", "wav", "m4a"]:
            return audio_model.transcribe(file_path)["text"]
        elif ext in ["mp4", "mkv", "avi"]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_name = tmp_audio.name
            try:
                with VideoFileClip(file_path) as clip:
                    if clip.audio is None:
                        return "ERROR: Targeted media container possesses no audio stream track."
                    clip.audio.write_audiofile(tmp_audio_name, logger=None)
                return audio_model.transcribe(tmp_audio_name)["text"]
            finally:
                if os.path.exists(tmp_audio_name):
                    os.remove(tmp_audio_name)
        else:
            return f"ERROR: Unsupported payload structure representation '.{ext}'."
    except Exception as e:
        logger.error(f"Parser processing pipeline hit an anomaly: {e}")
        return f"ERROR: Processing infrastructure collapsed: {str(e)}"

def chunk_text(text: str, words_per_chunk: int = 400, overlap_words: int = 50) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_chunk = []
            overlap_length = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len <= overlap_words:
                    overlap_chunk.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(sentence)
        current_length += sentence_length
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks if chunks else [text]

# ==========================================
# 4. DATA EXTRACTION COMPILING METRICS
# ==========================================
def run_qwen_relation_extraction(chunk: str, entities: List[str], retries: int = 2) -> str:
    if len(entities) < 2:
        return ""
    sanitized_chunk = chunk.replace("<|im_start|>", "").replace("<|im_end|>", "")
    entities_list_str = ", ".join(entities)
    
    prompt = (
        "<|im_start|>system\n"
        "You are a Factual Relation Mapping Engine.\n"
        "Analyze the text and extract explicit relationships ONLY between the provided entities.\n"
        "Output ONLY a valid Markdown table format: | Subject | Relation | Object |\n"
        "Do not invent structural inverted paths if direct pathways exist.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Verified Entities: {entities_list_str}\n"
        f"Text:\n{sanitized_chunk}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "| Subject | Relation | Object |\n|---|---|---|\n"
    )
    
    for attempt in range(retries + 1):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with MODEL_INFERENCE_LOCK:
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=384, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            if attempt == retries:
                logger.error(f"Qwen relation extraction failed after retries: {e}")
                return ""
            logger.warning("Relation extraction interrupted. Retrying generation call...")

async def run_community_auto_labeling(community_map_snapshot: Dict[str, List[str]] = None):
    global COMMUNITY_LABELS
    logger.info("Initializing LLM-driven community clustering optimization...")
    loop = asyncio.get_running_loop()
    target_map = community_map_snapshot or COMMUNITY_MAP
    
    for comm_id, nodes in target_map.items():
        if not nodes:
            continue
        top_nodes = nodes[:10]
        prompt = (
            "<|im_start|>system\nGenerate a conceptual summary title (3 words maximum) characterizing this group of entities.\n"
            "Output ONLY the clear title text without extra formatting.\n<|im_end|>\n"
            f"<|im_start|>user\nEntities: {', '.join(top_nodes)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        def _generate_label():
            tok_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with MODEL_INFERENCE_LOCK:
                with torch.inference_mode():
                    out = model.generate(**tok_inputs, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            return tokenizer.decode(out[0][tok_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
        try:
            label = await loop.run_in_executor(None, _generate_label)
            COMMUNITY_LABELS[comm_id] = label if label else "General Concept Base"
        except Exception as e:
            logger.error(f"Failed to extract semantic title for community {comm_id}: {e}")
            COMMUNITY_LABELS[comm_id] = "Unlabeled Cluster"

async def build_bm25_index_isolated():
    global BM25_INDEX, BM25_CORPUS_MAP
    loop = asyncio.get_running_loop()
    
    async with state_lock:
        corpus_snapshot = list(KNOWN_ENTITIES)
        
    if not corpus_snapshot:
        return
        
    new_index = await loop.run_in_executor(None, lambda: BM25Okapi([doc.split(" ") for doc in corpus_snapshot]))
    
    async with state_lock:
        BM25_CORPUS_MAP = corpus_snapshot
        BM25_INDEX = new_index
        
    logger.info(f"Rebuilt BM25 matching corpus across {len(corpus_snapshot)} records atomically.")

# ==========================================
# 5. WORKER PROCESSING LOOPS
# ==========================================
async def ingest_data(raw_text: str = None, file_obj=None) -> Tuple[str, str]:
    global DISPLAY_KG, KG_UNDIRECTED, IS_INDEX_BUILT, KNOWN_ENTITIES, ENTITY_LOWER_TO_ORIGINAL
    text_to_process = ""
    ux_warning = ""
    
    if file_obj is not None:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        if raw_text:
            ux_warning = "⚠️ FILE PRIORITY: Input prioritized file buffer. Ignoring text area fields.\n\n"
        parsed_result = parse_file(file_path)
        if parsed_result.startswith("ERROR:"):
            return parsed_result, await generate_graph_html()
        text_to_process = parsed_result
    elif raw_text:
        text_to_process = raw_text
        
    if not text_to_process.strip():
        return "System identified no transactional text context payloads.", await generate_graph_html()

    chunks = chunk_text(text_to_process)
    triplets_found = []
    new_nodes_to_index = []
    
    labels_to_extract = ["Person", "Organization", "Location", "Concept", "Artifact"]
    loop = asyncio.get_running_loop()
    
    for chunk in chunks:
        gliner_entities = await loop.run_in_executor(
            None, lambda: gliner_model.predict_entities(chunk, labels_to_extract, threshold=0.45)
        )
        extracted_entity_names = list(set([ent["text"].strip() for ent in gliner_entities if len(ent["text"].strip()) > 1]))
        
        if len(extracted_entity_names) >= 2:
            relations_md = await loop.run_in_executor(
                None, lambda: run_qwen_relation_extraction(chunk, extracted_entity_names)
            )
            
            chunk_triplets = []
            for line in relations_md.split('\n'):
                line = line.strip()
                if line.startswith('|') and line.endswith('|') and line.count('|') >= 4:
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) >= 3 and "Subject" not in parts[0] and "---" not in parts[0]:
                        subj, rel, obj = parts[0], parts[1], parts[2]
                        if subj and rel and obj:
                            chunk_triplets.append((subj, rel, obj))
                            
            if chunk_triplets:
                async with state_lock:
                    for subj, rel, obj in chunk_triplets:
                        subj_norm = subj.lower().strip()
                        obj_norm = obj.lower().strip()
                        
                        if subj not in KG:
                            KG.add_node(subj, context=[])
                            if subj_norm not in KNOWN_ENTITIES:
                                KNOWN_ENTITIES.add(subj_norm)
                                ENTITY_LOWER_TO_ORIGINAL[subj_norm] = subj
                                new_nodes_to_index.append(subj)
                        if obj not in KG:
                            KG.add_node(obj, context=[])
                            if obj_norm not in KNOWN_ENTITIES:
                                KNOWN_ENTITIES.add(obj_norm)
                                ENTITY_LOWER_TO_ORIGINAL[obj_norm] = obj
                                new_nodes_to_index.append(obj)
                                
                        if len(KG.nodes[subj]['context']) < MAX_CONTEXT_PER_NODE:
                            KG.nodes[subj]['context'].append(chunk[:250] + "...")
                        if len(KG.nodes[obj]['context']) < MAX_CONTEXT_PER_NODE:
                            KG.nodes[obj]['context'].append(chunk[:250] + "...")
                            
                        KG.add_edge(subj, obj, label=rel)
                        triplets_found.append(f"({subj}) --[{rel}]--> ({obj})")

    embeddings = None
    if new_nodes_to_index:
        embeddings = await loop.run_in_executor(
            None, lambda: embedder.encode(new_nodes_to_index, show_progress_bar=False).tolist()
        )
        
    communities_result = None
    kg_copy_for_community = None
    
    async with state_lock:
        KG_UNDIRECTED = KG.to_undirected()
        kg_copy_for_community = KG_UNDIRECTED.copy()
        
    if len(kg_copy_for_community.nodes) > 1:
        communities_result = await loop.run_in_executor(
            None, lambda: community.louvain_communities(kg_copy_for_community, seed=42)
        )

    records = None
    display_kg_snapshot = None
    kg_for_graphml = None
    context_payload = {}
    community_map_snapshot = None
    
    async with state_lock:
        if new_nodes_to_index and embeddings:
            records = [{"vector": emb, "entity": ent} for emb, ent in zip(embeddings, new_nodes_to_index)]
            
        if communities_result:
            COMMUNITY_MAP.clear()
            COMMUNITY_LABELS.clear() 
            for i, comm in enumerate(communities_result):
                c_id = f"Cluster_{i}"
                COMMUNITY_MAP[c_id] = list(comm)
                for node in comm:
                    if node in KG:
                        KG.nodes[node]['community_id'] = c_id
                        
        kg_for_graphml = KG.copy()
        for n in kg_for_graphml.nodes:
            kg_for_graphml.nodes[n].pop("context", None)
            
        context_payload = {n: KG.nodes[n].get("context", []) for n in KG.nodes}
        community_map_snapshot = {k: list(v) for k, v in COMMUNITY_MAP.items()}
        
        display_kg_snapshot = KG.copy()
        PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
        for node in display_kg_snapshot.nodes:
            c_id = display_kg_snapshot.nodes[node].get('community_id', 'Cluster_0')
            try: cluster_num = int(c_id.split("_")[1])
            except Exception: cluster_num = 0
            display_kg_snapshot.nodes[node]['color'] = PALETTE[cluster_num % len(PALETTE)]
            display_kg_snapshot.nodes[node].pop('context', None)
            
        DISPLAY_KG = display_kg_snapshot

    if records:
        await loop.run_in_executor(None, lambda: tbl.add(records))
        
    if communities_result:
        await run_community_auto_labeling(community_map_snapshot)
        await build_bm25_index_isolated()

    async with state_lock:
        should_build = (not IS_INDEX_BUILT)
        
    if should_build:
        row_count = await loop.run_in_executor(None, tbl.count_rows)
        if row_count >= 256:
            try:
                await loop.run_in_executor(None, lambda: tbl.create_index("vector", metric="cosine"))
                async with state_lock:
                    IS_INDEX_BUILT = True
            except Exception:
                pass
            
    try:
        await loop.run_in_executor(None, lambda: nx.write_graphml(kg_for_graphml, KG_PATH))
        await loop.run_in_executor(None, lambda: json.dump(context_payload, open(CONTEXT_PATH, "w"), ensure_ascii=False))
    except Exception as e:
        logger.error(f"Persistence save error written to disk tracks: {e}")

    return f"{ux_warning}GLiNER successfully verified text properties inside parsing tracks.\nMaterialized {len(triplets_found)} advanced conceptual structural links.\n\n" + "\n".join(triplets_found), await generate_graph_html()

async def generate_graph_html() -> str:
    async with state_lock:
        if len(DISPLAY_KG.nodes) == 0:
            return "<h3 style='color:white;text-align:center;padding:20px;'>Topology Storage Engine is currently unpopulated.</h3>"
        display_copy = DISPLAY_KG.copy()
        
    net = Network(notebook=False, directed=True, height="600px", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources='in_line')
    net.from_nx(display_copy)
    net.repulsion(node_distance=180, spring_length=220)
    escaped_html = html.escape(net.generate_html(), quote=True)
    return f'<iframe srcdoc="{escaped_html}" width="100%" height="600px" style="border:none;border-radius:8px;"></iframe>'

# ==========================================
# 6. RETRIEVAL PATH CROSS-ENCODER STRATIFICATION
# ==========================================
async def query_graph_stream(question: str):
    if not question or not question.strip():
        yield "Please provide a valid query to search the Knowledge Graph.", "Empty Retrieval Payload."
        return

    async with state_lock:
        if len(KG.nodes) == 0:
            yield "Database context is unpopulated.", "Graph empty."
            return

    loop = asyncio.get_running_loop()
    
    query_vector = await loop.run_in_executor(None, lambda: embedder.encode(question).tolist())
    vector_results = await loop.run_in_executor(None, lambda: tbl.search(query_vector).metric("cosine").limit(10).to_list())
    candidate_entities = [res["entity"] for res in vector_results]
    
    if BM25_INDEX is not None:
        tokenized_query = question.lower().split(" ")
        bm25_scores = BM25_INDEX.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0.0:
                ent_name = BM25_CORPUS_MAP[idx]
                original_ent = ENTITY_LOWER_TO_ORIGINAL.get(ent_name)
                if original_ent and original_ent not in candidate_entities:
                    candidate_entities.append(original_ent)

    if not candidate_entities:
        yield "Search parameters matched no valid topology candidates.", "Empty Retrieval Payload."
        return

    pairs = [[question, ent] for ent in candidate_entities]
    rerank_scores = await loop.run_in_executor(None, lambda: reranker.predict(pairs).tolist())
    ranked_entities = [ent for _, ent in sorted(zip(rerank_scores, candidate_entities), key=lambda x: x[0], reverse=True)[:4]]

    retrieved_edges = set()
    node_contexts = set()
    retrieved_communities = set()
    
    async with state_lock:
        kg_snap = KG.copy()
        kg_un_snap = KG_UNDIRECTED.copy()
        entity_contexts = {e: list(KG.nodes[e].get('context', [])) for e in ranked_entities if e in KG}
        entity_communities = {e: KG.nodes[e].get('community_id') for e in ranked_entities if e in KG}
        community_labels_snap = dict(COMMUNITY_LABELS)
        community_map_snap = {k: list(v) for k, v in COMMUNITY_MAP.items()}

    for entity in ranked_entities:
        if entity in kg_snap:
            for ctx in entity_contexts.get(entity, []):
                node_contexts.add(f"Source Context ({entity}): {ctx}")
                
            comm_id = entity_communities.get(entity)
            if comm_id:
                retrieved_communities.add(comm_id)
            
            neighbors = set(nx.single_source_shortest_path_length(kg_un_snap, entity, cutoff=2).keys())
            for n in neighbors:
                for u, v, data in kg_snap.out_edges(n, data=True):
                    retrieved_edges.add(f"{u} --[{data.get('label', 'related_to')}]--> {v}")
                for u, v, data in kg_snap.in_edges(n, data=True):
                    retrieved_edges.add(f"{u} --[{data.get('label', 'related_to')}]--> {v}")

    community_context = []
    for comm_id in retrieved_communities:
        lbl = community_labels_snap.get(comm_id, "General Concept Base")
        members = community_map_snap.get(comm_id, [])
        community_context.append(f"Sub-Network Identity Cluster [{comm_id} -> Summary Topic: {lbl}] encompasses: {', '.join(members[:10])}")

    context_str = "MACRO TOPOLOGY AUTOMATED CLUSTER LABELS:\n" + "\n".join(community_context) + "\n\nLOCAL HYBRID HYPER-GRAPHS RETRIEVAL (2-HOP BOUNDED):\n" + "\n".join(retrieved_edges) + "\n\nDEEP DOCUMENT TRACKING EXCERPTS:\n" + "\n".join(node_contexts)
    context_str = context_str.replace("<|im_start|>", "").replace("<|im_end|>", "")

    rag_prompt = (
        "<|im_start|>system\n"
        "Synthesize an accurate response using the verified GraphRAG multi-hop structural context map.\n"
        "Anchor your answer in the provided semantic clusters, graph paths, and document snippets.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context_str}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
    
    stop_event = threading.Event()
    
    def generate_with_stop(inputs_dict, t_streamer, s_event):
        with MODEL_INFERENCE_LOCK:
            with torch.inference_mode():
                model.generate(
                    **inputs_dict, 
                    streamer=t_streamer, 
                    max_new_tokens=450, 
                    do_sample=False, 
                    pad_token_id=tokenizer.pad_token_id
                )
        s_event.set()
            
    gen_thread = threading.Thread(target=generate_with_stop, args=(inputs, streamer, stop_event))
    gen_thread.daemon = True
    gen_thread.start()
    
    partial_answer = ""
    for new_token in streamer:
        partial_answer += new_token
        yield partial_answer, context_str
        if stop_event.is_set():
            break

# ==========================================
# 7. GRADIO WEB UI INTERFACE MOUNT
# ==========================================
app = FastAPI()

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# 🚀 GraphRAG v11.1 Enterprise: Hierarchical Multi-Modal Engine")
    
    with gr.Tabs():
        with gr.Tab("1. System Data Ingestion"):
            with gr.Row():
                file_upload = gr.File(label="Target Source Carrier (PDF, DOCX, CSV, MP4, MP3, HTML)")
                text_input = gr.Textbox(lines=5, label="Raw Text Intercept Entry Point")
            ingest_btn = gr.Button("Execute GLiNER Extraction & Graph Synthesis", variant="primary")
            ingest_logs = gr.Textbox(lines=8, label="Infrastructure Processing Execution Tracing Metrics")
            
        with gr.Tab("2. Stratified Hybrid RAG Exploration"):
            query_input = gr.Textbox(label="Target Query Objective")
            query_btn = gr.Button("Initialize BM25 + Vector Search Path Execution", variant="primary")
            answer_output = gr.Textbox(label="Cross-Encoder Reranked Synthesis Output", lines=8)
            context_output = gr.Textbox(label="Materialized Multi-Hop Graph Retrieval Payload Trace", lines=12)
                
        with gr.Tab("3. Topological Graph Spatial Analysis"):
            graph_html = gr.HTML(value="<h3 style='text-align:center;padding:20px;color:gray;'>Awaiting spatial visualization triggers... click refresh.</h3>")
            refresh_btn = gr.Button("Re-Read Storage View Graph Matrices Layer")

    ingest_btn.click(fn=ingest_data, inputs=[text_input, file_upload], outputs=[ingest_logs, graph_html], queue=True)
    query_btn.click(fn=query_graph_stream, inputs=[query_input], outputs=[answer_output, context_output], queue=True)
    refresh_btn.click(fn=generate_graph_html, inputs=[], outputs=[graph_html], queue=True)

app = gr.mount_gradio_app(app, demo, path="/")

async def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    
    if NGROK_TOKEN and NGROK_TOKEN != "YOUR_NGROK_AUTHTOKEN_HERE" and NGROK_TOKEN.strip() != "":
        try:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(8000)
            logger.info(f"\n🚀 SOTA PRODUCTION SEAMLESS TUNNEL ENGINE ONLINE: {public_url.public_url}\n")
        except Exception as e:
            logger.warning(f"ngrok routing allocation failed (Internet proxy bypass error?): {e}")
            logger.info("Serving fallback locally on port 8000 only.")
            
    await server.serve()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3)
    
    if os.path.exists(KG_PATH):
        try:
            KG = nx.read_graphml(KG_PATH)
            
            if os.path.exists(CONTEXT_PATH):
                with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                    ctx_data = json.load(f)
                for n in KG.nodes:
                    KG.nodes[n]["context"] = ctx_data.get(str(n), [])
            else:
                for n in KG.nodes:
                    KG.nodes[n]["context"] = []
                    
            KG_UNDIRECTED = KG.to_undirected()
            DISPLAY_KG = KG.copy()
            
            PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
            for n in DISPLAY_KG.nodes:
                c_id = DISPLAY_KG.nodes[n].get("community_id", "Cluster_0")
                try: cluster_num = int(c_id.split("_")[1])
                except: cluster_num = 0
                DISPLAY_KG.nodes[n]["color"] = PALETTE[cluster_num % len(PALETTE)]
                DISPLAY_KG.nodes[n].pop("context", None)
            
            for n in KG.nodes:
                norm_name = str(n).lower().strip()
                KNOWN_ENTITIES.add(norm_name)
                ENTITY_LOWER_TO_ORIGINAL[norm_name] = n
            
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(build_bm25_index_isolated())
            except RuntimeError:
                asyncio.run(build_bm25_index_isolated())
            
            for n in KG.nodes:
                c_id = KG.nodes[n].get('community_id')
                if c_id:
                    if c_id not in COMMUNITY_MAP: COMMUNITY_MAP[c_id] = []
                    COMMUNITY_MAP[c_id].append(n)
                    
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(run_community_auto_labeling())
            except RuntimeError:
                asyncio.run(run_community_auto_labeling())
            
            logger.info("Successfully synchronized database footprints back into working active cache.")
        except Exception as e:
            logger.error(f"Cold boot restoration skipped: {e}")
            
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(start_server())
        logger.info("Attached to existing Event Loop (Notebook Environment).")
    except RuntimeError:
        asyncio.run(start_server())
        logger.info("Created new Event Loop (Terminal Environment).")
