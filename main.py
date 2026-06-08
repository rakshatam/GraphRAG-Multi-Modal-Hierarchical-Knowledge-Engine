import os
import html
import torch
import networkx as nx
from networkx.algorithms import community
from pyvis.network import Network
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI
import gradio as gr
from pyngrok import ngrok
import uvicorn

# --- File Parsing Imports ---
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
import whisper
from moviepy import VideoFileClip

# ==========================================
# 0. CONFIGURATION & MULTIPROCESSING
# ==========================================
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# TODO: Paste your actual Ngrok Authtoken here
NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN_HERE" 
if NGROK_TOKEN != "YOUR_NGROK_AUTHTOKEN_HERE":
    ngrok.set_auth_token(NGROK_TOKEN)

print("Booting Vector & Graph Databases...")
db = lancedb.connect("./lancedb_graphrag_ultimate")
schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 384)), pa.field("entity", pa.string())])
if "knowledge_base" in db.table_names(): db.drop_table("knowledge_base")
tbl = db.create_table("knowledge_base", schema=schema)


KG = nx.DiGraph()
DISPLAY_KG = nx.DiGraph() # Cached display graph to prevent O(N) copy on UI refresh
COMMUNITY_MAP = {}        # O(1) dictionary for macro-community lookups

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Whisper (Audio/Video) and Qwen 2.5 (LLM)...")
audio_model = whisper.load_model("base", device="cuda")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ==========================================
# 1. THE UNIVERSAL PARSER
# ==========================================
def parse_file(file_path):
    ext = file_path.lower().split('.')[-1]
    text = ""
    
    if ext == "pdf":
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif ext == "doc":
        return "ERROR: Legacy .doc format unsupported. Please convert to .docx first."
    elif ext in ["xml", "html"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            text = soup.get_text(separator=' ')
    elif ext in ["txt", "md", "csv"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext in ["mp3", "wav", "m4a"]:
        try:
            result = audio_model.transcribe(file_path)
            text = result["text"]
        except Exception as e:
            return f"ERROR: Audio transcription failed: {e}"
    elif ext in ["mp4", "mkv", "avi"]:
        try:
            with VideoFileClip(file_path) as clip:
                if clip.audio is None:
                    return "ERROR: No audio track found in video."
                clip.audio.write_audiofile("temp_audio.wav", logger=None)
                
            result = audio_model.transcribe("temp_audio.wav")
            text = result["text"]
        except Exception as e:
            return f"ERROR: Video processing failed: {e}"
        finally:
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
    else:
        
        return f"ERROR: Unsupported file type '.{ext}'. Please use PDF, DOCX, TXT, HTML, XML, MP3, or MP4."
                
    return text

def chunk_text(text, words_per_chunk=400):
    words = text.split()
    return [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

# ==========================================
# 2. INGESTION & CLUSTERING ENGINE
# ==========================================
def extract_from_chunk(chunk):
    prompt = (
        "<|im_start|>system\n"
        "You are an elite Knowledge Graph Extraction Engine.\n"
        "RULES:\n"
        "1. Extract highly descriptive entities and factual relationships.\n"
        "2. Keep relations concise (1-3 words max).\n"
        "3. NO INVERSE RELATIONS. If you extract A -> B, do NOT extract B -> A.\n"
        "4. Output ONLY a valid Markdown table.\n"
        "Format: | Subject | Relation | Object |\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Text:\n{chunk}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "| Subject | Relation | Object |\n|---|---|---|\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def safe_add_to_lancedb(entity):
    try:
        tbl.add([{"vector": embedder.encode(entity).tolist(), "entity": entity}])
    except Exception as e:
        print(f"[DB ERROR] Failed to insert '{entity}': {e}")

def update_communities():
    global COMMUNITY_MAP
    if len(KG.nodes) > 1:
        communities = community.louvain_communities(KG.to_undirected(), seed=42)
        COMMUNITY_MAP.clear()
        
        for i, comm in enumerate(communities):
            cluster_name = f"Cluster_{i}"
            # FIX 4: Build the O(1) lookup dictionary
            COMMUNITY_MAP[cluster_name] = list(comm) 
            for node in comm:
                KG.nodes[node]['community_id'] = cluster_name

def ingest_data(raw_text=None, file_obj=None):
    global DISPLAY_KG
    text_to_process = ""
    ux_warning = ""
    
    if file_obj is not None:
        if raw_text:
            ux_warning = "⚠️ FILE PRIORITY: Both file and text provided. Ignoring raw text input.\n\n"
        parsed_result = parse_file(file_obj.name)
        if parsed_result.startswith("ERROR:"):
            return parsed_result, generate_graph_html()
        text_to_process = parsed_result
    elif raw_text:
        text_to_process = raw_text
        
    if not text_to_process.strip():
        return "No text found to process.", generate_graph_html()

    chunks = chunk_text(text_to_process)
    triplets_found = []
    
    for i, chunk in enumerate(chunks):
        generated = extract_from_chunk(chunk)
        for line in generated.split('\n'):
            line = line.strip()
            if line.startswith('|') and line.endswith('|') and line.count('|') >= 4:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 3 and "Subject" not in parts[0] and "---" not in parts[0]:
                    subj, rel, obj = parts[0], parts[1], parts[2]
                    
                    if subj and rel and obj:
                        if subj not in KG: 
                            KG.add_node(subj, context=[])
                            safe_add_to_lancedb(subj)
                        if obj not in KG: 
                            KG.add_node(obj, context=[])
                            safe_add_to_lancedb(obj)
                        
                        KG.nodes[subj]['context'].append(chunk[:200] + "...")
                        KG.nodes[obj]['context'].append(chunk[:200] + "...")
                        KG.add_edge(subj, obj, label=rel)
                        triplets_found.append(f"({subj}) --[{rel}]--> ({obj})")

    update_communities()
    
    
    if tbl.count_rows() >= 256:
        try:
            tbl.create_index(metric="cosine")
        except Exception as e:
            print(f"[INDEX WARNING] {e}")
            
    
    DISPLAY_KG = KG.copy()
    PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for node in DISPLAY_KG.nodes:
        cluster_id = DISPLAY_KG.nodes[node].get('community_id', 'Cluster_0')
        cluster_num = int(cluster_id.split("_")[1])
        DISPLAY_KG.nodes[node]['color'] = PALETTE[cluster_num % len(PALETTE)]
        
        DISPLAY_KG.nodes[node].pop('context', None)
        
    num_communities = len(COMMUNITY_MAP.keys())

    log_output = f"{ux_warning}Processed {len(chunks)} chunks.\nExtracted {len(triplets_found)} relations.\nDetected {num_communities} macro-communities via Louvain.\n\n" + "\n".join(triplets_found)
    return log_output, generate_graph_html()

def generate_graph_html():
    if len(DISPLAY_KG.nodes) == 0: return "<h3 style='color:white;text-align:center;'>Graph is empty.</h3>"
    
    net = Network(notebook=False, directed=True, height="600px", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources='in_line')
    
    # We now read directly from the cached DISPLAY_KG. No O(N) loops or copies occur here.
    net.from_nx(DISPLAY_KG)
    net.repulsion(node_distance=180, spring_length=220)
    
    raw_html = net.generate_html()
    escaped_html = html.escape(raw_html, quote=True)
    return f'<iframe srcdoc="{escaped_html}" width="100%" height="600px" style="border:none;"></iframe>'

# ==========================================
# 3. TRUE GRAPHRAG RETRIEVAL ENGINE
# ==========================================
def query_graph(question):
    if len(KG.nodes) == 0: return "Database empty.", ""

    query_vector = embedder.encode(question).tolist()
    results = tbl.search(query_vector).metric("cosine").limit(3).to_list()
    
    retrieved_edges = set()
    node_contexts = set()
    retrieved_communities = set()
    
    KG_undirected = KG.to_undirected()
    
    for res in results:
        entity = res["entity"]
        if entity in KG:
            for ctx in KG.nodes[entity].get('context', []):
                node_contexts.add(f"Document Excerpt about {entity}: {ctx}")
            
            if 'community_id' in KG.nodes[entity]:
                retrieved_communities.add(KG.nodes[entity]['community_id'])
                
            neighbors = set(nx.single_source_shortest_path_length(KG_undirected, entity, cutoff=2).keys())
            
            for neighbor in neighbors:
                for u, v, data in KG.out_edges(neighbor, data=True):
                    retrieved_edges.add(f"{u} --[{data['label']}]--> {v}")
                for u, v, data in KG.in_edges(neighbor, data=True):
                    retrieved_edges.add(f"{u} --[{data['label']}]--> {v}")

    community_context = []
    for comm_id in retrieved_communities:
        # FIX 4: O(1) execution instead of O(N x C) list comprehension
        members = COMMUNITY_MAP.get(comm_id, [])
        community_context.append(f"{comm_id} contains: {', '.join(members)}")

    context_str = "MACRO COMMUNITY STRUCTURE:\n" + "\n".join(community_context) + "\n\nLOCAL GRAPH TRAVERSAL (2-HOP):\n" + "\n".join(retrieved_edges) + "\n\nDEEP SOURCE CONTEXT:\n" + "\n".join(node_contexts)

    rag_prompt = (
        "<|im_start|>system\n"
        "Answer the user's question using the provided GraphRAG context. "
        "Synthesize the Macro Communities, Graph Traversal, and Source Excerpts intelligently.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context_str}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=350, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer, context_str

# ==========================================
# 4. GRADIO DASHBOARD
# ==========================================
app = FastAPI()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 True GraphRAG: Hierarchical & Multi-Hop Engine")
    
    with gr.Tabs():
        with gr.Tab("1. Ingest Data"):
            gr.Markdown("Upload documents/media. System runs OpenIE, builds graph, and clusters via Louvain Community Detection.")
            with gr.Row():
                file_upload = gr.File(label="Upload File (PDF, DOCX, MP4, MP3, XML)")
                text_input = gr.Textbox(lines=5, label="Or Paste Text Here")
                
            ingest_btn = gr.Button("Process & Build Graph", variant="primary")
            ingest_logs = gr.Textbox(lines=8, label="Extraction & Clustering Logs")
            
        with gr.Tab("2. Query Graph"):
            query_input = gr.Textbox(label="Ask a question")
            query_btn = gr.Button("Search", variant="primary")
            answer_output = gr.Textbox(label="LLM Answer", lines=5)
            context_output = gr.Textbox(label="Retrieval Payload (Communities + O(1) Maps + 2-Hop + Deep Context)", lines=12)
                
        with gr.Tab("3. Visualizer"):
            gr.Markdown("Nodes are automatically colored by their detected macro-community.")
            graph_html = gr.HTML(value=generate_graph_html())
            refresh_btn = gr.Button("Refresh Map")

    ingest_btn.click(fn=ingest_data, inputs=[text_input, file_upload], outputs=[ingest_logs, graph_html])
    query_btn.click(fn=query_graph, inputs=[query_input], outputs=[answer_output, context_output])
    refresh_btn.click(fn=generate_graph_html, inputs=[], outputs=[graph_html])

app = gr.mount_gradio_app(app, demo, path="/")

async def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    if NGROK_TOKEN != "YOUR_NGROK_AUTHTOKEN_HERE":
        public_url = ngrok.connect(8000)
        print(f"\n🚀 TRUE GRAPHRAG ONLINE: {public_url.public_url}\n")
    await server.serve()

await start_server()
