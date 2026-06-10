# GraphRAG: Multi-Modal Knowledge Graph & Retrieval Engine

A Retrieval-Augmented Generation (RAG) system that combines dense vector search, sparse keyword indexing, and knowledge graph topologies to process and query user-provided documents. The system builds an explorable network of entities and relationships, allowing for multi-hop reasoning and macro-level concept retrieval.

## Features

* **Multi-Modal Parsing:** Ingests and processes standard text documents (PDF, DOCX, CSV, HTML, TXT, MD) and media files (MP4, MP3, WAV) using local transcription models.
* **Hybrid Retrieval System:** Combines semantic vector search (LanceDB) with lexical keyword matching (BM25) to identify relevant context.
* **Cross-Encoder Reranking:** Applies a secondary scoring layer to retrieved candidates to improve query-to-context precision.
* **Zero-Shot Entity Extraction:** Utilizes a dedicated Named Entity Recognition (NER) model (GLiNER) for fast, deterministic entity identification before relationship mapping.
* **Automated Community Detection:** Uses the Louvain algorithm to group graph nodes into sub-networks, which are then summarized and labeled by the LLM.
* **Interactive Visualization:** Renders the active knowledge graph structure in a local, interactive HTML viewport.
* **Local Persistence:** Saves vector embeddings, graph topology (GraphML), and document context (JSON) locally to maintain state across sessions.
* **Asynchronous UI:** Built on FastAPI and Gradio, utilizing thread executors and state locks to safely handle concurrent requests.

## How the Workflow Operates

The system pipeline is divided into two primary phases: Data Ingestion and Query Retrieval.

### Phase 1: Data Ingestion & Graph Construction
1. **Parsing & Chunking:** Uploaded files are parsed into raw text. The text is divided into chunks using a sentence-aware sliding window to ensure entities and context are not split across boundaries.
2. **Entity Extraction:** The GLiNER model scans each chunk to extract named entities (Persons, Organizations, Locations, Concepts, Artifacts).
3. **Relationship Mapping:** Chunks containing multiple entities are passed to the primary LLM, which extracts explicit factual relationships in a structured triplet format (Subject -> Relation -> Object).
4. **Graph Updating & Storage:** Triplets are appended to the global directed graph. Duplicate nodes are automatically merged. New entities are embedded and stored in the LanceDB vector database.
5. **Community Clustering:** The graph undergoes Louvain community detection. The top nodes of each new cluster are passed to the LLM to generate a macro-level summary label for that specific sub-network.
6. **Indexing:** The BM25 lexical corpus is rebuilt to include all newly discovered entities.

### Phase 2: Stratified Query Retrieval
1. **Initial Search:** A user query is embedded into a vector and sent to LanceDB for semantic similarity matching. Simultaneously, the query is tokenized and run against the BM25 index for exact entity matches. 
2. **Reranking:** The combined candidate pool is evaluated by a Cross-Encoder, which scores the entities against the original user query. The top-scoring entities are selected as the retrieval anchors.
3. **Graph Traversal:** The system locates the anchor entities within the Knowledge Graph and executes a 2-hop breadth-first search. It collects:
   * Direct relationships and edges surrounding the anchors.
   * The original raw document excerpts associated with those nodes.
   * The macro-community labels those nodes belong to.
4. **Synthesis:** The aggregated topological data, document excerpts, and community summaries are compiled into a single context string. This is fed to the LLM, which streams the final synthesized answer back to the user interface.

## Installation

Ensure you have Python 3.10+ installed. If you intend to process audio or video files, `ffmpeg` must be installed on your system.

```bash
# System dependency for media processing (Debian/Ubuntu)
!apt-get update && apt-get install -y ffmpeg


# Install Python dependencies
pip install networkx pyvis lancedb sentence-transformers gradio fastapi uvicorn pyngrok pypdf python-docx beautifulsoup4 openai-whisper moviepy loguru gliner rank_bm25 bitsandbytes accelerate
