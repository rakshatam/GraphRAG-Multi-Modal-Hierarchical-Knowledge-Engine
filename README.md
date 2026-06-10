# 🚀 GraphRAG Styled Multi-Modal Hierarchical Knowledge Engine

A **GraphRAG Styled** implementation that bridges the gap between unstructured multi-modal data and structured reasoning. Unlike standard VectorRAG, which relies on simple semantic similarity, this engine leverages **Community Detection (Louvain)**, **2-Hop Traversal**, and **Deep Context Synthesis** to retrieve information exactly how humans think: through relationships and macro-concepts.

---

### 🌟 Key Features

* **Multi-Modal Ingestion:** Ingest any document or media. Automatically transcribe audio/video (Whisper), parse PDFs/DOCX, and convert them into a structured knowledge graph.
* **GraphRAG Retrieval:** Moves beyond 1-hop vector search. Our retrieval logic uses **2-hop undirected traversal** to find connections between entities that aren't semantically similar but are logically linked.
* **Macro-Community Awareness:** Uses **Louvain Community Detection** to cluster your data. When querying, the engine provides the LLM with the macro-context of the community the entity belongs to, not just the local triplet.
* **Production-Hardened:** Built to handle memory limits, binary file edge-cases, and database concurrency. Features automated ANN indexing, memory-efficient Pyvis visualization, and robust error handling.
* **Visualizer:** Interactive force-directed graphs, color-coded by community cluster, rendered directly via embedded IFrames.

---

### 🛠️ The Tech Stack

This engine is built on a high-performance stack optimized for local/edge deployment:

| Layer | Technology |
| :--- | :--- |
| **Language Model** | **Qwen 2.5 3B (Base)** |
| **Vector Database** | **LanceDB** (with Cosine ANN Indexing) |
| **Graph Framework** | **NetworkX** (with Louvain Clustering) |
| **Transcriber** | **OpenAI Whisper** |
| **Orchestration** | **FastAPI + Gradio** |
| **Visualization** | **Pyvis** (Inline HTML rendering) |
| **Video Processing**| **MoviePy** |

---

### 📦 Quick Start

#### Prerequisites
* Python 3.10+
* A CUDA-enabled GPU (recommended)
* [Ngrok Authtoken](https://ngrok.com/) (for remote tunnel access)

#### Requirements Installation
```bash
pip install networkx pyvis lancedb sentence-transformers gradio fastapi uvicorn pyngrok pypdf python-docx beautifulsoup4 openai-whisper moviepy
!apt-get update && apt-get install -y ffmpeg
