# 📄 Local RAG Agent

A local Retrieval-Augmented Generation (RAG) system that accepts CSV, processes the content, and enables contextual question answering using **LLaMA 3.2 via Ollama**, **Ollama Embeddings**, and **ChromaDB**, all through an interactive **Gradio** interface.

---

## 🚀 Features

- 🔍 **Contextual Q&A**: Ask questions about the document and get accurate, grounded responses from the local LLM.
- 💡 **Fully Local Pipeline**: No internet or API calls required—runs entirely offline using Ollama and ChromaDB.
- ⚡ **Fast Retrieval**: Vector similarity search ensures fast and relevant document chunk matching.
- 🖥️ **Interactive UI**: Clean Gradio interface for uploading files and querying them in real-time.

---

## 🛠 Tech Stack

- **Language Model**: [LLaMA 3.2](https://ollama.com/library/llama3) via Ollama  
- **Embeddings**: Ollama Embeddings  
- **Vector Store**: ChromaDB  
- **Frontend**: Gradio  
- **Programming Language**: Python  

---

## 🧠 Architecture Overview

CSV Upload  →  Text Extraction & Chunking  →  Ollama Embeddings (Vectorization)  →  ChromaDB (Vector Store for Retrieval)  →  User Query  →  Relevant Chunks Retrieved  →  LLaMA 3.2 (Local LLM Inference)  →  Answer (Context-Aware Response)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/local-pdf-rag-agent.git
cd local-pdf-rag-agent
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```
3. **Start Ollama and download the model**

First you need to download Ollama from their official [website](https://ollama.com/).
Now open your terminal or command prompt 

```bash
ollama run llama3.2
```

Before running the project make sure Ollama is running in the background.

To run the project, from the root directory : 

```bash
uv run python src/main.py           # for running on terminal
uv run python src/gradio_app.py     # for running the gradio app 
```

---

## 🛠 Troubleshooting: Environment Setup Issues

If you encounter the following warning:

> [!WARNING]
> Mismatched Virtual Environment

This typically means the current Python environment doesn't match the expected `.venv`.

### ✅ Fix Options:

1. **Create a fresh virtual environment with `uv` (recommended)**

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Then run the app with
```bash
uv run python src/main.py
```
or
```bash
uv run --active python src/main.py
```

---


## 🧪 Example Use Cases

1. Upload a research paper and ask for summaries or definitions.

2. Upload a policy document and ask for specific clauses.

3. Upload a user manual and ask how a feature works.

4. Upload financial statements and ask key insights.

