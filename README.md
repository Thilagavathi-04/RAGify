# 📋 RAG_AI — Retrieval-Augmented Generation System

A modular **RAG (Retrieval-Augmented Generation)** pipeline built in Python. It ingests documents from **6 formats**, applies **8 smart chunking strategies**, stores them in a **FAISS vector database**, and answers questions using a **hybrid LLM layer** (local Ollama + cloud Groq with automatic fallback).

Comes with a **Streamlit UI** and a **FastAPI REST API**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Powered-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal)

---

## ✨ Features

- **Multi-format ingestion** — PDF, HTML, JSON, Email (.eml), Image (OCR), Audio (Whisper)
- **8 chunking strategies** — 5 custom (sentence, paragraph, key-value, section, line) + 3 LangChain-based (semantic, character, recursive) — auto-selected per document type
- **4 retrieval strategies** — Cosine similarity, MMR, BM25 keyword search, and Hybrid Fusion — auto-routed by doc type with `type_filter` support
- **Hybrid LLM** — Ollama (local, free) + Groq (cloud, fast) with per-query provider selection and automatic fallback
- **Streamlit UI** — Chat, Upload, Document management, and Settings in a single app
- **FastAPI REST API** — Full CRUD + interactive Swagger docs
- **Document management** — Upload, list, and delete documents from the knowledge base
- **LLM-powered processing** — Document classification and entity extraction via Ollama

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Required |
| **Ollama** | Must be running locally — pull `mistral` and `nomic-embed-text` models |
| **Groq API key** | Optional — for cloud LLM (get one free at [console.groq.com](https://console.groq.com)) |
| **Tesseract OCR** | Only needed for image ingestion (`sudo apt install tesseract-ocr`) |

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/RAG_AI.git
cd RAG_AI
python -m venv venv
source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here   # Optional — leave empty for Ollama-only mode
DEFAULT_LLM_PROVIDER=ollama            # "ollama" or "groq"
OLLAMA_MODEL=mistral
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Start Ollama & Pull Models

```bash
ollama serve
ollama pull mistral            # LLM for classification & entity extraction
ollama pull nomic-embed-text   # Embedding model for vector store
```

### 4. Run the Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```

Open 🔗 **[http://localhost:8501](http://localhost:8501)** — the UI has 4 pages:

| Page | Description |
|---|---|
| 💬 **Ask Question** | Chat interface — pick LLM provider/model, optional doc type filter |
| 📤 **Upload Document** | Upload any supported file — auto-ingested into the pipeline |
| 📚 **Documents** | View and **delete** ingested documents |
| ⚙️ **Settings** | Switch default LLM provider/model, view Groq API key status |

### 5. Or run the FastAPI server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open 🔗 **[http://localhost:8000](http://localhost:8000)** for the HTML/JS UI, or **[http://localhost:8000/docs](http://localhost:8000/docs)** for interactive API docs.

---

## 🏗️ Architecture

### Data Flow

```
                    ┌──────────────────────────────────────────────────────┐
                    │               INGESTION PIPELINE                     │
                    │                                                      │
  [Document]  ──►  Loader  ──►  Cleaner  ──►  Smart Chunker  ──►         │
  (.pdf/.html/       │                     (auto by doc_type)    │         │
   .json/.eml/       │                            │              │         │
   .png/.mp3)        │                  ┌─────────┴──────────┐   │         │
                     │                  ▼                    ▼   │         │
                     │            Classifier          Entity     │         │
                     │                  │            Extractor    │         │
                     │                  ▼                │       │         │
                     │            SQL Store ◄────────────┘       │         │
                     │          (metadata.db)                    │         │
                     │                                           │         │
                     │            Vector Store ◄── chunks        │         │
                     │          (faiss.index)                    │         │
                    └──────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────────┐
                    │               QUERY PIPELINE                         │
                    │                                                      │
  [User Query] ──►  Streamlit / API  ──►  Smart Retriever  ──►  RAGChain │
  + doc_type          │                  (auto-selects by        (Ollama   │
  + provider          │                   doc_type + filter)     or Groq)  │
                     │                          │                   │      │
                     │                          ▼                   ▼      │
                     │                    [Answer + Sources]               │
                    └──────────────────────────────────────────────────────┘
```

### Module Breakdown

#### 📁 `ingestion/` — Document Loaders (6 formats)

| File | Format | Library |
|---|---|---|
| `pdf_loader.py` | `.pdf` | PyPDF2 |
| `html_loader.py` | `.html` | BeautifulSoup4 |
| `ocr_loader.py` | `.png`, `.jpg`, `.jpeg` | Pytesseract + Pillow |
| `json_loader.py` | `.json` | stdlib `json` |
| `email_loader.py` | `.eml` | stdlib `email` |
| `audio_loader.py` | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | OpenAI Whisper |

#### 📁 `processing/` — Text Processing (4 modules)

| File | Purpose |
|---|---|
| `cleaner.py` | Removes control characters, normalizes whitespace |
| `chunker.py` | **8 chunking strategies** — auto-selects by doc type (see below) |
| `classifier.py` | LLM-based classification (Invoice / Resume / Legal / Research / Other) |
| `entity_extractor.py` | LLM-based entity extraction (Names, Dates, Money, Orgs) |

**Chunking strategies (auto-selected by `doc_type`):**

| Doc Type | Strategy | Why |
|---|---|---|
| `pdf` | Sentence | Well-structured prose with punctuation |
| `html` | Paragraph | Natural `\n\n` paragraph breaks |
| `json` | Key-value | Each top-level key → one chunk |
| `email` | Section | Headers / body / signature sections |
| `image` | Line | OCR output has irregular lines |
| `audio` | Sentence | Transcribed speech has natural sentences |

Plus 3 LangChain-based strategies available via API: **semantic** (embedding-based), **character**, and **recursive**.

#### 📁 `database/` — Storage

| File | Purpose | Technology |
|---|---|---|
| `vector_store.py` | Embedding storage, similarity search, delete, disk persistence | FAISS + `nomic-embed-text` |
| `sql_store.py` | Document metadata CRUD | SQLite (`database/metadata.db`) |

#### 📁 `retrieval/` — 4 Retrieval Strategies

| # | Strategy | Best For |
|---|---|---|
| 1 | **Cosine Similarity** | Natural prose (PDF, audio) |
| 2 | **MMR** (Maximal Marginal Relevance) | Redundant content (HTML, email) |
| 3 | **BM25 Keyword Search** | Structured data (JSON) |
| 4 | **Hybrid Fusion** (vector + cosine + BM25) | Noisy content (OCR/image), general queries |

Auto-routed by `doc_type` with `type_filter` pre-filtering to avoid cross-type noise.

#### 📁 `llm/` — Hybrid LLM Layer

| File | Purpose |
|---|---|
| `chain.py` | `RAGChain` with Ollama + Groq support, per-query provider selection, automatic fallback |
| `generator.py` | Direct LLM generation (used by classifier & entity extractor) |
| `prompt_builder.py` | RAG prompt construction with context injection |

**Supported models:**

| Provider | Models |
|---|---|
| **Ollama** (local) | mistral, llama3, llama3.2, gemma2, phi3 |
| **Groq** (cloud) | llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it, mixtral-8x7b-32768 |

#### 📁 `api/` — FastAPI REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | RAG query (question, doc_type, provider, model) |
| `/upload` | POST | Upload & ingest a document |
| `/documents` | GET | List all ingested documents |
| `/documents/{source}` | DELETE | Delete a document from both stores |
| `/models` | GET | List available LLM providers & models |
| `/settings` | POST | Update default LLM provider/model |
| `/strategies` | GET | List available chunking strategies |

---

## 📁 Project Structure

```
RAG_AI/
├── streamlit_app.py           # Streamlit UI (primary frontend)
├── main.py                    # Ingestion pipeline orchestrator
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys — not in git)
├── pyproject.toml             # Project metadata
├── README.md
│
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI server + HTML/JS UI
│
├── ingestion/
│   ├── pdf_loader.py          # PDF → text
│   ├── html_loader.py         # HTML → text
│   ├── ocr_loader.py          # Image → text (OCR)
│   ├── json_loader.py         # JSON → text
│   ├── email_loader.py        # .eml → text
│   └── audio_loader.py        # Audio → text (Whisper)
│
├── processing/
│   ├── cleaner.py             # Text cleaning
│   ├── chunker.py             # 8 chunking strategies
│   ├── classifier.py          # Document classification
│   └── entity_extractor.py    # Entity extraction
│
├── database/
│   ├── vector_store.py        # FAISS vector store
│   ├── sql_store.py           # SQLite metadata store
│   ├── faiss.index            # Persisted FAISS index (not in git)
│   └── metadata.db            # SQLite database (not in git)
│
├── retrieval/
│   ├── retriever.py           # Smart retriever (auto-selects strategy)
│   └── hybrid_retriever.py    # BM25 + Cosine + MMR + Hybrid engine
│
├── llm/
│   ├── chain.py               # Hybrid Ollama/Groq RAG chain
│   ├── generator.py           # LLM generation
│   └── prompt_builder.py      # Prompt construction
│
├── ui/                        # HTML/JS frontend (served by FastAPI)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── test_data/                 # Sample files for testing
└── uploads/                   # User-uploaded files (not in git)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| LLM (local) | Ollama (mistral, llama3) |
| LLM (cloud) | Groq (llama-3.3-70b-versatile) |
| Embeddings | Ollama `nomic-embed-text` |
| Vector DB | FAISS (faiss-cpu) |
| Metadata DB | SQLite |
| Orchestration | LangChain |
| PDF | PyPDF2 |
| HTML | BeautifulSoup4 |
| OCR | Pytesseract + Pillow |
| Audio | OpenAI Whisper |

---

## 📡 API Usage Examples

```bash
# Ask a question (defaults to Ollama)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# Ask with Groq cloud LLM
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key points", "provider": "groq"}'

# Filter by document type
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total?", "doc_type": "json"}'

# Upload a file
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# List documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/document.pdf

# Get available models
curl http://localhost:8000/models

# Get chunking strategies
curl http://localhost:8000/strategies
```

---

