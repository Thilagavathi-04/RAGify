import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional

# Your modules
from database.vector_store import VectorStore
from database.sql_store import SQLStore
from retrieval.retriever import Retriever
from llm.chain import (
    RAGChain,
    GROQ_MODELS, OLLAMA_MODELS,
    DEFAULT_LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL, GROQ_API_KEY,
)


# ----------------------------
# Initialize App
# ----------------------------
app = FastAPI(title="RAG_AI API")

# Serve static UI files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------
# Initialize Core Components
# ----------------------------

# Initialize vector store with persistence (FAISS + LangChain embeddings)
vector_store = VectorStore(index_path="database/faiss.index")

# Initialize SQL store
sql_store = SQLStore()

# Initialize your custom retriever (keeps all routing logic)
retriever = Retriever(vector_store=vector_store)

# Initialize LangChain RAG chain (hybrid Ollama + Groq)
rag_chain = RAGChain(retriever=retriever)


# ----------------------------
# Request Model
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    doc_type: Optional[str] = None     # Optional: pdf, html, json, email, image, audio
    source_filter: Optional[str] = None  # Optional: filter by document source name
    provider: Optional[str] = None     # Optional: "ollama" or "groq"
    model_name: Optional[str] = None   # Optional: specific model name


# ----------------------------
# UI: Serve Frontend
# ----------------------------
@app.get("/")
def serve_ui():
    """Serve the main UI page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")


# ----------------------------
# Favicon (prevents 404)
# ----------------------------
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ----------------------------
# Upload & Ingest Endpoint
# ----------------------------
@app.post("/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    chunk_strategy: Optional[str] = None,
):
    """Upload a file and ingest it into the RAG pipeline.
    
    Optional query param: ?chunk_strategy=semantic|recursive|character|sentence|paragraph|...
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run ingestion pipeline — pass our own stores so the data
        # is immediately available for queries without a restart
        from main import ingest
        ingest(file_path, vector_store=vector_store, sql_store=sql_store,
               chunk_strategy=chunk_strategy)

        return {
            "status": "success",
            "message": f"File '{file.filename}' ingested successfully.",
            "filename": file.filename,
            "chunk_strategy": chunk_strategy or "auto",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Chunking Strategies Endpoint
# ----------------------------
@app.get("/strategies")
def list_strategies():
    """List available chunking strategies."""
    from processing.chunker import AVAILABLE_STRATEGIES, DEFAULT_STRATEGY_MAP
    return {
        "strategies": AVAILABLE_STRATEGIES,
        "defaults_by_type": DEFAULT_STRATEGY_MAP,
    }


# ----------------------------
# RAG Query Endpoint
# ----------------------------
@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        # Use LangChain RAG chain with hybrid LLM support
        result = rag_chain.query(
            question=request.question,
            top_k=5,
            doc_type=request.doc_type,
            source_filter=request.source_filter,
            provider=request.provider,
            model_name=request.model_name,
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "provider_used": result.get("provider_used", "unknown"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Document Metadata Endpoint
# ----------------------------
@app.get("/documents")
def list_documents():
    """List all ingested document metadata."""
    try:
        docs = sql_store.get_all_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Delete Document Endpoint
# ----------------------------
@app.delete("/documents/{source}")
def delete_document(source: str):
    """
    Delete a document and all its chunks by source filename.
    Removes from both the SQL metadata store and the FAISS vector store.
    """
    try:
        # 1. Remove chunks from vector store (and rebuild index)
        chunks_removed = vector_store.delete_by_source(source)

        # 2. Remove metadata rows from SQL
        sql_store.delete_documents_by_source(source)

        # 3. Remove the uploaded file if it exists
        upload_path = os.path.join(UPLOAD_DIR, source)
        if os.path.exists(upload_path):
            os.remove(upload_path)

        if chunks_removed == 0:
            return {
                "status": "warning",
                "message": f"No chunks found for '{source}', but SQL metadata was cleaned up.",
            }

        return {
            "status": "success",
            "message": f"Deleted '{source}': {chunks_removed} chunks removed.",
            "chunks_removed": chunks_removed,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# LLM Models Endpoint
# ----------------------------
@app.get("/models")
def list_models():
    """List available LLM providers and their models."""
    import llm.chain as chain_module
    current_key = chain_module.GROQ_API_KEY
    groq_available = bool(current_key and current_key != "your_groq_api_key_here")
    return {
        "default_provider": chain_module.DEFAULT_LLM_PROVIDER,
        "providers": [
            {
                "name": "ollama",
                "available": True,
                "default_model": chain_module.OLLAMA_MODEL,
                "models": chain_module.OLLAMA_MODELS,
            },
            {
                "name": "groq",
                "available": groq_available,
                "default_model": chain_module.GROQ_MODEL,
                "models": chain_module.GROQ_MODELS,
            },
        ],
    }


# ----------------------------
# Settings: Update Groq API Key
# ----------------------------
class SettingsRequest(BaseModel):
    groq_api_key: Optional[str] = None


@app.post("/settings")
def update_settings(request: SettingsRequest):
    """Update settings like the Groq API key at runtime."""
    import llm.chain as chain_module

    if request.groq_api_key is not None:
        chain_module.GROQ_API_KEY = request.groq_api_key

        # Also persist to .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        try:
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    lines = f.readlines()
                with open(env_path, "w") as f:
                    found = False
                    for line in lines:
                        if line.startswith("GROQ_API_KEY="):
                            f.write(f"GROQ_API_KEY={request.groq_api_key}\n")
                            found = True
                        else:
                            f.write(line)
                    if not found:
                        f.write(f"\nGROQ_API_KEY={request.groq_api_key}\n")
        except Exception:
            pass  # Non-critical — runtime update is enough

        return {"status": "success", "message": "Groq API key updated."}

    return {"status": "no_changes"}
