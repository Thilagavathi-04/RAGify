# main.py

import os
import logging
from typing import List, Dict

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Ingestion Loaders
# -----------------------------
# Ingestion Loaders (imported lazily inside route_loader)
# -----------------------------
from ingestion.pdf_loader import load_pdf
from ingestion.html_loader import load_html
from ingestion.ocr_loader import load_image
from ingestion.json_loader import load_json
from ingestion.email_loader import load_email
from ingestion.audio_loader import load_audio
from ingestion.txt_loader import load_txt
from ingestion.xml_loader import load_xml
from ingestion.csv_loader import load_csv

# -----------------------------
# Processing
# -----------------------------
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from processing.classifier import classify_document
from processing.entity_extractor import extract_entities

# -----------------------------
# Databases
# -----------------------------
from database.vector_store import VectorStore
from database.sql_store import SQLStore


# -----------------------------
# Initialize Databases
# -----------------------------
vector_db = VectorStore(index_path="database/faiss.index")
sql_db = SQLStore()


# -----------------------------
# File Router
# -----------------------------
def route_loader(file_path: str) -> List[Dict]:
    """
    Routes file to appropriate loader based on extension.
    Returns a list of document dictionaries:
    {
        "content": "...",
        "metadata": {
            "source": "...",
            "type": "..."
        }
    }
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)

    elif ext == ".html":
        return load_html(file_path)

    elif ext in [".png", ".jpg", ".jpeg"]:
        return load_image(file_path)

    elif ext == ".json":
        return load_json(file_path)

    elif ext == ".eml":
        return load_email(file_path)

    elif ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        return load_audio(file_path)

    elif ext == ".docx":
        from ingestion.docx_loader import load_docx
        return load_docx(file_path)

    elif ext == ".pptx":
        from ingestion.pptx_loader import load_pptx
        return load_pptx(file_path)

    elif ext in [".txt", ".md"]:
        return load_txt(file_path)

    elif ext == ".xml":
        return load_xml(file_path)

    elif ext == ".csv":
        return load_csv(file_path)

    elif ext in [".db", ".sqlite", ".sqlite3"]:
        from ingestion.sqlite_loader import load_sqlite
        return load_sqlite(file_path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


# -----------------------------
# Ingestion Pipeline
# -----------------------------
def ingest(file_path: str, vector_store=None, sql_store=None, chunk_strategy=None):
    """
    Full ingestion pipeline:
    1. Load file
    2. Clean text
    3. Combine all pages into one document
    4. Chunk the full document
    5. Classify document (once)
    6. Extract entities (once)
    7. Store ONE metadata row in SQL
    8. Store chunks in Vector DB
    9. Persist vector index to disk

    Args:
        file_path: Path to the file to ingest.
        vector_store: Optional VectorStore instance (reuse from caller).
        sql_store: Optional SQLStore instance (reuse from caller).
        chunk_strategy: Optional chunking strategy name (e.g. "semantic",
                        "recursive", "character"). None = auto by doc type.
    """
    # Use provided stores or fall back to module-level defaults
    vs = vector_store or vector_db
    ss = sql_store or sql_db

    logger.info(f"Starting ingestion for: {file_path}")

    # Step 1: Load
    logger.info(f"[Step 1/9] Loading document...")
    try:
        documents = route_loader(file_path)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return

    if not documents:
        logger.warning("No documents found!")
        return

    logger.info(f"[Step 1/9] Loaded {len(documents)} page(s)")

    # Step 2: Clean all pages/sections
    logger.info(f"[Step 2/9] Cleaning text...")
    cleaned_parts = []
    doc_type = "unknown"
    source_name = os.path.basename(file_path)

    for i, doc in enumerate(documents):
        raw_text = doc.get("content", "")
        cleaned = clean_text(raw_text)
        if cleaned.strip():
            cleaned_parts.append(cleaned)
        doc_type = doc.get("metadata", {}).get("type", doc_type)
        logger.info(f"  Cleaned page {i + 1}/{len(documents)}")

    if not cleaned_parts:
        logger.warning("No content after cleaning, skipping.")
        return

    logger.info(f"[Step 2/9] Cleaned {len(cleaned_parts)} page(s)")

    # Step 3: Combine all pages into one full text
    full_text = "\n\n".join(cleaned_parts)
    logger.info(f"[Step 3/9] Combined {len(cleaned_parts)} page(s) into {len(full_text)} chars")

    # Step 4: Chunk the full document (strategy by doc type)
    strategy_label = chunk_strategy or f"auto ({doc_type})"
    logger.info(f"[Step 4/9] Chunking text (strategy: {strategy_label})...")
    all_chunks = chunk_text(full_text, doc_type=doc_type, strategy=chunk_strategy)
    logger.info(f"[Step 4/9] Created {len(all_chunks)} chunks")
    for i, chunk in enumerate(all_chunks):
        logger.info(f"  Chunk {i + 1}/{len(all_chunks)}: {len(chunk)} chars — \"{chunk[:80]}...\"")

    if not all_chunks:
        logger.warning("No chunks created, skipping.")
        return

    # Step 5: Classify document (once, using first 2000 chars)
    logger.info(f"[Step 5/9] Classifying document with AI...")
    try:
        classification = classify_document(full_text[:2000])
        logger.info(f"[Step 5/9] Classification: {classification[:100]}")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        classification = "unknown"

    # Step 6: Extract entities (once, using first 2000 chars)
    logger.info(f"[Step 6/9] Extracting entities with AI...")
    try:
        entities = extract_entities(full_text[:2000])
        logger.info(f"[Step 6/9] Entities: {entities[:200]}...")
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        entities = "{}"

    # Step 7: Store ONE metadata row in SQL (not per page!)
    logger.info(f"[Step 7/9] Saving metadata to SQL...")
    ss.insert_document(
        source=source_name,
        doc_type=doc_type,
        classification=classification
    )
    logger.info(f"[Step 7/9] Metadata saved for {source_name}")

    # Step 8: Store chunks in Vector DB
    logger.info(f"[Step 8/9] Embedding {len(all_chunks)} chunks into vector store...")
    metadata = {"source": source_name, "type": doc_type}
    for i, chunk in enumerate(all_chunks):
        vs.add(text=chunk, metadata=metadata)
        logger.info(f"  Embedded chunk {i + 1}/{len(all_chunks)}")

    # Step 9: Persist vector index to disk
    logger.info(f"[Step 9/9] Saving vector index to disk...")
    vs.save()

    logger.info(f"✅ Ingestion Complete! {len(all_chunks)} chunks stored for {source_name}")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    file_to_ingest = "ADS.pdf"   # Change this to your file
    ingest(file_to_ingest)
