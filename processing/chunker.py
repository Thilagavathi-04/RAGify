# processing/chunker.py
import re
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================
# Available Chunking Strategies
# =============================================================
AVAILABLE_STRATEGIES = [
    "sentence",
    "paragraph",
    "key_value",
    "section",
    "line",
    "semantic",       # LangChain SemanticChunker (embedding-based)
    "character",      # LangChain CharacterTextSplitter
    "recursive",      # LangChain RecursiveCharacterTextSplitter
]

# Default strategy per doc type (used when no explicit strategy is given)
DEFAULT_STRATEGY_MAP = {
    "pdf": "sentence",
    "html": "paragraph",
    "json": "key_value",
    "email": "section",
    "image": "line",
    "audio": "sentence",
    "docx": "paragraph",
    "pptx": "section",
    "txt": "sentence",
    "markdown": "paragraph",
    "xml": "key_value",
    "csv": "line",
    "sqlite": "line",
}


# =============================================================
# Chunking Strategy Router
# =============================================================
def chunk_text(text, doc_type="default", chunk_size=500, overlap=50, strategy=None):
    """
    Routes to the correct chunking strategy.

    Args:
        text: The input text to chunk.
        doc_type: The type of document (pdf, html, json, email, image, audio).
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.
        strategy: Explicit chunking strategy name (overrides doc_type routing).
                  One of AVAILABLE_STRATEGIES.

    Returns:
        List of text chunks.
    """
    doc_type = doc_type.lower().strip()

    # Determine which strategy to use
    chosen = (strategy or "").lower().strip()
    if not chosen:
        chosen = DEFAULT_STRATEGY_MAP.get(doc_type, "sentence")

    logger.info(f"Chunking with strategy '{chosen}' (doc_type={doc_type}, chunk_size={chunk_size})")

    if chosen == "sentence":
        return chunk_by_sentence(text, chunk_size, overlap)
    elif chosen == "paragraph":
        return chunk_by_paragraph(text, chunk_size, overlap)
    elif chosen == "key_value":
        return chunk_by_key_value(text, chunk_size)
    elif chosen == "section":
        return chunk_by_section(text, chunk_size, overlap)
    elif chosen == "line":
        return chunk_by_line(text, chunk_size)
    elif chosen == "semantic":
        return chunk_semantic(text, chunk_size)
    elif chosen == "character":
        return chunk_character(text, chunk_size, overlap)
    elif chosen == "recursive":
        return chunk_recursive(text, chunk_size, overlap)
    else:
        logger.warning(f"Unknown strategy '{chosen}', falling back to sentence.")
        return chunk_by_sentence(text, chunk_size, overlap)


# =============================================================
# Strategy 1: Sentence-Based Chunking (PDF, Audio, Default)
# Best for: well-structured prose with proper punctuation
# =============================================================
def chunk_by_sentence(text, chunk_size=500, overlap=50):
    """Split text at sentence boundaries, grouping into chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())

            # Overlap: keep tail of previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""

        current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


# =============================================================
# Strategy 2: Paragraph-Based Chunking (HTML)
# Best for: web content with natural paragraph breaks
# =============================================================
def chunk_by_paragraph(text, chunk_size=500, overlap=50):
    """Split text at paragraph boundaries (double newlines or <br> gaps)."""
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If no paragraph breaks found, fall back to sentence chunking
    if len(paragraphs) <= 1:
        return chunk_by_sentence(text, chunk_size, overlap)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())

            # Overlap: keep last paragraph
            if overlap > 0:
                current_chunk = para
            else:
                current_chunk = ""

        current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


# =============================================================
# Strategy 3: Key-Value Chunking (JSON)
# Best for: structured data — each top-level key becomes a chunk
# =============================================================
def chunk_by_key_value(text, chunk_size=500):
    """Split JSON into chunks per top-level key or array item."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # If it's not valid JSON, fall back to sentence chunking
        return chunk_by_sentence(text, chunk_size)

    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            entry = json.dumps({key: value}, indent=2)
            # If a single entry is too large, split it further
            if len(entry) > chunk_size:
                sub_chunks = chunk_by_sentence(entry, chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(entry)

    elif isinstance(data, list):
        current_chunk = ""
        for item in data:
            entry = json.dumps(item, indent=2)
            if len(current_chunk) + len(entry) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += "\n" + entry if current_chunk else entry

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    else:
        chunks = [text]

    return chunks if chunks else [text]


# =============================================================
# Strategy 4: Section-Based Chunking (Email)
# Best for: emails with headers, greeting, body, signature
# =============================================================
def chunk_by_section(text, chunk_size=500, overlap=50):
    """Split email text by sections: headers, body paragraphs, signature."""
    # Try to separate common email sections
    sections = re.split(
        r'\n\s*(?:---+|___+|===+)\s*\n'   # horizontal rules
        r'|\n(?=(?:From:|To:|Subject:|Date:|Cc:|Bcc:)\s)',  # email headers
        text
    )
    sections = [s.strip() for s in sections if s and s.strip()]

    # If no sections found, split by paragraphs
    if len(sections) <= 1:
        return chunk_by_paragraph(text, chunk_size, overlap)

    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += "\n\n" + section if current_chunk else section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


# =============================================================
# Strategy 5: Line-Based Chunking (OCR / Image)
# Best for: OCR output with irregular line breaks, no punctuation
# =============================================================
def chunk_by_line(text, chunk_size=500):
    """Split OCR text by lines, grouping lines into chunks."""
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += "\n" + line if current_chunk else line

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


# =============================================================
# Strategy 6: Semantic Chunking (LangChain — Embedding-based)
# Best for: long prose where you want to split at meaning shifts.
# Uses OllamaEmbeddings (nomic-embed-text) to find natural breaks.
# =============================================================
def chunk_semantic(text, chunk_size=500):
    """
    Split text using LangChain SemanticChunker.
    Groups sentences by embedding similarity — chunks stay semantically coherent.
    Falls back to recursive splitting if the embedding model is unavailable.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",   # split at large gaps
            breakpoint_threshold_amount=75,            # 75th-percentile gap
        )

        docs = splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs if doc.page_content.strip()]

        if not chunks:
            logger.warning("Semantic chunker produced no chunks, falling back to recursive.")
            return chunk_recursive(text, chunk_size)

        # Post-process: if any chunk is way too large, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size * 3:
                sub = chunk_recursive(chunk, chunk_size)
                final_chunks.extend(sub)
            else:
                final_chunks.append(chunk)

        logger.info(f"Semantic chunker produced {len(final_chunks)} chunks")
        return final_chunks

    except Exception as e:
        logger.warning(f"Semantic chunking failed ({e}), falling back to recursive.")
        return chunk_recursive(text, chunk_size)


# =============================================================
# Strategy 7: Character Text Splitter (LangChain)
# Best for: simple fixed-size splitting by character count
# =============================================================
def chunk_character(text, chunk_size=500, overlap=50):
    """Split text using LangChain CharacterTextSplitter."""
    try:
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return chunks if chunks else [text]

    except Exception as e:
        logger.warning(f"CharacterTextSplitter failed ({e}), falling back to sentence.")
        return chunk_by_sentence(text, chunk_size, overlap)


# =============================================================
# Strategy 8: Recursive Character Text Splitter (LangChain)
# Best for: hierarchical splitting (\n\n → \n → space → char)
# =============================================================
def chunk_recursive(text, chunk_size=500, overlap=50):
    """Split text using LangChain RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return chunks if chunks else [text]

    except Exception as e:
        logger.warning(f"RecursiveCharacterTextSplitter failed ({e}), falling back to sentence.")
        return chunk_by_sentence(text, chunk_size, overlap)
