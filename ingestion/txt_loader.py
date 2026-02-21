# ingestion/txt_loader.py
import os
import logging

logger = logging.getLogger(__name__)


def load_txt(file_path):
    """Load text from a plain text (.txt) or Markdown (.md) file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Failed to read text file '{file_path}': {e}")
        return []

    if not text.strip():
        logger.warning(f"No text content found in '{file_path}'")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    doc_type = "markdown" if ext == ".md" else "txt"

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": doc_type
        }
    }]
