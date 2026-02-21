# ingestion/docx_loader.py
import logging
from docx import Document

logger = logging.getLogger(__name__)


def load_docx(file_path):
    """Load text from a Word (.docx) file, paragraph by paragraph."""
    try:
        doc = Document(file_path)
    except Exception as e:
        logger.error(f"Failed to read DOCX '{file_path}': {e}")
        return []

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        logger.warning(f"No text content found in '{file_path}'")
        return []

    text = "\n\n".join(paragraphs)

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "docx"
        }
    }]
