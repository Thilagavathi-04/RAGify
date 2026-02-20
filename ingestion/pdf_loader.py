# ingestion/pdf_loader.py
import logging
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(file_path):
    """Load text from a PDF file, page by page."""
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        logger.error(f"Failed to read PDF '{file_path}': {e}")
        return []

    text_pages = []

    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_number + 1}: {e}")
            continue

        if text and text.strip():
            text_pages.append({
                "content": text,
                "metadata": {
                    "page": page_number + 1,
                    "source": file_path,
                    "type": "pdf"
                }
            })

    return text_pages
