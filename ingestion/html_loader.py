# ingestion/html_loader.py
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def load_html(file_path):
    """Load and extract text content from an HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        logger.error(f"Failed to read HTML '{file_path}': {e}")
        return []

    text = soup.get_text(separator=" ", strip=True)

    if not text.strip():
        logger.warning(f"No text content found in '{file_path}'")
        return []

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "html"
        }
    }]
