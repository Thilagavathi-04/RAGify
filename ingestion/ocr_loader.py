# ingestion/ocr_loader.py
import logging
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(file_path):
    """Extract text from an image using OCR (Tesseract)."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"Failed to OCR image '{file_path}': {e}")
        return []

    if not text.strip():
        logger.warning(f"No text found in image '{file_path}'")
        return []

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "image"
        }
    }]
