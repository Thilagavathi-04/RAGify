# ingestion/xml_loader.py
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _extract_text(element, depth=0):
    """Recursively extract text from an XML element tree."""
    parts = []

    tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

    text = (element.text or "").strip()
    if text:
        parts.append(f"{tag}: {text}")

    for child in element:
        parts.extend(_extract_text(child, depth + 1))

    tail = (element.tail or "").strip()
    if tail:
        parts.append(tail)

    return parts


def load_xml(file_path):
    """Load and extract text content from an XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to read XML '{file_path}': {e}")
        return []

    lines = _extract_text(root)
    text = "\n".join(lines)

    if not text.strip():
        logger.warning(f"No text content found in '{file_path}'")
        return []

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "xml"
        }
    }]
