# ingestion/json_loader.py
import json
import logging

logger = logging.getLogger(__name__)


def load_json(file_path):
    """Load and stringify a JSON file for ingestion."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON '{file_path}': {e}")
        return []

    text = json.dumps(data, indent=2)

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "json"
        }
    }]
