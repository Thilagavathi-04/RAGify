# ingestion/csv_loader.py
import csv
import logging

logger = logging.getLogger(__name__)


def load_csv(file_path):
    """
    Load a CSV file and convert each row into a readable text chunk.
    Each row becomes: "column1: value1 | column2: value2 | ..."
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.error(f"Failed to read CSV '{file_path}': {e}")
        return []

    if not rows:
        logger.warning(f"No rows found in '{file_path}'")
        return []

    # Convert each row to a readable string
    lines = []
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items() if v]
        if parts:
            lines.append(" | ".join(parts))

    text = "\n".join(lines)

    return [{
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "csv",
            "rows": len(rows),
            "columns": list(rows[0].keys()) if rows else []
        }
    }]
