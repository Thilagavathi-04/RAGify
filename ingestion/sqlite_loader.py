# ingestion/sqlite_loader.py
import sqlite3
import logging

logger = logging.getLogger(__name__)


def load_sqlite(file_path):
    """
    Load all tables from a SQLite database.
    Each table becomes a document with rows formatted as readable text.
    """
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Failed to open SQLite database '{file_path}': {e}")
        return []

    documents = []

    try:
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            logger.warning(f"No tables found in '{file_path}'")
            conn.close()
            return []

        for table in tables:
            try:
                cursor.execute(f'SELECT * FROM "{table}"')
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                if not rows:
                    continue

                # Format each row as "col1: val1 | col2: val2 | ..."
                lines = []
                for row in rows:
                    parts = [f"{col}: {val}" for col, val in zip(columns, row) if val is not None]
                    if parts:
                        lines.append(" | ".join(parts))

                if lines:
                    header = f"Table: {table} ({len(rows)} rows, {len(columns)} columns)\n"
                    header += f"Columns: {', '.join(columns)}\n\n"
                    text = header + "\n".join(lines)

                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": file_path,
                            "type": "sqlite",
                            "table": table,
                            "rows": len(rows),
                            "columns": columns
                        }
                    })

            except Exception as e:
                logger.warning(f"Failed to read table '{table}': {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to query SQLite database '{file_path}': {e}")
    finally:
        conn.close()

    if not documents:
        logger.warning(f"No data found in '{file_path}'")

    return documents
