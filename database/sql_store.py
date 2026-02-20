# database/sql_store.py
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "metadata.db")


class SQLStore:
    def __init__(self, db_name=None):
        db = db_name or DB_PATH
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            doc_type TEXT,
            classification TEXT
        )
        """)
        self.conn.commit()

    def insert_document(self, source, doc_type, classification):
        self.conn.execute(
            "INSERT INTO documents (source, doc_type, classification) VALUES (?, ?, ?)",
            (source, doc_type, classification)
        )
        self.conn.commit()

    def get_all_documents(self):
        """Retrieve all stored document metadata."""
        cursor = self.conn.execute("SELECT * FROM documents")
        return [dict(row) for row in cursor.fetchall()]

    def get_documents_by_type(self, doc_type):
        """Retrieve documents filtered by type."""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE doc_type = ?", (doc_type,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_document_by_source(self, source):
        """Retrieve documents filtered by source."""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE source = ?", (source,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, doc_id):
        """Delete a document by its ID."""
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()

    def delete_documents_by_source(self, source):
        """Delete all document metadata rows matching the given source name."""
        self.conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
