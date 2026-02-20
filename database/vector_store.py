# database/vector_store.py
import os
import json
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings


class VectorStore:
    def __init__(self, dim=768, index_path=None):
        """
        Initialize the vector store using LangChain OllamaEmbeddings + raw FAISS.
        Args:
            dim: Embedding dimension (default 768 for nomic-embed-text).
            index_path: Optional path to persist the FAISS index on disk.
        """
        self.dim = dim
        self.index_path = index_path
        self.chunks_path = index_path + ".chunks.json" if index_path else None

        # LangChain embeddings model (replaces raw ollama.embeddings calls)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Load existing index from disk or create a new one
        if index_path and os.path.exists(index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(index_path)
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.chunks = [tuple(item) for item in json.load(f)]
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.chunks = []

    def embed(self, text):
        """Generate embedding using LangChain OllamaEmbeddings."""
        vector = self.embeddings.embed_query(text)
        return np.array(vector, dtype="float32")

    def add(self, text, metadata):
        """
        Add a text chunk and its metadata to the vector store.
        Args:
            text: The text content to embed and store.
            metadata: Associated metadata dictionary.
        """
        embedding = self.embed(text)
        self.index.add(np.array([embedding]))
        self.chunks.append((text, metadata))

    def save(self):
        """Persist the FAISS index and chunks to disk."""
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)

    def delete_by_source(self, source_name):
        """
        Delete all chunks belonging to a given source file.
        Rebuilds the FAISS index from scratch without those chunks.

        Args:
            source_name: The source filename (e.g. "sample.eml").

        Returns:
            Number of chunks removed.
        """
        # Separate chunks to keep vs remove
        keep = []
        removed = 0
        for text, meta in self.chunks:
            if meta.get("source", "") == source_name:
                removed += 1
            else:
                keep.append((text, meta))

        if removed == 0:
            return 0

        # Rebuild FAISS index with only the kept chunks
        new_index = faiss.IndexFlatL2(self.dim)
        for text, meta in keep:
            embedding = self.embed(text)
            new_index.add(np.array([embedding]))

        self.index = new_index
        self.chunks = keep
        self.save()
        return removed

    def search(self, query, k=3):
        """
        Search for the top-k most similar chunks to the query.
        Args:
            query: The query text string.
            k: Number of results to return.
        Returns:
            List of (chunk_text, metadata) tuples.
        """
        if self.index.ntotal == 0:
            return []

        # Clamp k to available vectors
        k = min(k, self.index.ntotal)

        query_vec = self.embed(query)
        distances, indices = self.index.search(
            np.array([query_vec]), k
        )

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results

    def search_with_scores(self, query, k=3):
        """
        Search and return results with L2 distance scores.
        Args:
            query: The query text string.
            k: Number of results to return.
        Returns:
            List of (chunk_text, metadata, distance) tuples.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)

        query_vec = self.embed(query)
        distances, indices = self.index.search(
            np.array([query_vec]), k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk_text, metadata = self.chunks[idx]
                results.append((chunk_text, metadata, float(distances[0][i])))

        return results

    def search_with_embeddings(self, query, k=3):
        """
        Search and return results with their embedding vectors.
        Used by MMR and cosine similarity strategies.
        Args:
            query: The query text string.
            k: Number of results to return.
        Returns:
            Tuple of (query_embedding, results) where results is a list of
            (chunk_text, metadata, embedding, distance) tuples.
        """
        if self.index.ntotal == 0:
            return None, []

        k = min(k, self.index.ntotal)

        query_vec = self.embed(query)
        distances, indices = self.index.search(
            np.array([query_vec]), k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk_text, metadata = self.chunks[idx]
                # Reconstruct the embedding from FAISS index
                embedding = self.index.reconstruct(int(idx))
                results.append((chunk_text, metadata, embedding, float(distances[0][i])))

        return query_vec, results
