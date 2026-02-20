# retrieval/hybrid_retriever.py
import re
import numpy as np
from collections import Counter


class HybridRetriever:
    """
    Advanced retriever with 4 strategies:

    1. BM25 Keyword Search    — exact term matching with TF-IDF weighting
    2. Cosine Similarity      — angular distance between embeddings (better than L2 for text)
    3. MMR (Maximal Marginal Relevance) — relevance + diversity (avoids redundant results)
    4. Hybrid Fusion          — weighted combination of vector + BM25 + cosine scores

    Each strategy is best suited for different input formats.
    All strategies support an optional `type_filter` parameter that pre-filters
    chunks by document type BEFORE ranking, so that minority types (email, image,
    audio, html, json) are not drowned out by dominant types (pdf).
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    # =============================================================
    # Helper: Pre-filter chunks by document type
    # =============================================================
    def _get_filtered_chunks(self, type_filter=None):
        """
        Return chunks (and their indices) that match the given type filter.
        If type_filter is None, returns all chunks.

        Returns:
            List of (original_index, chunk_text, metadata) tuples.
        """
        if type_filter is None:
            return [
                (i, text, meta)
                for i, (text, meta) in enumerate(self.vector_store.chunks)
            ]
        type_filter = type_filter.lower().strip()
        return [
            (i, text, meta)
            for i, (text, meta) in enumerate(self.vector_store.chunks)
            if meta.get("type", "").lower().strip() == type_filter
        ]

    # =============================================================
    # Strategy 1: BM25 Keyword Search
    # Best for: JSON, structured data, exact lookups
    # =============================================================
    def retrieve_bm25(self, query, top_k=5, type_filter=None):
        """
        Pure BM25 keyword retrieval over stored chunks.
        Best for structured data (JSON) where exact term matches matter.

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            type_filter: Optional doc type to pre-filter chunks (e.g. "json").

        Returns:
            List of (chunk_text, metadata) tuples ranked by BM25 score.
        """
        filtered = self._get_filtered_chunks(type_filter)
        if not filtered:
            return []

        query_tokens = self._tokenize(query)
        all_doc_tokens = [self._tokenize(text) for _, text, _ in filtered]
        avg_dl = sum(len(dt) for dt in all_doc_tokens) / len(all_doc_tokens) if all_doc_tokens else 1

        # Number of documents containing each term (for IDF)
        n_docs = len(all_doc_tokens)
        doc_freqs = Counter()
        for doc_tokens in all_doc_tokens:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1

        scored = []
        for i, (_, chunk, metadata) in enumerate(filtered):
            score = self._bm25_score_with_idf(
                query_tokens, all_doc_tokens[i], avg_dl, doc_freqs, n_docs
            )
            scored.append((score, chunk, metadata))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(chunk, metadata) for _, chunk, metadata in scored[:top_k]]

    # =============================================================
    # Strategy 2: Cosine Similarity Search
    # Best for: PDF, audio — natural prose where direction matters
    # =============================================================
    def retrieve_cosine(self, query, top_k=5, type_filter=None):
        """
        Cosine similarity retrieval using embedding vectors.
        Better than L2 for text because it measures angular distance,
        making it invariant to embedding magnitude.

        When type_filter is set, only chunks of that type are considered,
        preventing dominant types from drowning out minority types.

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            type_filter: Optional doc type to pre-filter chunks (e.g. "pdf", "audio").

        Returns:
            List of (chunk_text, metadata) tuples ranked by cosine similarity.
        """
        filtered = self._get_filtered_chunks(type_filter)
        if not filtered:
            return []

        # Embed the query
        query_vec = self.vector_store.embed(query)

        # Reconstruct embeddings only for filtered chunks and compute cosine
        scored = []
        for orig_idx, chunk_text, metadata in filtered:
            try:
                embedding = self.vector_store.index.reconstruct(orig_idx)
            except RuntimeError:
                continue
            cosine_sim = self._cosine_similarity(query_vec, embedding)
            scored.append((cosine_sim, chunk_text, metadata))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(chunk, metadata) for _, chunk, metadata in scored[:top_k]]

    # =============================================================
    # Strategy 3: MMR (Maximal Marginal Relevance)
    # Best for: HTML, email — avoids redundant/duplicate results
    # =============================================================
    def retrieve_mmr(self, query, top_k=5, lambda_param=0.7, type_filter=None):
        """
        MMR retrieval balances relevance and diversity.
        Prevents returning multiple chunks that say the same thing.

        When type_filter is set, only chunks of that type are considered.

        Formula: MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            lambda_param: Balance between relevance (1.0) and diversity (0.0).
                          Higher = more relevant, Lower = more diverse.
            type_filter: Optional doc type to pre-filter chunks (e.g. "html", "email").

        Returns:
            List of (chunk_text, metadata) tuples ranked by MMR score.
        """
        filtered = self._get_filtered_chunks(type_filter)
        if not filtered:
            return []

        # Embed the query
        query_vec = self.vector_store.embed(query)

        # Reconstruct embeddings for filtered chunks and compute cosine to query
        candidate_data = []
        for orig_idx, chunk_text, metadata in filtered:
            try:
                embedding = self.vector_store.index.reconstruct(orig_idx)
            except RuntimeError:
                continue
            sim_to_query = self._cosine_similarity(query_vec, embedding)
            candidate_data.append({
                "chunk": chunk_text,
                "metadata": metadata,
                "embedding": embedding,
                "sim_to_query": sim_to_query
            })

        if not candidate_data:
            return []

        # Greedy MMR selection
        selected = []
        remaining = list(range(len(candidate_data)))

        while len(selected) < top_k and remaining:
            best_idx = None
            best_mmr = -float("inf")

            for i in remaining:
                relevance = candidate_data[i]["sim_to_query"]

                # Max similarity to any already-selected document
                if selected:
                    max_sim_to_selected = max(
                        self._cosine_similarity(
                            candidate_data[i]["embedding"],
                            candidate_data[s]["embedding"]
                        )
                        for s in selected
                    )
                else:
                    max_sim_to_selected = 0.0

                # MMR score
                mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * max_sim_to_selected)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [
            (candidate_data[i]["chunk"], candidate_data[i]["metadata"])
            for i in selected
        ]

    # =============================================================
    # Strategy 4: Hybrid Fusion (BM25 + Cosine + Vector Rank)
    # Best for: general queries, mixed content
    # =============================================================
    def retrieve_hybrid(self, query, top_k=5,
                        vector_weight=0.4, cosine_weight=0.35, keyword_weight=0.25,
                        type_filter=None):
        """
        Full hybrid retrieval combining three signals:
        - Vector rank (FAISS L2 ordering)
        - Cosine similarity (angular embedding distance)
        - BM25 keyword score (term frequency matching)

        When type_filter is set, only chunks of that type are considered.

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            vector_weight: Weight for FAISS vector rank score.
            cosine_weight: Weight for cosine similarity score.
            keyword_weight: Weight for BM25 keyword score.
            type_filter: Optional doc type to pre-filter chunks (e.g. "image").

        Returns:
            List of (chunk_text, metadata) tuples ranked by fused score.
        """
        filtered = self._get_filtered_chunks(type_filter)
        if not filtered:
            return []

        # Embed the query
        query_vec = self.vector_store.embed(query)

        # Reconstruct embeddings and compute L2 distances for filtered chunks
        candidates = []
        for orig_idx, chunk_text, metadata in filtered:
            try:
                embedding = self.vector_store.index.reconstruct(orig_idx)
            except RuntimeError:
                continue
            distance = float(np.linalg.norm(query_vec - embedding))
            candidates.append((chunk_text, metadata, embedding, distance))

        if not candidates:
            return []

        # Sort by L2 distance first (for vector rank signal)
        candidates.sort(key=lambda x: x[3])
        # Trim to reasonable candidate count
        candidate_count = min(top_k * 3, len(candidates))
        candidates = candidates[:candidate_count]

        query_tokens = self._tokenize(query)
        all_doc_tokens = [self._tokenize(chunk) for chunk, _, _, _ in candidates]
        avg_dl = sum(len(dt) for dt in all_doc_tokens) / len(all_doc_tokens) if all_doc_tokens else 1

        scored = []
        for i, (chunk_text, metadata, embedding, distance) in enumerate(candidates):
            # Signal 1: Vector rank (reciprocal rank)
            vector_score = 1.0 / (i + 1)

            # Signal 2: Cosine similarity
            cosine_sim = self._cosine_similarity(query_vec, embedding)
            # Normalize to 0-1 range (cosine can be -1 to 1)
            cosine_normalized = (cosine_sim + 1) / 2

            # Signal 3: BM25 keyword score
            bm25 = self._bm25_score(query_tokens, all_doc_tokens[i], avg_dl)
            max_possible = len(query_tokens) * 2.5
            bm25_normalized = min(bm25 / max_possible, 1.0) if max_possible > 0 else 0

            # Fused score
            combined = (
                vector_weight * vector_score +
                cosine_weight * cosine_normalized +
                keyword_weight * bm25_normalized
            )
            scored.append((combined, chunk_text, metadata))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(chunk, metadata) for _, chunk, metadata in scored[:top_k]]

    # =============================================================
    # Helper: Cosine Similarity
    # =============================================================
    def _cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two vectors."""
        vec_a = np.array(vec_a, dtype="float32").flatten()
        vec_b = np.array(vec_b, dtype="float32").flatten()
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # =============================================================
    # Helper: BM25 Score (basic)
    # =============================================================
    def _bm25_score(self, query_tokens, doc_tokens, avg_dl, k1=1.5, b=0.75):
        """Compute BM25 score without IDF."""
        doc_len = len(doc_tokens)
        doc_freq = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            tf = doc_freq.get(token, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            score += numerator / denominator if denominator > 0 else 0
        return score

    # =============================================================
    # Helper: BM25 Score with IDF
    # =============================================================
    def _bm25_score_with_idf(self, query_tokens, doc_tokens, avg_dl,
                              doc_freqs, n_docs, k1=1.5, b=0.75):
        """Compute BM25 score with IDF weighting for better ranking."""
        doc_len = len(doc_tokens)
        doc_freq = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            tf = doc_freq.get(token, 0)
            df = doc_freqs.get(token, 0)

            # IDF: log((N - df + 0.5) / (df + 0.5))
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            score += idf * (numerator / denominator) if denominator > 0 else 0
        return score

    # =============================================================
    # Helper: Tokenizer
    # =============================================================
    def _tokenize(self, text):
        """Simple word tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())
