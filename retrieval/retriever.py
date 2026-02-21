# retrieval/retriever.py
from retrieval.hybrid_retriever import HybridRetriever


class Retriever:
    """
    Smart retriever that auto-selects the best retrieval strategy
    based on the document type of the source content.

    4 Strategies available:
      1. Cosine Similarity  — angular distance, best for natural prose (pdf, audio)
      2. MMR                — relevance + diversity, best for web/email (html, email)
      3. BM25 Keyword       — exact term matching, best for structured data (json)
      4. Hybrid Fusion      — combined cosine + BM25 + vector rank (default, image)

    Routing by doc_type:
      - pdf       → Cosine Similarity (prose with proper sentences)
      - audio     → Cosine Similarity (transcribed natural speech)
      - docx      → Cosine Similarity (Word docs — structured prose)
      - txt       → Cosine Similarity (plain text prose)
      - markdown  → Cosine Similarity (Markdown notes)
      - html      → MMR (web pages have redundant sections, need diversity)
      - email     → MMR (headers/signatures repeat, need diverse body chunks)
      - pptx      → MMR (slides have repetitive headers/footers)
      - json      → BM25 Keyword (structured data needs exact term matches)
      - xml       → BM25 Keyword (tagged structured data)
      - csv       → BM25 Keyword (tabular row-based data)
      - sqlite    → BM25 Keyword (database table rows)
      - image     → Hybrid Fusion (noisy OCR benefits from all signals)
      - default   → Hybrid Fusion (balanced approach for unknown types)
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.hybrid = HybridRetriever(vector_store)

    # -----------------------------------------
    # Public: auto-routing retrieve
    # -----------------------------------------
    def retrieve(self, query, top_k=5, doc_type=None, source_filter=None):
        """
        Retrieve relevant chunks, auto-selecting strategy by doc_type.

        When doc_type is specified, chunks are PRE-FILTERED by type before
        ranking so that non-PDF types (email, image, audio, html, json) are
        not drowned out by the dominant PDF chunks.

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            doc_type: Optional document type hint (pdf, html, json, email, image, audio).
                      If None, uses hybrid fusion across ALL chunks.
            source_filter: Optional source filename to filter results (e.g. "ADS.pdf").

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        if doc_type:
            doc_type = doc_type.lower().strip()

        # Fetch more results if filtering by source (some may be filtered out)
        fetch_k = top_k * 3 if source_filter else top_k

        # PDF → Cosine Similarity (well-structured prose)
        if doc_type == "pdf":
            results = self.hybrid.retrieve_cosine(query, fetch_k, type_filter="pdf")

        # Audio → Cosine Similarity (natural speech transcripts)
        elif doc_type == "audio":
            results = self.hybrid.retrieve_cosine(query, fetch_k, type_filter="audio")

        # Word docs → Cosine Similarity (structured prose)
        elif doc_type == "docx":
            results = self.hybrid.retrieve_cosine(query, fetch_k, type_filter="docx")

        # Plain text → Cosine Similarity
        elif doc_type == "txt":
            results = self.hybrid.retrieve_cosine(query, fetch_k, type_filter="txt")

        # Markdown → Cosine Similarity
        elif doc_type == "markdown":
            results = self.hybrid.retrieve_cosine(query, fetch_k, type_filter="markdown")

        # HTML → MMR (web pages have redundant boilerplate, need diversity)
        elif doc_type == "html":
            results = self.hybrid.retrieve_mmr(query, fetch_k, lambda_param=0.7, type_filter="html")

        # Email → MMR with more diversity (headers/sigs repeat across chunks)
        elif doc_type == "email":
            results = self.hybrid.retrieve_mmr(query, fetch_k, lambda_param=0.6, type_filter="email")

        # PowerPoint → MMR (slides have repetitive headers/footers)
        elif doc_type == "pptx":
            results = self.hybrid.retrieve_mmr(query, fetch_k, lambda_param=0.7, type_filter="pptx")

        # JSON → BM25 Keyword (structured data needs exact key/value matches)
        elif doc_type == "json":
            results = self.hybrid.retrieve_bm25(query, fetch_k, type_filter="json")

        # XML → BM25 Keyword (tagged structured data)
        elif doc_type == "xml":
            results = self.hybrid.retrieve_bm25(query, fetch_k, type_filter="xml")

        # CSV → BM25 Keyword (tabular row-based data)
        elif doc_type == "csv":
            results = self.hybrid.retrieve_bm25(query, fetch_k, type_filter="csv")

        # SQLite → BM25 Keyword (database table rows)
        elif doc_type == "sqlite":
            results = self.hybrid.retrieve_bm25(query, fetch_k, type_filter="sqlite")

        # Image → Hybrid Fusion (noisy OCR benefits from all signals)
        elif doc_type == "image":
            results = self.hybrid.retrieve_hybrid(
                query, fetch_k,
                vector_weight=0.3, cosine_weight=0.4, keyword_weight=0.3,
                type_filter="image"
            )

        # Default → Hybrid Fusion (balanced for unknown types)
        else:
            results = self.hybrid.retrieve_hybrid(
                query, fetch_k,
                vector_weight=0.4, cosine_weight=0.35, keyword_weight=0.25
            )

        # Apply source filter if specified
        if source_filter:
            source_filter = source_filter.strip()
            results = [
                (text, meta) for text, meta in results
                if meta.get("source", "") == source_filter
            ]

        return results[:top_k]
