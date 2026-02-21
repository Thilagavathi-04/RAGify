# RAG_AI — Sample Markdown Document

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that enhances large language model (LLM) responses by retrieving relevant information from external knowledge sources before generating an answer.

## Architecture Overview

The RAG pipeline consists of three stages:

1. **Ingestion** — Documents are loaded, cleaned, chunked, and stored in a vector database.
2. **Retrieval** — When a user asks a question, the most relevant chunks are fetched using similarity search.
3. **Generation** — The retrieved chunks are passed as context to the LLM, which generates a grounded answer.

## Supported Document Formats

| Format | Loader |
|--------|--------|
| PDF | PyPDF2 |
| Word | python-docx |
| PowerPoint | python-pptx |
| CSV | stdlib csv |
| SQLite | stdlib sqlite3 |

## Code Example

```python
from main import ingest
ingest("document.pdf")
```

---

*This markdown file is used to test the Markdown ingestion loader in the RAG_AI pipeline.*
