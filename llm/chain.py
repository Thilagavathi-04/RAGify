# llm/chain.py
"""
LangChain RAG Chain — Hybrid LLM (Ollama local + Groq cloud).

Supports two LLM providers:
  1. Ollama  — local (mistral, llama3, etc.) — free, private, slower
  2. Groq    — cloud (llama-3.3-70b, mixtral, etc.) — fast, needs API key

The user can pick the provider per query, or it auto-selects based on config.
If Groq fails (no API key, rate limit), it falls back to Ollama automatically.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# ---- Environment config ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Available Groq models (for UI dropdown)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# Available Ollama models
OLLAMA_MODELS = [
    "mistral",
    "llama3",
    "llama3.2",
    "gemma2",
    "phi3",
]

# ---- Shared prompt template ----
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful AI assistant. "
        "Answer ONLY from the provided context. "
        "If the answer is not in context, say \"I don't know.\""
    )),
    ("human", (
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )),
])


def _build_ollama_llm(model_name=None, temperature=0):
    """Create an Ollama LLM instance."""
    from langchain_ollama import ChatOllama
    model = model_name or OLLAMA_MODEL
    logger.info(f"Using Ollama LLM: {model}")
    return ChatOllama(model=model, temperature=temperature)


def _build_groq_llm(model_name=None, temperature=0):
    """Create a Groq LLM instance."""
    from langchain_groq import ChatGroq
    api_key = GROQ_API_KEY
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set. Add it to the .env file.")
    model = model_name or GROQ_MODEL
    logger.info(f"Using Groq LLM: {model}")
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def get_llm(provider=None, model_name=None, temperature=0):
    """
    Get the LLM based on provider selection.

    Args:
        provider: "ollama", "groq", or None (uses DEFAULT_LLM_PROVIDER).
        model_name: Optional specific model name.
        temperature: LLM temperature.

    Returns:
        A LangChain chat model instance.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower().strip()

    if provider == "groq":
        return _build_groq_llm(model_name, temperature)
    else:
        return _build_ollama_llm(model_name, temperature)


def create_rag_chain(provider=None, model_name=None, temperature=0):
    """
    Create a LangChain RAG chain (prompt → LLM → string output).

    Args:
        provider: "ollama" or "groq".
        model_name: Specific model name (overrides default).
        temperature: LLM temperature.

    Returns:
        A LangChain Runnable chain.
    """
    llm = get_llm(provider=provider, model_name=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    chain = RAG_PROMPT | llm | output_parser
    return chain


class RAGChain:
    """
    High-level RAG chain with hybrid Ollama + Groq support.

    The LLM provider can be chosen per query. If Groq fails,
    it automatically falls back to the local Ollama model.
    """

    def __init__(self, retriever, temperature=0):
        self.retriever = retriever
        self.temperature = temperature

    def query(self, question, top_k=5, doc_type=None, source_filter=None,
              provider=None, model_name=None):
        """
        Full RAG pipeline: retrieve → format → generate.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            doc_type: Optional doc type hint for retrieval routing.
            source_filter: Optional source filename filter.
            provider: "ollama" or "groq" (None = default from .env).
            model_name: Optional specific model name.

        Returns:
            Dict with 'answer', 'sources', 'context', and 'provider_used'.
        """
        # 1. Retrieve
        chunks = self.retriever.retrieve(
            question, top_k=top_k, doc_type=doc_type, source_filter=source_filter
        )

        if not chunks:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "context": "",
                "provider_used": "none",
            }

        # 2. Build context
        context_parts = []
        for item in chunks:
            if isinstance(item, tuple):
                context_parts.append(item[0])
            else:
                context_parts.append(str(item))
        context = "\n\n".join(context_parts)

        chain_input = {"context": context, "question": question}

        # 3. Try primary provider, fall back if it fails
        used_provider = (provider or DEFAULT_LLM_PROVIDER).lower().strip()
        try:
            chain = create_rag_chain(
                provider=used_provider,
                model_name=model_name,
                temperature=self.temperature,
            )
            answer = chain.invoke(chain_input)
        except Exception as primary_err:
            logger.warning(f"{used_provider} failed: {primary_err}")

            # Fallback: if Groq failed → try Ollama, and vice versa
            fallback = "ollama" if used_provider == "groq" else "groq"
            logger.info(f"Falling back to {fallback}...")
            try:
                chain = create_rag_chain(
                    provider=fallback,
                    temperature=self.temperature,
                )
                answer = chain.invoke(chain_input)
                used_provider = f"{fallback} (fallback)"
            except Exception as fallback_err:
                logger.error(f"Fallback also failed: {fallback_err}")
                return {
                    "answer": f"Both LLM providers failed.\n"
                              f"Primary ({used_provider}): {primary_err}\n"
                              f"Fallback ({fallback}): {fallback_err}",
                    "sources": [],
                    "context": context,
                    "provider_used": "error",
                }

        # 4. Format sources
        sources = []
        for chunk_text, metadata in chunks:
            sources.append({
                "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "metadata": metadata,
            })

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "provider_used": used_provider,
        }
