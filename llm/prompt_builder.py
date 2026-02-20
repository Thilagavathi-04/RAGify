# llm/prompt_builder.py
from langchain_core.prompts import ChatPromptTemplate


# LangChain prompt template for RAG
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
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


class PromptBuilder:
    def __init__(self):
        self.template = RAG_PROMPT_TEMPLATE

    def build(self, query, context_chunks):
        """
        Build a grounded RAG prompt from query and retrieved chunks.
        Args:
            query: The user's question.
            context_chunks: List of (chunk_text, metadata) tuples or plain strings.
        Returns:
            The formatted prompt string.
        """
        # Handle both tuple format (from VectorStore) and plain strings
        extracted = []
        for item in context_chunks:
            if isinstance(item, tuple):
                extracted.append(item[0])
            else:
                extracted.append(str(item))

        context = "\n\n".join(extracted)

        # Format via LangChain template and return as a string
        messages = self.template.format_messages(
            context=context,
            question=query
        )

        # Convert LangChain messages back to a single prompt string
        # This keeps compatibility with the existing Generator.generate() API
        parts = []
        for msg in messages:
            parts.append(msg.content)
        return "\n\n".join(parts)

    def build_chain_input(self, query, context_chunks):
        """
        Build the input dict for a LangChain chain.
        Args:
            query: The user's question.
            context_chunks: List of (chunk_text, metadata) tuples or plain strings.
        Returns:
            Dict with 'context' and 'question' keys.
        """
        extracted = []
        for item in context_chunks:
            if isinstance(item, tuple):
                extracted.append(item[0])
            else:
                extracted.append(str(item))

        return {
            "context": "\n\n".join(extracted),
            "question": query
        }
