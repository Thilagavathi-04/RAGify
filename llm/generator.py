# llm/generator.py
"""
Standalone LLM generator with hybrid Ollama + Groq support.
Uses the same get_llm() factory from chain.py.
"""

from langchain_core.messages import HumanMessage
from llm.chain import get_llm


class Generator:
    def __init__(self, provider=None, model_name=None, temperature=0):
        """
        Initialize the LLM generator.
        Args:
            provider: "ollama" or "groq" (None = default from .env).
            model_name: Specific model name (overrides default).
            temperature: LLM temperature.
        """
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=temperature)

    def generate(self, prompt):
        """
        Generate a response using the LLM.
        Args:
            prompt: The full prompt string including context.
        Returns:
            The generated answer string.
        """
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content
