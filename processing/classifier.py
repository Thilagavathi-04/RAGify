# processing/classifier.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def classify_document(text):
    prompt = f"""
    Classify this document into:
    - Invoice
    - Resume
    - Legal
    - Research Paper
    - Other

    Text:
    {text[:2000]}
    """

    llm = ChatOllama(model="mistral", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content
