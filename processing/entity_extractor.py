# processing/entity_extractor.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def extract_entities(text):
    prompt = f"""
    Extract key entities (Names, Dates, Money, Organizations).
    Return JSON.

    Text:
    {text[:2000]}
    """

    llm = ChatOllama(model="mistral", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content
