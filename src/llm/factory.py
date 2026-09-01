from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import GROQ_API_KEY, LLM_MODEL

def get_llm():
    """Returns the configured Groq LLM."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in your .env file.")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )

def get_embeddings():
    """Returns local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
