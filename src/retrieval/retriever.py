from src.retrieval.vector_store import get_vector_store
from src.config import TOP_K

def get_retriever():
    """Returns a pre-configured retriever based on the vector store."""
    vectorstore = get_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
