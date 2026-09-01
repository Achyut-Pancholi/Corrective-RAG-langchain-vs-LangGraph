from langchain_chroma import Chroma
from src.llm.factory import get_embeddings
from src.ingestion.loader import load_and_split_documents

def get_vector_store():
    """Initializes the local Chroma vector store from documents in-memory for the POC."""
    embeddings = get_embeddings()
    splits = load_and_split_documents()
    
    # We build it in memory for the POC to ensure it stays fresh and matches the documents on disk.
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
