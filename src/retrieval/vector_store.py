import os
from langchain_chroma import Chroma
from src.llm.factory import get_embeddings
from src.ingestion.loader import load_and_split_documents
from src.config import CHROMA_PERSIST_DIR

_VECTOR_STORE = None

def get_vector_store():
    """Initializes or loads the persistent local Chroma vector store."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    embeddings = get_embeddings()

    # If already indexed on disk, load directly (instant)
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        _VECTOR_STORE = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings
        )
        return _VECTOR_STORE

    # Otherwise build once and persist to disk
    splits = load_and_split_documents()
    _VECTOR_STORE = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )
    return _VECTOR_STORE
