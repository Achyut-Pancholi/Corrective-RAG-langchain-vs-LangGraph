import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCUMENTS_DIR

def load_and_split_documents():
    """Loads all text files from the data/documents directory and splits them."""
    if not os.path.exists(DOCUMENTS_DIR):
        raise FileNotFoundError(f"Documents directory not found at {DOCUMENTS_DIR}")

    loader = DirectoryLoader(str(DOCUMENTS_DIR), glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits
