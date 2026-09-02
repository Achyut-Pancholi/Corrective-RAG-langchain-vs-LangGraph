import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCUMENTS_DIR

def load_and_split_documents():
    """Loads all text and PDF files from the data/documents directory and splits them."""
    if not os.path.exists(DOCUMENTS_DIR):
        raise FileNotFoundError(f"Documents directory not found at {DOCUMENTS_DIR}")

    # Load TXT files
    txt_loader = DirectoryLoader(str(DOCUMENTS_DIR), glob="**/*.txt", loader_cls=TextLoader)
    txt_documents = txt_loader.load()

    # Load PDF files
    pdf_loader = DirectoryLoader(str(DOCUMENTS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    all_docs = txt_documents + pdf_documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    return splits
