import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# LLM Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# Retrieval Config
TOP_K = int(os.getenv("TOP_K", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
