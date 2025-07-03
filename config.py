import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"  # Make sure this matches exactly what you have
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval parameters
SIMILARITY_THRESHOLD = 0.7
MAX_RETRIEVED_CHUNKS = 5

# Streamlit configuration
PAGE_TITLE = "Personal Knowledge Assistant"
PAGE_ICON = "🧠"