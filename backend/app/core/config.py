import os
from dotenv import load_dotenv

# Get the backend directory (where .env lives)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    INDEXED_FILES_LOG = os.path.join(DATA_DIR, "indexed_files.txt")
    
    # Model Configs
    # Models (Using GPT-4o-mini exclusively for optimal balance of speed and logic)
    REASONING_MODEL = "gpt-4o-mini"
    GENERATION_MODEL = "gpt-4o-mini"
    FALLBACK_MODEL = "gpt-4o-mini"
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 15 

# Ensure directories exist
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
