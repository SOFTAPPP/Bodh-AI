import os
from dotenv import load_dotenv

# Get the backend directory (where .env lives)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    INDEXED_FILES_LOG = os.path.join(DATA_DIR, "indexed_files.txt")

    # ── Groq Model Strategy ─────────────────────────────────────────────────
    # Primary: best versatile throughput at ultra-low latency
    GENERATION_MODEL = "llama-3.3-70b-versatile"
    # Reasoning / complex analysis: deep-think distilled model
    REASONING_MODEL  = "deepseek-r1-distill-llama-70b"
    # Fallback: lightest, fastest model when primary is rate-limited
    FALLBACK_MODEL   = "gemma2-9b-it"

    # ── Embedding Model (local, zero API cost) ──────────────────────────────
    # all-MiniLM-L6-v2 → 384-dim, ~14ms/query on CPU, excellent semantic accuracy
    EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG Settings (optimized for 10-100+ page docs) ──────────────────────
    CHUNK_SIZE        = 1000   # tokens (sweet spot: 800-1200)
    CHUNK_OVERLAP     = 200    # tokens (sweet spot: 150-250)
    TOP_K             = 12     # chunks retrieved per query
    SIMILARITY_THRESHOLD = 0.25  # min cosine score to include a chunk

    # ── Query Cache ─────────────────────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD = 0.98  # cosine sim for cache hit

# Ensure directories exist
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
