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
    # Fallback: lightest, fastest model when primary is rate-limited
    # Note: gemma2-9b-it was decommissioned by Groq; using llama-3.1-8b-instant instead
    FALLBACK_MODEL   = "llama-3.1-8b-instant"

    # ── Embedding Model (local, zero API cost) ──────────────────────────────
    # all-MiniLM-L6-v2 → 384-dim, ~14ms/query on CPU, excellent semantic accuracy
    EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

    # ── HyDE (Hypothetical Document Embeddings) ─────────────────────────────
    # Generates a short hypothetical answer → embeds it → searches FAISS.
    # Bridges the semantic gap between short queries and long document chunks.
    HYDE_ENABLED = True
    HYDE_MODEL = "llama-3.1-8b-instant"  # lightweight model for HyDE generation

    # ── Domain-Specific RAG Settings ────────────────────────────────────────
    # Each domain has optimized chunking + retrieval parameters for precision.
    DOMAIN_CONFIG = {
        "legal": {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "top_k": 15,
            "similarity_threshold": 0.20,
            "description": "Legal docs need larger chunks to preserve statute/article context",
        },
        "medical": {
            "chunk_size": 600,
            "chunk_overlap": 100,
            "top_k": 10,
            "similarity_threshold": 0.35,
            "description": "Medical docs need smaller, precise chunks for clinical accuracy",
        },
        "hr": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "top_k": 8,
            "similarity_threshold": 0.25,
            "description": "Resume/CV docs need moderate chunks for skill/experience extraction",
        },
        "financial": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "top_k": 10,
            "similarity_threshold": 0.30,
            "description": "Financial docs need precise chunks for numbers and transactions",
        },
        "academic": {
            "chunk_size": 1200,
            "chunk_overlap": 250,
            "top_k": 12,
            "similarity_threshold": 0.25,
            "description": "Academic papers need larger chunks for methodology/findings context",
        },
        "general": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 12,
            "similarity_threshold": 0.25,
            "description": "Default settings for general documents",
        },
    }

    # ── Default RAG Settings (fallback) ─────────────────────────────────────
    CHUNK_SIZE        = 1000   # tokens (sweet spot: 800-1200)
    CHUNK_OVERLAP     = 200    # tokens (sweet spot: 150-250)
    TOP_K             = 12     # chunks retrieved per query
    SIMILARITY_THRESHOLD = 0.25  # min cosine score to include a chunk

    # ── Query Cache ─────────────────────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD = 0.98  # cosine sim for cache hit

    @classmethod
    def get_domain_config(cls, domain: str) -> dict:
        """Returns domain-specific RAG config, falling back to defaults."""
        domain_cfg = cls.DOMAIN_CONFIG.get(domain, cls.DOMAIN_CONFIG["general"])
        return {
            "chunk_size": domain_cfg["chunk_size"],
            "chunk_overlap": domain_cfg["chunk_overlap"],
            "top_k": domain_cfg["top_k"],
            "similarity_threshold": domain_cfg["similarity_threshold"],
        }

# Ensure directories exist
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
