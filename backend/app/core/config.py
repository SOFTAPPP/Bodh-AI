import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "bodhai_verify_token")
    META_API_TOKEN = os.getenv("META_API_TOKEN", "")
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

    BASE_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    INDEXED_FILES_LOG = os.path.join(DATA_DIR, "indexed_files.txt")

    GENERATION_MODEL = "llama-3.3-70b-versatile"
    REASONING_MODEL  = "deepseek-r1-distill-llama-70b"
    # Note: gemma2-9b-it was decommissioned by Groq; using llama-3.1-8b-instant instead
    FALLBACK_MODEL   = "llama-3.1-8b-instant"

    # all-MiniLM-L6-v2 → 384-dim, ~14ms/query on CPU, excellent semantic accuracy
    EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

    # Generates a short hypothetical answer → embeds it → searches FAISS.
    # Bridges the semantic gap between short queries and long document chunks.
    HYDE_ENABLED = True
    HYDE_MODEL = "llama-3.1-8b-instant"

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

    CHUNK_SIZE        = 1000   # sweet spot: 800-1200 tokens
    CHUNK_OVERLAP     = 200    # sweet spot: 150-250 tokens
    TOP_K             = 12
    SIMILARITY_THRESHOLD = 0.25

    CACHE_SIMILARITY_THRESHOLD = 0.98

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

os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
