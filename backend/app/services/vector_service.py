import os
import asyncio
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import Config


class VectorService:
    """
    Singleton FAISS vector store with local HuggingFace embeddings.

    Why local embeddings?
    - Zero latency to an external API (no network round-trip for embed calls)
    - all-MiniLM-L6-v2 encodes a query in ~10-20ms on CPU
    - 384-dim vectors → compact FAISS index, fast ANN search
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        print("--- VectorService: Loading local embedding model (first-time warm-up) ---")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
        )
        self.vector_store = None
        self.load_index()

    # ─── Persistence ────────────────────────────────────────────────────────

    def load_index(self) -> bool:
        """Loads FAISS index from disk; auto-clears on model mismatch."""
        if os.path.exists(Config.VECTOR_STORE_DIR) and os.listdir(Config.VECTOR_STORE_DIR):
            try:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_STORE_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"--- Vector Store Loaded from {Config.VECTOR_STORE_DIR} ---")
                return True
            except Exception as e:
                print(f"--- Incompatible Index ({e}). Auto-clearing for fresh start. ---")
                self.clear_all_data()
        return False

    def clear_all_data(self):
        """Wipes vector store and index log for a clean rebuild."""
        import shutil
        if os.path.exists(Config.VECTOR_STORE_DIR):
            shutil.rmtree(Config.VECTOR_STORE_DIR)
            os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
        if os.path.exists(Config.INDEXED_FILES_LOG):
            os.remove(Config.INDEXED_FILES_LOG)
        self.vector_store = None
        print("--- System Reset: Vector Store and Index Log Cleared ---")

    def save_index(self):
        """Persists the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(Config.VECTOR_STORE_DIR)
            print(f"--- Vector Store Saved to {Config.VECTOR_STORE_DIR} ---")

    # ─── Ingestion ──────────────────────────────────────────────────────────

    def add_documents(self, documents: list):
        """Batch-adds documents to FAISS; creates store if it doesn't exist."""
        print(f"--- VectorService: Embedding & indexing {len(documents)} chunks ---")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.save_index()
        print("--- VectorService: Index saved. ---")

    # ─── Retrieval ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = Config.TOP_K,
        filter_dict: Optional[dict] = None,
        embedding: Optional[List[float]] = None,
    ) -> list:
        """
        Similarity search with optional pre-computed embedding.
        Returns docs filtered by SIMILARITY_THRESHOLD to suppress noise.
        """
        if not self.vector_store:
            return []

        try:
            if embedding is not None:
                # Use pre-computed embedding → avoids a double encode
                results_with_scores = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                    embedding, k=k, filter=filter_dict
                )
            else:
                results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=k, filter=filter_dict
                )

            # Filter by similarity threshold to eliminate irrelevant chunks
            filtered = [
                doc for doc, score in results_with_scores
                if score >= Config.SIMILARITY_THRESHOLD
            ]
            print(f"--- VectorService: Retrieved {len(results_with_scores)} chunks, {len(filtered)} above threshold ---")
            return filtered if filtered else [doc for doc, _ in results_with_scores[:3]]  # always return at least top-3

        except Exception as e:
            print(f"--- VectorService search error: {e}. Falling back to basic search. ---")
            # Graceful fallback
            if embedding is not None:
                return self.vector_store.similarity_search_by_vector(embedding, k=k)
            return self.vector_store.similarity_search(query, k=k)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async wrapper for local embedding (runs in thread pool to avoid blocking).
        Returns a normalized embedding vector for the given text.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embeddings.embed_query, text
        )
