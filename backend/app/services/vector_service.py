import os
import asyncio
import threading
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from app.core.config import Config


class VectorService:
    """
    Singleton FAISS vector store with local HuggingFace embeddings.

    Why local embeddings?
    - Zero latency to an external API (no network round-trip for embed calls)
    - all-MiniLM-L6-v2 encodes a query in ~10-20ms on CPU
    - 384-dim vectors → compact FAISS index, fast ANN search

    Optimizations:
    - Metadata filtering by domain and source_file for precision
    - Domain-aware similarity threshold per search
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

        self._embeddings = None
        self.vector_store = None
        self._index_attempted = False
        self._lock = threading.Lock()
        
        print("--- VectorService: Initialized (Lazy Loading Enabled) ---")

    @property
    def embeddings(self):
        """Lazy-loaded HuggingFace embeddings."""
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    print("--- VectorService: Loading local embedding model (first-time warm-up) ---")
                    # Move import here to avoid top-level overhead
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=Config.EMBEDDING_MODEL,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
        return self._embeddings

    def warmup(self):
        """Triggers model loading to avoid first-query delay."""
        _ = self.embeddings

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

    def add_documents(self, documents: list):
        """Batch-adds documents to FAISS; creates store if it doesn't exist."""
        print(f"--- VectorService: Embedding & indexing {len(documents)} chunks ---")
        
        # Ensure we try to load existing index before creating a new one
        if self.vector_store is None and not self._index_attempted:
            self._index_attempted = True
            self.load_index()

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.save_index()
        print("--- VectorService: Index saved. ---")

    def search(
        self,
        query: str,
        k: int = Config.TOP_K,
        filter_dict: Optional[dict] = None,
        embedding: Optional[List[float]] = None,
        similarity_threshold: float = Config.SIMILARITY_THRESHOLD,
        active_file: Optional[str] = None,
    ) -> list:
        """
        Similarity search with metadata filtering and domain-aware threshold.

        Parameters:
        - query: text query string
        - k: number of chunks to retrieve
        - filter_dict: optional FAISS metadata filter (e.g., {"domain": "legal"})
        - embedding: pre-computed embedding (avoids double encode)
        - similarity_threshold: minimum cosine score to include a chunk
        - active_file: if set, boosts chunks from this file in the result set

        Returns filtered docs sorted by relevance.
        """
        if self.vector_store is None and not self._index_attempted:
            self._index_attempted = True
            self.load_index()

        if not self.vector_store:
            return []

        try:
            if embedding is not None:
                # Use _with_score (L2 distance) since _with_relevance_scores is not available for by_vector
                results_with_scores = self.vector_store.similarity_search_by_vector_with_score(
                    embedding, k=k, filter=filter_dict
                )
                scored = [
                    (doc, max(0.0, 1.0 - (score / 2.0)))
                    for doc, score in results_with_scores
                ]
            else:
                scored = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=k, filter=filter_dict
                )

            filtered = [
                doc for doc, score in scored
                if score >= similarity_threshold
            ]

            # Active file boosting: if user selected a specific file, ensure at least 50% of results come from that file
            if active_file and filtered:
                active_docs = [d for d in filtered if d.metadata.get("source_file", "").lower() == active_file.lower()]
                other_docs = [d for d in filtered if d.metadata.get("source_file", "").lower() != active_file.lower()]

                if active_docs:
                    target_active = max(len(filtered) // 2, min(len(active_docs), 3))
                    boosted = active_docs[:target_active] + other_docs[:max(1, len(filtered) - target_active)]
                    filtered = boosted[:k]

            print(f"--- VectorService: Retrieved {len(scored)} chunks, "
                  f"{len(filtered)} above threshold (threshold={similarity_threshold}) ---")

            if filtered:
                return filtered
            return [doc for doc, _ in scored[:3]]

        except Exception as e:
            print(f"--- VectorService search error: {e}. Falling back to basic search. ---")
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
