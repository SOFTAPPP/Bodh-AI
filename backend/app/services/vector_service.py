import os
import asyncio
import threading
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from app.core.config import Config

class VectorService:
    """
    FAISS vector store with local HuggingFace embeddings.

    Each instance is fully independent — pass a different `store_dir` to
    create an isolated index (e.g. web app vs WhatsApp).

    Why local embeddings?
    - Zero latency to an external API (no network round-trip for embed calls)
    - all-MiniLM-L6-v2 encodes a query in ~10-20ms on CPU
    - 384-dim vectors → compact FAISS index, fast ANN search
    """

    _shared_embeddings = None
    _shared_lock = threading.Lock()

    def __init__(self, store_dir: Optional[str] = None):
        self._store_dir = store_dir or Config.VECTOR_STORE_DIR
        self.vector_store = None
        self._index_attempted = False

        print(f"--- VectorService: Initialized -> store: {self._store_dir} ---")

    @property
    def embeddings(self):
        """Lazy-loaded HuggingFace embeddings (shared model, separate index)."""
        if VectorService._shared_embeddings is None:
            with VectorService._shared_lock:
                if VectorService._shared_embeddings is None:
                    print("--- VectorService: Loading local embedding model (first-time warm-up) ---")
                    from langchain_huggingface import HuggingFaceEmbeddings
                    VectorService._shared_embeddings = HuggingFaceEmbeddings(
                        model_name=Config.EMBEDDING_MODEL,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
        return VectorService._shared_embeddings

    def warmup(self):
        """Triggers model loading to avoid first-query delay."""
        _ = self.embeddings

    def load_index(self) -> bool:
        """Loads FAISS index from this instance's store_dir. Skips if already loaded."""
        if self.vector_store is not None:
            return True
        if os.path.exists(self._store_dir) and os.listdir(self._store_dir):
            try:
                self.vector_store = FAISS.load_local(
                    self._store_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"--- Vector Store Loaded from {self._store_dir} ---")
                return True
            except Exception as e:
                print(f"--- Incompatible Index ({e}). Auto-clearing for fresh start. ---")
                self.clear_all_data()
        return False

    def clear_all_data(self):
        """Wipes this instance's vector store for a clean rebuild."""
        import shutil
        if os.path.exists(self._store_dir):
            shutil.rmtree(self._store_dir)
            os.makedirs(self._store_dir, exist_ok=True)
        self.vector_store = None
        print(f"--- System Reset: Vector Store cleared at {self._store_dir} ---")

    def save_index(self):
        """Persists the FAISS index to this instance's store_dir."""
        if self.vector_store:
            self.vector_store.save_local(self._store_dir)
            print(f"--- Vector Store Saved to {self._store_dir} ---")

    def add_documents(self, documents: list):
        """Batch-adds documents to FAISS; creates store if it doesn't exist."""
        print(f"--- VectorService: Embedding & indexing {len(documents)} chunks ---")

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

        self.load_index()

        if not self.vector_store:
            return []

        try:
            if embedding is not None:

                results_with_scores = self.vector_store.similarity_search_with_score_by_vector(
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

            if active_file:
                af_lower = active_file.lower()
                active_docs = [d for d in filtered if d.metadata.get("source_file", "").lower() == af_lower]
                if active_docs:
                    filtered = active_docs[:k]
                else:

                    filtered = []

            print(f"--- VectorService: Retrieved {len(scored)} chunks, "
                  f"{len(filtered)} above threshold (threshold={similarity_threshold}) ---")

            if filtered:
                return filtered

            print(f"--- VectorService: No chunks above threshold ({similarity_threshold}). Returning empty. ---")
            return []

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
