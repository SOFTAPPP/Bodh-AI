import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import Config

class VectorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
            # Reverting to OpenAI for stable, managed performance
            cls._instance.embeddings = OpenAIEmbeddings()
            cls._instance.vector_store = None
            cls._instance.load_index()
        return cls._instance

    def load_index(self):
        """Loads the FAISS index from disk with auto-healing for model mismatches."""
        if os.path.exists(Config.VECTOR_STORE_DIR) and os.listdir(Config.VECTOR_STORE_DIR):
            try:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_STORE_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"--- Vector Store Loaded from {Config.VECTOR_STORE_DIR} ---")
                return True
            except Exception as e:
                print(f"--- Incompatible Index Detected ({e}). Auto-clearing for new model... ---")
                self.clear_all_data()
                return False
        return False

    def clear_all_data(self):
        """Wipes the vector store and indexed files log to allow a fresh start."""
        import shutil
        if os.path.exists(Config.VECTOR_STORE_DIR):
            shutil.rmtree(Config.VECTOR_STORE_DIR)
            os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
        if os.path.exists(Config.INDEXED_FILES_LOG):
            os.remove(Config.INDEXED_FILES_LOG)
        self.vector_store = None
        print("--- System Reset: Vector Store and Index Log Cleared ---")

    def save_index(self):
        """Saves the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(Config.VECTOR_STORE_DIR)
            print(f"--- Vector Store Saved to {Config.VECTOR_STORE_DIR} ---")

    def add_documents(self, documents):
        """Adds new documents to the vector store."""
        print(f"--- VectorService: Adding {len(documents)} docs to FAISS ---")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        print("--- VectorService: Saving Index... ---")
        self.save_index()

    def search(self, query: str, k: int = Config.TOP_K, filter_dict: Optional[dict] = None):
        """Performs a similarity search with optional filtering."""
        if not self.vector_store:
            return []
        
        # Standard similarity search is much faster than MMR for high-volume chats
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
