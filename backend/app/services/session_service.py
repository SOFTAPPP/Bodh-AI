import os
import shutil
import gc
from typing import Dict, Optional, List
from app.core.config import Config
from app.services.vector_service import VectorService

class QAChain:
    """
    Scoped QA Chain that wraps an isolated retriever and handles similarity-score thresholding
    to avoid hallucinations and ensure strictly grounded responses.
    """
    def __init__(self, retriever, llm_service):
        self.retriever = retriever
        self.llm_service = llm_service

    async def ainvoke(
        self,
        query: str,
        embedding: List[float],
        history: List[dict],
        domain: str,
        similarity_threshold: float,
        active_file: str,
        response_intent: str
    ):
        # Access the underlying vectorstore from the isolated retriever
        vectorstore = self.retriever.vectorstore
        k = self.retriever.search_kwargs.get("k", 12)
        
        # Search with similarity scores (L2 distance squared) using pre-computed vector embedding
        scored_raw = vectorstore.similarity_search_with_score_by_vector(embedding, k=k)
        
        # Convert L2 distance score to cosine similarity proxy: cosine_similarity ≈ 1.0 - (L2_distance^2 / 2)
        # Note: Langchain FAISS returns the squared L2 distance as the score.
        scored = [
            (doc, max(0.0, 1.0 - (score / 2.0)))
            for doc, score in scored_raw
        ]
        
        # Print scored chunks for diagnostic purposes
        print(f"--- [RAG SEARCH] Scored chunks for query '{query}': ---")
        for doc, score in scored:
            safe_content = doc.page_content[:60].encode('ascii', errors='replace').decode('ascii').replace('\n', ' ')
            print(f"  - Score: {score:.4f} | File: {doc.metadata.get('source_file')} | Chunk: {doc.metadata.get('chunk_index')} | Content: {safe_content}...")
            
        # Filter chunks that do not meet the minimum similarity threshold
        filtered_docs = [
            doc for doc, score in scored
            if score >= similarity_threshold
        ]

        # strict active file double-check (redundant but safe)
        if active_file:
            af_lower = active_file.lower()
            filtered_docs = [d for d in filtered_docs if d.metadata.get("source_file", "").lower() == af_lower]
        
        if not filtered_docs:
            return [], ""
            
        sorted_docs = sorted(
            filtered_docs,
            key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0))
        )
        
        context_text = "\n\n".join(
            f"--- SOURCE: {d.metadata.get('source_file', 'unknown')} | "
            f"Page {d.metadata.get('page', '?')} ---\n{d.page_content}"
            for d in sorted_docs
        )
        
        return sorted_docs, context_text

class SessionState:
    """
    Maintains isolated session state variables in-memory for zero stale context leakage.
    """
    def __init__(self, session_id: str, platform: str):
        self.session_id = session_id
        self.platform = platform
        self.active_document = None
        self.faiss_index_path = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.memory = []  # Scoped message history

    def unload(self):
        """Explicitly unloads retriever, vector store, and clears memory to prevent leakage."""
        print(f"--- [SESSION STATE UNLOAD] Purging session {self.session_id} on {self.platform} ---")
        self.retriever = None
        self.qa_chain = None
        self.vector_store = None
        self.memory = []
        gc.collect()

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "platform": self.platform,
            "active_document": self.active_document,
            "faiss_index_path": self.faiss_index_path,
        }

class SessionManager:
    """
    Manages session context, document switching, dynamic on-demand isolated index builds,
    and cross-platform index directory structure configuration.
    """
    # Schema: {platform: {session_id: SessionState}}
    _instances: Dict[str, Dict[str, SessionState]] = {
        "app": {},
        "whatsapp": {}
    }

    @classmethod
    def get_session(cls, platform: str, session_id: str) -> SessionState:
        platform = platform.lower()
        if platform not in cls._instances:
            cls._instances[platform] = {}
        if session_id not in cls._instances[platform]:
            cls._instances[platform][session_id] = SessionState(session_id, platform)
        return cls._instances[platform][session_id]

    @classmethod
    def remove_session(cls, platform: str, session_id: str):
        platform = platform.lower()
        if platform in cls._instances and session_id in cls._instances[platform]:
            cls._instances[platform][session_id].unload()
            del cls._instances[platform][session_id]
            gc.collect()

    @classmethod
    def switch_document(cls, platform: str, session_id: str, new_document: str, bot_instance) -> SessionState:
        session = cls.get_session(platform, session_id)
        if session.active_document != new_document or session.vector_store is None:
            print(f"--- [DOCUMENT SWITCH TRIGGERED] Platform: {platform} | Session: {session_id} | Document: {new_document} ---")
            # 1. Purge previous in-memory variables and invoke GC
            session.unload()
            
            # 2. Update active document name
            session.active_document = new_document
            
            # 3. Setup paths and load new vector store, retriever, and qa_chain
            if new_document:
                clean_doc = "".join(c for c in new_document if c.isalnum() or c in (".", "_", "-")).lower()
                session.faiss_index_path = os.path.join(Config.DATA_DIR, "indexes", session_id, clean_doc)
                
                # 4. Load/build isolated vector store and rebuild chain/retriever
                cls.load_session_vectorstore(session, bot_instance)
                if session.vector_store:
                    session.retriever = session.vector_store.as_retriever()
                    session.qa_chain = QAChain(session.retriever, bot_instance.llm_service)
            else:
                session.faiss_index_path = None
        return session

    @classmethod
    def load_session_vectorstore(cls, session: SessionState, bot_instance):
        if not session.active_document:
            return
            
        index_dir = session.faiss_index_path
        
        # Case 1: Session-specific index exists
        if os.path.exists(index_dir) and os.listdir(index_dir):
            print(f"--- [LOAD INDEX] Loading existing session-specific index from {index_dir} ---")
            session_vector_service = VectorService(store_dir=index_dir)
            session_vector_service.load_index()
            session.vector_store = session_vector_service.vector_store
            
        else:
            # Case 2: Default index exists for this document (pre-indexed/synchronized)
            clean_doc = "".join(c for c in session.active_document if c.isalnum() or c in (".", "_", "-")).lower()
            default_index_dir = os.path.join(Config.DATA_DIR, "indexes", "default", clean_doc)
            
            if os.path.exists(default_index_dir) and os.listdir(default_index_dir):
                print(f"--- [COPY INDEX] Copying default index from {default_index_dir} to {index_dir} ---")
                os.makedirs(index_dir, exist_ok=True)
                for f in os.listdir(default_index_dir):
                    shutil.copy2(os.path.join(default_index_dir, f), os.path.join(index_dir, f))
                
                session_vector_service = VectorService(store_dir=index_dir)
                session_vector_service.load_index()
                session.vector_store = session_vector_service.vector_store
                
            else:
                # Case 3: Parse and index on-the-fly
                print(f"--- [BUILD INDEX ON-THE-FLY] Building index for {session.active_document} in session {session.session_id} ---")
                upload_dir = Config.WA_UPLOAD_DIR if session.platform == "whatsapp" else Config.UPLOAD_DIR
                pdf_path = os.path.join(upload_dir, session.active_document)
                
                if not os.path.exists(pdf_path) and os.path.exists(upload_dir):
                    # Case-insensitive scan
                    for f in os.listdir(upload_dir):
                        if f.lower() == session.active_document.lower():
                                pdf_path = os.path.join(upload_dir, f)
                                break
                               
                if os.path.exists(pdf_path):
                    chunks = bot_instance.doc_service.process_pdf(pdf_path)
                    os.makedirs(index_dir, exist_ok=True)
                    session_vector_service = VectorService(store_dir=index_dir)
                    session_vector_service.add_documents(chunks)
                    session.vector_store = session_vector_service.vector_store
                    
                    # Cache a copy under default/ so future sessions can reuse it instantly
                    os.makedirs(default_index_dir, exist_ok=True)
                    for f in os.listdir(index_dir):
                        shutil.copy2(os.path.join(index_dir, f), os.path.join(default_index_dir, f))
                    print(f"--- [CACHE DEFAULT INDEX] Cached default index for {session.active_document} in {default_index_dir} ---")
                else:
                    print(f"--- [ERROR] PDF file not found for RAG: {pdf_path} ---")
                    session.vector_store = None
