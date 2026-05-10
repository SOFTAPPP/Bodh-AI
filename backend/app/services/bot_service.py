import asyncio
import numpy as np
from typing import List, Dict, Optional, AsyncIterable
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.document_service import DocumentService
from app.core.config import Config
import os

class PDFChatBot:
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.doc_service = DocumentService()
        # Cache for duplicate detection: list of (embedding, answer_text)
        self.query_cache: List[Dict] = []
        self.similarity_threshold = 0.92

    async def ask(self, query: str, history: List[Dict[str, str]], active_file: Optional[str] = None) -> AsyncIterable[str]:
        # 1. Contextualize & Detect Intent (Parallel for speed)
        context_task = self.llm_service.contextualize_query(query, history)
        intent_task = self.llm_service.detect_intent_and_entities(query)
        
        standalone_query, intent = await asyncio.gather(context_task, intent_task)

        # 2. Duplicate Detection (Problem 1) - Use standalone query for better accuracy
        query_embedding = self.vector_service.embeddings.embed_query(standalone_query)
        
        best_match = None
        max_sim = -1.0
        
        for cached in self.query_cache:
            sim = np.dot(query_embedding, cached["embedding"])
            if sim > max_sim:
                max_sim = sim
                best_match = cached
        
        if max_sim > self.similarity_threshold:
            print(f"--- Duplicate Detected (Sim: {max_sim:.4f}). Returning cached answer. ---")
            yield best_match["answer"]
            return
        
        # 3. Smart Search
        filter_file = intent.get("person_name") or active_file
        docs = self.vector_service.search(standalone_query)
        
        if not docs:
            await self.sync_data_folder()
            docs = self.vector_service.search(standalone_query)

        if not docs:
            msg = "Not mentioned in the provided context."
            # Cache the negative result too
            self.query_cache.append({
                "embedding": query_embedding,
                "answer": msg
            })
            yield msg
            return

        # 4. Prepare Context
        sorted_docs = sorted(docs, key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0)))
        context_text = "\n\n".join([
            f"--- SOURCE: {d.metadata.get('source_file')} ---\n{d.page_content}" 
            for d in sorted_docs
        ])

        # 5. Stream Response & Cache Result
        domain = intent.get("domain", "general")
        full_response = ""
        async for chunk in self.llm_service.generate_response(standalone_query, context_text, domain=domain):
            full_response += chunk
            yield chunk
        
        # Cache the result for future duplicate detection
        if full_response:
            self.query_cache.append({
                "embedding": query_embedding,
                "answer": full_response
            })

    async def process_new_pdf(self, file_path: str):
        """Processes a single PDF and adds to vector store."""
        print(f"--- Processing PDF: {file_path} ---")
        chunks = self.doc_service.process_pdf(file_path)
        print(f"--- Created {len(chunks)} chunks. Adding to Vector Store... ---")
        self.vector_service.add_documents(chunks)
        print(f"--- Vector Store Updated. Marking as indexed... ---")
        self.doc_service.mark_as_indexed(os.path.basename(file_path))
        return len(chunks)

    async def sync_data_folder(self):
        """Production sync logic."""
        indexed_files = self.doc_service.get_indexed_files()
        new_files_count = 0
        
        for file in os.listdir(Config.UPLOAD_DIR):
            if file.lower().endswith(".pdf") and file.lower() not in indexed_files:
                file_path = os.path.join(Config.UPLOAD_DIR, file)
                await self.process_new_pdf(file_path)
                new_files_count += 1
        
        return new_files_count
