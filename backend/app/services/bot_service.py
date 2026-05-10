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
        self.similarity_threshold = 0.98

    async def ask(self, query: str, history: List[Dict[str, str]], active_file: Optional[str] = None) -> AsyncIterable[str]:
        import time
        start_time = time.perf_counter()
        
        # 1. Parallel Analysis & Embedding
        if not history:
            analysis_task = self.llm_service.analyze_query(query, history)
            embed_task = self.vector_service.embeddings.aembed_query(query)
            intent, query_embedding = await asyncio.gather(analysis_task, embed_task)
            standalone_query = query
        else:
            intent = await self.llm_service.analyze_query(query, history)
            standalone_query = intent.get("standalone_query", query)
            query_embedding = await self.vector_service.embeddings.aembed_query(standalone_query)
        
        analysis_end = time.perf_counter()
        print(f"--- [PERF] Analysis & Embedding: {analysis_end - start_time:.4f}s ---")
        
        best_match = None
        max_sim = -1.0
        
        for cached in self.query_cache:
            sim = np.dot(query_embedding, cached["embedding"])
            if sim > max_sim:
                max_sim = sim
                best_match = cached
        
        if max_sim > self.similarity_threshold:
            print(f"--- [PERF] Cache Hit (Sim: {max_sim:.4f}). Returning cached answer. ---")
            yield best_match["answer"]
            return
        
        # 4. Prepare Context & Domain Verification (with background-wait logic)
        search_start = time.perf_counter()
        docs = self.vector_service.search(standalone_query, embedding=query_embedding)
        
        # If no docs found, maybe background indexing is still running? 
        # Retry up to 3 times with 1.5s delay if an active file is provided
        retries = 0
        while not docs and active_file and retries < 3:
            print(f"--- [RETRY] Context empty. Waiting for background indexing... (Attempt {retries+1}) ---")
            await asyncio.sleep(1.5)
            # Force a mini-sync to check if file was just completed
            docs = self.vector_service.search(standalone_query, embedding=query_embedding)
            retries += 1

        if not docs:
            await self.sync_data_folder()
            docs = self.vector_service.search(standalone_query, embedding=query_embedding)

        search_end = time.perf_counter()
        print(f"--- [PERF] Vector Search: {search_end - search_start:.4f}s ---")

        if not docs:
            msg = "Not mentioned in the provided context."
            self.query_cache.append({"embedding": query_embedding, "answer": msg})
            yield msg
            return

        # Refine Domain from Metadata
        meta_domain = docs[0].metadata.get("domain", "general")
        final_domain = intent.get("domain") or meta_domain

        # 5. Stream Response
        sorted_docs = sorted(docs, key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0)))
        context_text = "\n\n".join([
            f"--- SOURCE: {d.metadata.get('source_file')} ---\n{d.page_content}" 
            for d in sorted_docs
        ])

        generation_start = time.perf_counter()
        first_token_sent = False
        full_response = ""
        
        # Pass the detected intent (defaulting to fact_extraction)
        response_intent = intent.get("intent", "fact_extraction")
        async for chunk in self.llm_service.generate_response(standalone_query, context_text, domain=final_domain, intent=response_intent):
            if not first_token_sent:
                ttft = time.perf_counter() - generation_start
                print(f"--- [PERF] Time To First Token (TTFT): {ttft:.4f}s ---")
                first_token_sent = True
            full_response += chunk
            yield chunk
        
        total_time = time.perf_counter() - start_time
        print(f"--- [PERF] Total Query Latency: {total_time:.4f}s ---")
        
        if full_response:
            self.query_cache.append({"embedding": query_embedding, "answer": full_response})

    async def process_new_pdf(self, file_path: str):
        """Processes a single PDF and adds to vector store."""
        try:
            chunks = self.doc_service.process_pdf(file_path)
            self.vector_service.add_documents(chunks)
            self.doc_service.mark_as_indexed(os.path.basename(file_path))
            return len(chunks)
        except Exception as e:
            print(f"--- Error processing {file_path}: {e} ---")
            return 0

    async def sync_data_folder(self):
        """Ultra-fast parallel indexing of the entire uploads folder."""
        indexed_files = self.doc_service.get_indexed_files()
        pending_files = [
            os.path.join(Config.UPLOAD_DIR, f) 
            for f in os.listdir(Config.UPLOAD_DIR) 
            if f.lower().endswith(".pdf") and f.lower() not in indexed_files
        ]
        
        if not pending_files:
            return 0
        
        print(f"--- Parallel Indexing {len(pending_files)} files... ---")
        tasks = [self.process_new_pdf(f) for f in pending_files]
        results = await asyncio.gather(*tasks)
        return sum(results)
