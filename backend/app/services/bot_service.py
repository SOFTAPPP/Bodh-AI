import asyncio
import numpy as np
from typing import List, Dict, Optional, AsyncIterable
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.document_service import DocumentService
from app.core.config import Config
import os


class PDFChatBot:
    """
    Enterprise-grade RAG orchestrator with domain-optimized pipeline.

    Pipeline per query:
    1. Heuristic classify (0ms) or lightweight Groq classify (follow-ups only)
    2. [HyDE] Generate hypothetical answer → embed it (bridges semantic gap)
    3. Async local embedding (~15ms)
    4. In-process query cache check (cosine sim, ~0ms) — per-domain
    5. FAISS similarity search with metadata filtering + domain threshold (~5ms)
    6. Active file boosting (prioritize selected file's chunks)
    7. Groq streaming generation (TTFT ~200-500ms)
    """

    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service    = LLMService()
        self.doc_service    = DocumentService()

        # Per-domain query cache: {domain: [{"embedding": [...], "answer": "..."}]}
        self._query_caches: Dict[str, List[Dict]] = {}
        self._cache_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._cache_max = 256  # evict oldest when exceeded per domain

    # ─── Public API ──────────────────────────────────────────────────────────

    async def ask(
        self,
        query: str,
        history: List[Dict[str, str]],
        active_file: Optional[str] = None,
    ) -> AsyncIterable[str]:
        """
        Full RAG pipeline with async streaming.
        Yields text tokens as soon as Groq starts generating.
        """
        import time
        t0 = time.perf_counter()

        # ── Step 1: Classify + Embed (parallel for first turn) ───────────────
        if not history:
            # First turn: heuristic classify (0ms) + embed in parallel
            intent_task   = asyncio.create_task(self.llm_service.analyze_query(query, history))
            embed_task    = asyncio.create_task(self.vector_service.aembed_query(query))
            intent, query_embedding = await asyncio.gather(intent_task, embed_task)
            standalone_query = query
        else:
            # Follow-up: need rephrased query before embedding
            intent           = await self.llm_service.analyze_query(query, history)
            standalone_query = intent.get("standalone_query", query)
            query_embedding  = await self.vector_service.aembed_query(standalone_query)

        t_embed = time.perf_counter()
        print(f"--- [PERF] Classify + Embed: {t_embed - t0:.4f}s ---")

        domain = intent.get("domain", "general")
        response_intent = intent.get("intent", "fact_extraction")

        # ── Step 2: Per-domain Cache lookup ──────────────────────────────────
        cached_answer = self._cache_lookup(query_embedding, domain)
        if cached_answer is not None:
            print(f"--- [PERF] Cache HIT ({domain}) — serving instantly ---")
            yield cached_answer
            return

        # ── Step 3: HyDE — Generate hypothetical answer for better retrieval ─
        hyde_embedding = None
        if Config.HYDE_ENABLED:
            hyde_text = await self.llm_service.generate_hypothetical_answer(standalone_query, domain)
            if hyde_text:
                hyde_embedding = await self.vector_service.aembed_query(hyde_text)
                print(f"--- [PERF] HyDE generated & embedded ({len(hyde_text)} chars) ---")

        # Use HyDE embedding if available, otherwise use query embedding
        search_embedding = hyde_embedding if hyde_embedding is not None else query_embedding

        # ── Step 4: Domain-specific config for search ────────────────────────
        domain_cfg = Config.get_domain_config(domain)
        top_k = domain_cfg["top_k"]
        similarity_threshold = domain_cfg["similarity_threshold"]

        # Build metadata filter for domain (if not general)
        filter_dict = None
        if domain != "general":
            filter_dict = {"domain": domain}

        # ── Step 5: FAISS vector search with metadata filtering ──────────────
        t_search_start = time.perf_counter()
        docs = self.vector_service.search(
            standalone_query,
            k=top_k,
            filter_dict=filter_dict,
            embedding=search_embedding,
            similarity_threshold=similarity_threshold,
            active_file=active_file,
        )

        # Retry loop while background indexing is still running
        retries = 0
        while not docs and active_file and retries < 3:
            print(f"--- [RETRY] Empty context. Waiting for background indexing... ({retries+1}/3) ---")
            await asyncio.sleep(1.5)
            docs = self.vector_service.search(
                standalone_query,
                k=top_k,
                filter_dict=filter_dict,
                embedding=search_embedding,
                similarity_threshold=similarity_threshold,
                active_file=active_file,
            )
            retries += 1

        # Last-resort: trigger sync and retry
        if not docs:
            await self.sync_data_folder()
            docs = self.vector_service.search(
                standalone_query,
                k=top_k,
                filter_dict=filter_dict,
                embedding=search_embedding,
                similarity_threshold=similarity_threshold,
                active_file=active_file,
            )

        t_search = time.perf_counter()
        print(f"--- [PERF] Vector Search: {t_search - t_search_start:.4f}s | "
              f"{len(docs)} chunks [domain={domain}, top_k={top_k}, threshold={similarity_threshold}] ---")

        # ── Step 6: Guard — no context ───────────────────────────────────────
        if not docs:
            msg = "I don't know based on the provided context."
            self._cache_store(query_embedding, msg, domain)
            yield msg
            return

        # ── Step 7: Build context string ─────────────────────────────────────
        meta_domain   = docs[0].metadata.get("domain", "general")
        final_domain  = domain or meta_domain

        # Sort chunks by source file + chunk_index for coherent reading order
        sorted_docs = sorted(
            docs,
            key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0))
        )
        context_text = "\n\n".join(
            f"--- SOURCE: {d.metadata.get('source_file', 'unknown')} | "
            f"Page {d.metadata.get('page', '?')} ---\n{d.page_content}"
            for d in sorted_docs
        )

        # ── Step 8: Stream Groq response ─────────────────────────────────────
        t_gen = time.perf_counter()
        first_token = False
        full_response = ""

        async for chunk in self.llm_service.generate_response(
            standalone_query, context_text, domain=final_domain, intent=response_intent
        ):
            if not first_token:
                ttft = time.perf_counter() - t_gen
                print(f"--- [PERF] TTFT (Time-To-First-Token): {ttft:.4f}s ---")
                first_token = True
            full_response += chunk
            yield chunk

        total = time.perf_counter() - t0
        print(f"--- [PERF] Total Query Latency: {total:.4f}s ---")

        # Cache the answer per-domain
        if full_response:
            self._cache_store(query_embedding, full_response, domain)

    # ─── PDF Ingestion ───────────────────────────────────────────────────────

    async def process_new_pdf(self, file_path: str) -> int:
        """Indexes a single PDF into FAISS (runs in background task)."""
        try:
            # Run CPU-bound PDF parsing in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self.doc_service.process_pdf, file_path)
            # Add documents (embedding is batched inside HuggingFace)
            await loop.run_in_executor(None, self.vector_service.add_documents, chunks)
            self.doc_service.mark_as_indexed(os.path.basename(file_path))
            print(f"--- Indexed {len(chunks)} chunks from {os.path.basename(file_path)} ---")
            return len(chunks)
        except Exception as e:
            print(f"--- Error processing {file_path}: {e} ---")
            return 0

    async def sync_data_folder(self) -> int:
        """Parallel indexing of all un-indexed PDFs in the uploads folder."""
        indexed_files = self.doc_service.get_indexed_files()
        pending = [
            os.path.join(Config.UPLOAD_DIR, f)
            for f in os.listdir(Config.UPLOAD_DIR)
            if f.lower().endswith(".pdf") and f.lower() not in indexed_files
        ]
        if not pending:
            return 0
        print(f"--- Parallel Indexing {len(pending)} pending file(s)... ---")
        results = await asyncio.gather(*[self.process_new_pdf(f) for f in pending])
        return sum(results)

    # ─── Per-Domain Cache Helpers ────────────────────────────────────────────

    def _get_domain_cache(self, domain: str) -> List[Dict]:
        """Returns the cache list for a given domain, creating if needed."""
        if domain not in self._query_caches:
            self._query_caches[domain] = []
        return self._query_caches[domain]

    def _cache_lookup(self, embedding: List[float], domain: str) -> Optional[str]:
        """Returns cached answer if cosine similarity exceeds threshold (per-domain)."""
        cache = self._get_domain_cache(domain)
        if not cache:
            return None
        emb = np.array(embedding)
        best_sim, best_ans = -1.0, None
        for entry in cache:
            sim = float(np.dot(emb, np.array(entry["embedding"])))
            if sim > best_sim:
                best_sim, best_ans = sim, entry["answer"]
        if best_sim >= self._cache_threshold:
            return best_ans
        return None

    def _cache_store(self, embedding: List[float], answer: str, domain: str):
        """Stores a query-answer pair per-domain; evicts oldest if cache is full."""
        cache = self._get_domain_cache(domain)
        if len(cache) >= self._cache_max:
            cache.pop(0)
        cache.append({"embedding": embedding, "answer": answer})
