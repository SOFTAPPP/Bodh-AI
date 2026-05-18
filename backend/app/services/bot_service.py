import asyncio
import numpy as np
from typing import List, Dict, Optional, AsyncIterable
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.document_service import DocumentService
from app.core.config import Config
import os


class PDFChatBot:

    def __init__(self, platform: str = "web"):
        """
        platform: "web"       → uses Config.UPLOAD_DIR / VECTOR_STORE_DIR / INDEXED_FILES_LOG
                  "whatsapp"  → uses Config.WA_UPLOAD_DIR / WA_VECTOR_STORE_DIR / WA_INDEXED_FILES_LOG
        Each platform is fully isolated — documents indexed on one are invisible to the other.
        """
        if platform == "whatsapp":
            vector_store_dir  = Config.WA_VECTOR_STORE_DIR
            upload_dir        = Config.WA_UPLOAD_DIR
            indexed_files_log = Config.WA_INDEXED_FILES_LOG
        else:
            vector_store_dir  = Config.VECTOR_STORE_DIR
            upload_dir        = Config.UPLOAD_DIR
            indexed_files_log = Config.INDEXED_FILES_LOG

        self.platform       = platform
        self.vector_service = VectorService(store_dir=vector_store_dir)
        self.llm_service    = LLMService()
        self.doc_service    = DocumentService(upload_dir=upload_dir, indexed_files_log=indexed_files_log)

        # Per-domain query cache: {domain: [{"embedding": [...], "answer": "..."}]}
        self._query_caches: Dict[str, List[Dict]] = {}
        self._cache_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._cache_max = 256  # evict oldest when exceeded per domain

    async def ask(
        self,
        query: str,
        history: List[Dict[str, str]],
        active_file: Optional[str] = None,
        niche_hint: Optional[str] = None,
    ) -> AsyncIterable[str]:
        """
        Full RAG pipeline with async streaming.
        Yields text tokens as soon as Groq starts generating.
        niche_hint: if provided by the frontend (user selected a niche tab),
                    skip classification entirely — faster and more accurate.
        """
        import time
        t0 = time.perf_counter()

        # Check if the user has uploaded/selected any document yet
        if not active_file and not niche_hint:
            print(f"--- [CONVERSATIONAL] No active file/niche — responding conversationally ---")
            async for chunk in self.llm_service.generate_chitchat_response(query, history, prompt_for_document=True):
                yield chunk
            return

        # Fast-path for simple chitchat/greetings/acknowledgements
        q = query.lower().strip().strip("?").strip("!").strip(".")
        chitchat_phrases = {
            "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "howdy", "who are you", "what is your name", "what can you do",
            "thanks", "thank you", "bye", "goodbye", "help", "what's up", "sup",
            "ok", "okay", "sure", "cool", "great", "nice", "fine", "wow", "got it", "i see", "perfect", "awesome",
            "yes", "no", "thank you", "thx", "congrats", "well", "alright"
        }
        words = q.split()
        if q in chitchat_phrases or (len(words) <= 3 and any(w in chitchat_phrases for w in words)):
            print(f"--- [CHITCHAT] Direct chitchat bypass ---")
            async for chunk in self.llm_service.generate_chitchat_response(query, history):
                yield chunk
            return

        if niche_hint:
            # Frontend already knows the niche — skip LLM classification
            standalone_query = query
            intent = {
                "standalone_query": query,
                "domain": niche_hint,
                "intent": "fact_extraction",
            }
            query_embedding = await self.vector_service.aembed_query(query)
            print(f"--- [NICHE HINT] Domain pre-seeded as '{niche_hint}' — skipping classify ---")
        elif not history:
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

        cached_answer = self._cache_lookup(query_embedding, domain)
        if cached_answer is not None:
            print(f"--- [PERF] Cache HIT ({domain}) — serving instantly ---")
            yield cached_answer
            return

        hyde_embedding = None
        if Config.HYDE_ENABLED:
            hyde_text = await self.llm_service.generate_hypothetical_answer(standalone_query, domain)
            if hyde_text:
                hyde_embedding = await self.vector_service.aembed_query(hyde_text)
                print(f"--- [PERF] HyDE generated & embedded ({len(hyde_text)} chars) ---")

        search_embedding = hyde_embedding if hyde_embedding is not None else query_embedding
        domain_cfg = Config.get_domain_config(domain)
        top_k = domain_cfg["top_k"]
        similarity_threshold = domain_cfg["similarity_threshold"]

        filter_dict = None
        if domain != "general":
            filter_dict = {"domain": domain}
        t_search_start = time.perf_counter()
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

        # ── Fallback: if active_file is known but nothing passed the threshold ──
        # Re-search with threshold=0.0 scoped only to that file, so "what is this about"
        # always works even on short / single-chunk documents.
        if not docs and active_file:
            print(f"--- [FALLBACK] No docs above threshold. Retrying with threshold=0 for active_file={active_file} ---")
            docs = self.vector_service.search(
                standalone_query,
                k=top_k,
                filter_dict=None,          # drop domain filter — file filter is enough
                embedding=search_embedding,
                similarity_threshold=0.0,  # accept everything
                active_file=active_file,
            )
            print(f"--- [FALLBACK] Retrieved {len(docs)} chunks from active file ---")

        if not docs:
            msg = "I couldn't find specific details regarding that in the uploaded document. Feel free to ask another question about the file, or upload a new PDF!"
            self._cache_store(query_embedding, msg, domain)
            yield msg
            return

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

        if full_response:
            self._cache_store(query_embedding, full_response, domain)

    async def process_new_pdf(self, file_path: str) -> int:
        """Indexes a single PDF into FAISS (runs in background task)."""
        try:
            # Run CPU-bound PDF parsing in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self.doc_service.process_pdf, file_path)
            await loop.run_in_executor(None, self.vector_service.add_documents, chunks)
            self.doc_service.mark_as_indexed(os.path.basename(file_path))
            print(f"--- Indexed {len(chunks)} chunks from {os.path.basename(file_path)} ---")
            return len(chunks)
        except Exception as e:
            print(f"--- Error processing {file_path}: {e} ---")
            return 0

    async def sync_data_folder(self) -> int:
        """Parallel indexing of all un-indexed PDFs in this platform's uploads folder."""
        upload_dir = self.doc_service.upload_dir
        indexed_files = self.doc_service.get_indexed_files()
        pending = [
            os.path.join(upload_dir, f)
            for f in os.listdir(upload_dir)
            if f.lower().endswith(".pdf") and f.lower() not in indexed_files
        ]
        if not pending:
            return 0
        print(f"--- [{self.platform}] Parallel Indexing {len(pending)} pending file(s)... ---")
        results = await asyncio.gather(*[self.process_new_pdf(f) for f in pending])
        return sum(results)

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
