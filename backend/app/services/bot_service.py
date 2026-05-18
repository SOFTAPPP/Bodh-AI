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

        STRICT ISOLATION: Each platform can only see its own documents.
        No cross-platform document leakage.
        """
        self.platform = platform
        self.llm_service = LLMService()

        # Each platform only initializes its own services
        if platform == "whatsapp":
            self.doc_service = DocumentService(
                upload_dir=Config.WA_UPLOAD_DIR,
                indexed_files_log=Config.WA_INDEXED_FILES_LOG
            )
            self.vector_service = VectorService(store_dir=Config.WA_VECTOR_STORE_DIR)
        else:
            self.doc_service = DocumentService(
                upload_dir=Config.UPLOAD_DIR,
                indexed_files_log=Config.INDEXED_FILES_LOG
            )
            self.vector_service = VectorService(store_dir=Config.VECTOR_STORE_DIR)

        # Per-domain query cache: {domain: [{\"embedding\": [...], \"answer\": \"...\"}]}
        self._query_caches: Dict[str, List[Dict]] = {}
        self._cache_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._cache_max = 256  # evict oldest when exceeded per domain

    def get_indexed_files(self) -> List[str]:
        """Returns this platform's indexed files as a sorted list. No cross-platform leakage."""
        return sorted(list(self.doc_service.get_indexed_files()))

    async def ask(
        self,
        query: str,
        history: List[Dict[str, str]],
        active_file: Optional[str] = None,
        niche_hint: Optional[str] = None,
        is_selection_retry: bool = False,
    ) -> AsyncIterable[str]:
        """
        Full RAG pipeline with async streaming.
        Yields text tokens as soon as Groq starts generating.

        CASE 1: No documents → instructs user to upload.
        CASE 2: One document → auto-selects it.
        CASE 3: Multiple documents, no active_file → prompts for selection.
        CASE 4: active_file set → retrieves only from that document.
        """
        import time
        t0 = time.perf_counter()

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
            prompt_doc = (active_file is None)
            async for chunk in self.llm_service.generate_chitchat_response(query, history, prompt_for_document=prompt_doc):
                yield chunk
            return

        # ── CASE 1: Check if platform has ANY indexed documents ──────────────────
        platform_files = self.get_indexed_files()

        if not platform_files:
            yield "No files are currently available in the database. Please upload a document first."
            return

        # ── CASE 2 & 3: Resolve active document ─────────────────────────────────
        if len(platform_files) == 1 and not active_file:
            # CASE 2: Only one document — auto-select it
            active_file = platform_files[0]
            print(f"--- [AUTO-SELECT] Single document auto-selected: {active_file} ---")

        elif len(platform_files) > 1 and not active_file and not is_selection_retry:
            # CASE 3: Multiple documents, user hasn't selected one yet.
            # Smart Document Detection (filename match)
            inferred_doc = None
            q_words = set(q.split())
            for f in platform_files:
                f_name_lower = f.lower()
                f_name_no_ext = f_name_lower.replace(".pdf", "")
                if any(w in f_name_lower for w in q_words if len(w) > 3) or q in f_name_no_ext:
                    inferred_doc = f
                    break
            
            if inferred_doc:
                active_file = inferred_doc
                print(f"--- [AUTO-DETECT] Intelligently inferred document: {active_file} ---")
            else:
                print(f"--- [RAG] No active file set with {len(platform_files)} files — prompting user ---")
                msg = "I found multiple documents. Which one are you referring to?\n\n"
                for idx, f in enumerate(platform_files, 1):
                    msg += f"{idx}. **{f}**\n"
                msg += "\nPlease select a document by replying with its number or name."
                yield msg
                return

        # ── CASE 4: Active document is set — run retrieval pipeline ─────────────
        # Normalize active_file against known platform files (case-insensitive)
        if active_file:
            matched_file = None
            for f in platform_files:
                if f.lower() == active_file.lower():
                    matched_file = f
                    break
            active_file = matched_file or active_file

        if niche_hint:
            standalone_query = query
            intent = {
                "standalone_query": query,
                "domain": niche_hint,
                "intent": "fact_extraction",
            }
            query_embedding = await self.vector_service.aembed_query(query)
            print(f"--- [NICHE HINT] Domain pre-seeded as '{niche_hint}' — skipping classify ---")
        elif not history:
            intent_task  = asyncio.create_task(self.llm_service.analyze_query(query, history))
            embed_task   = asyncio.create_task(self.vector_service.aembed_query(query))
            intent, query_embedding = await asyncio.gather(intent_task, embed_task)
            standalone_query = query
        else:
            intent           = await self.llm_service.analyze_query(query, history)
            standalone_query = intent.get("standalone_query", query)
            query_embedding  = await self.vector_service.aembed_query(standalone_query)

        t_embed = time.perf_counter()
        print(f"--- [PERF] Classify + Embed: {t_embed - t0:.4f}s ---")

        domain = intent.get("domain", "general")
        response_intent = intent.get("intent", "fact_extraction")

        cached_answer = self._cache_lookup(query_embedding, domain, active_file)
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
        if active_file:
            filter_dict = {"source_file": active_file}
        elif domain != "general":
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

        # ── Fallback when no docs pass threshold for the active file ────────────
        if not docs and active_file:
            print(f"--- [FALLBACK] No docs above threshold for '{active_file}'. ---")
            msg = f"I could not find this information inside the selected document."
            
            other_files = [f for f in platform_files if f.lower() != active_file.lower()]
            if other_files:
                msg += "\n\nWould you like to search another document? Here are the available options:\n\n"
                for idx, f in enumerate(other_files, 1):
                    msg += f"{idx}. **{f}**\n"
                msg += "\nPlease select a document by replying with its number or name."
            else:
                msg += " Feel free to ask a different question or upload a new PDF."
                self._cache_store(query_embedding, msg, domain, active_file)
                
            yield msg
            return

        meta_domain  = docs[0].metadata.get("domain", "general")
        final_domain = domain or meta_domain

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
            self._cache_store(query_embedding, full_response, domain, active_file)

    async def process_new_pdf(self, file_path: str) -> int:
        """Indexes a single PDF into FAISS (runs in background task)."""
        try:
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

    def _get_domain_cache(self, domain: str, active_file: Optional[str]) -> List[Dict]:
        """Returns the cache list for a given domain and active file, creating if needed."""
        key = f"{domain}_{active_file}"
        if key not in self._query_caches:
            self._query_caches[key] = []
        return self._query_caches[key]

    def _cache_lookup(self, embedding: List[float], domain: str, active_file: Optional[str]) -> Optional[str]:
        """Returns cached answer if cosine similarity exceeds threshold (per-domain/document)."""
        cache = self._get_domain_cache(domain, active_file)
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

    def _cache_store(self, embedding: List[float], answer: str, domain: str, active_file: Optional[str]):
        """Stores a query-answer pair per-domain/document; evicts oldest if cache is full."""
        cache = self._get_domain_cache(domain, active_file)
        if len(cache) >= self._cache_max:
            cache.pop(0)
        cache.append({"embedding": embedding, "answer": answer})
