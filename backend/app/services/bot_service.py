import asyncio
import numpy as np
import os
import shutil
from typing import List, Dict, Optional, AsyncIterable
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.document_service import DocumentService
from app.services.session_service import SessionManager, SessionState, QAChain
from app.core.config import Config

class PDFChatBot:

    def __init__(self, platform: str = "app"):
        """
        platform: "app"       → uses Config.UPLOAD_DIR / VECTOR_STORE_DIR / INDEXED_FILES_LOG
                  "whatsapp"  → uses Config.WA_UPLOAD_DIR / WA_VECTOR_STORE_DIR / WA_INDEXED_FILES_LOG

        STRICT ISOLATION: Each platform can only see its own documents.
        No cross-platform document leakage.
        """
        self.platform = platform

        self.llm_service = LLMService()

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

        self._query_caches: Dict[str, List[Dict]] = {}
        self._cache_threshold = Config.CACHE_SIMILARITY_THRESHOLD
        self._cache_max = 100
        self._metadata_cache: Dict[str, Dict] = {}
        self._metadata_cache_ts: float = 0.0

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
        session_id: str = "default",
    ) -> AsyncIterable[str]:
        """
        Full isolated RAG pipeline with intelligent document routing and async streaming.
        """
        import time
        import re
        t0 = time.perf_counter()

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

        platform_files = self.get_indexed_files()

        if not platform_files:
            yield "No files are currently available in the database. Please upload a document first."
            return

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
            intent_task  = asyncio.create_task(self.llm_service.analyze_query(query, history))
            embed_task   = asyncio.create_task(self.vector_service.aembed_query(query))
            intent, raw_embedding = await asyncio.gather(intent_task, embed_task)

            standalone_query = intent.get("standalone_query", query)

            if standalone_query.lower().strip() != query.lower().strip():
                print(f"--- [CLASSIFY] Query resolved: '{query}' → '{standalone_query}' ---")
                query_embedding = await self.vector_service.aembed_query(standalone_query)
            else:
                query_embedding = raw_embedding

        t_embed = time.perf_counter()
        print(f"--- [PERF] Classify + Embed: {t_embed - t0:.4f}s ---")

        auto_switch_message = ""

        if not is_selection_retry:
            route_task = asyncio.create_task(self.route_query(standalone_query, platform_files))
            route_res = await route_task
            best_doc = route_res["best_doc"]
            confidence = route_res["confidence"]

            if confidence == "HIGH":
                if not active_file or active_file.lower() != best_doc.lower():
                    auto_switch_message = f"*(Auto-switched context to {best_doc})*\n\n"
                    print(f"--- [AUTO-SWITCH] Switching active document from {active_file} to {best_doc} ---")
                    active_file = best_doc
            elif confidence == "MEDIUM":
                if not active_file:
                    print(f"--- [ROUTER] Medium confidence with no active file. Prompting user. ---")
                    msg = "Which document are you referring to?\n\n"
                    for idx, f in enumerate(platform_files, 1):
                        msg += f"{idx}. **{f}**\n"
                    msg += "\nPlease select a document by replying with its number or name."
                    yield msg
                    return
                else:
                    print(f"--- [ROUTER] Medium confidence but active file exists ({active_file}). Trusting user's recent upload. ---")
            elif confidence == "LOW":
                if active_file:
                    print(f"--- [ROUTER] Low confidence but active file exists ({active_file}). Proceeding. ---")
                else:
                    print(f"--- [ROUTER] Low confidence for all documents. Rejecting query. ---")
                    yield "This question does not appear related to any uploaded document."
                    return
        else:
            route_res = await self.route_query(standalone_query, platform_files)

        if active_file:
            matched_file = None
            for f in platform_files:
                if f.lower() == active_file.lower():
                    matched_file = f
                    break
            active_file = matched_file or active_file

        session = SessionManager.switch_document(
            platform=self.platform,
            session_id=session_id,
            new_document=active_file,
            bot_instance=self
        )

        print({
            "active_document": session.active_document,
            "index_path": session.faiss_index_path,
            "platform": self.platform,
            "session_id": session_id
        })

        domain = intent.get("domain", "general")
        response_intent = intent.get("intent", "fact_extraction")

        cached_answer = self._cache_lookup(query_embedding, domain, active_file)
        if cached_answer is not None:
            print(f"--- [PERF] Cache HIT ({domain}) — serving instantly ---")
            if auto_switch_message:
                yield auto_switch_message
            yield cached_answer
            return

        is_ordinal_query = any(
            word in standalone_query.lower()
            for word in ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                         "first", "second", "third", "fourth", "fifth", "last", "page", "paragraph", "question",
                         "no.", "num", "number", "list"]
        ) or any(char.isdigit() for char in standalone_query)

        hyde_embedding = None
        if Config.HYDE_ENABLED and not is_ordinal_query and response_intent == "complex_analysis":
            hyde_text = await self.llm_service.generate_hypothetical_answer(standalone_query, domain)
            if hyde_text:
                hyde_embedding = await self.vector_service.aembed_query(hyde_text)
                print(f"--- [PERF] HyDE generated & embedded ({len(hyde_text)} chars) ---")

        search_embedding = hyde_embedding if hyde_embedding is not None else query_embedding
        domain_cfg = Config.get_domain_config(domain)
        similarity_threshold = domain_cfg["similarity_threshold"]
        if is_ordinal_query:
            print("--- [RAG] Ordinal/structural query detected. Setting similarity threshold to 0.0 to prevent filtering correct chunks. ---")
            similarity_threshold = 0.0

        t_search_start = time.perf_counter()
        docs = []
        context_text = ""
        if session.qa_chain:
            docs, context_text = await session.qa_chain.ainvoke(
                query=standalone_query,
                embedding=search_embedding,
                history=history,
                domain=domain,
                similarity_threshold=similarity_threshold,
                active_file=active_file,
                response_intent=response_intent
            )
        else:
            print("--- [WARNING] QA Chain not initialized. Index likely missing or empty. ---")

        t_search = time.perf_counter()
        print(f"--- [PERF] Vector Search: {t_search - t_search_start:.4f}s | "
              f"{len(docs)} chunks [domain={domain}, threshold={similarity_threshold}] ---")

        if not docs and active_file:
            relaxed_threshold = max(similarity_threshold * 0.5, 0.1)
            print(f"--- [FALLBACK] No docs above threshold for '{active_file}'. Retrying with relaxed threshold {relaxed_threshold:.2f}... ---")
            if session.qa_chain:
                docs, context_text = await session.qa_chain.ainvoke(
                    query=standalone_query,
                    embedding=search_embedding,
                    history=history,
                    domain=domain,
                    similarity_threshold=relaxed_threshold,
                    active_file=active_file,
                    response_intent=response_intent
                )

        def get_fallback_suggestion_message(current_file: Optional[str]) -> str:
            if not current_file:
                other_files = platform_files
            else:
                other_files = [f for f in platform_files if f.lower() != current_file.lower()]

            msg = "This information was not found in the selected document."
            if other_files:
                msg += "\n\nWould you like to search another document? Here are the available options:\n\n"
                for idx, doc in enumerate(other_files, 1):
                    msg += f"{idx}. **{doc}**\n"
                msg += "\nPlease select a document by replying with its number or name."
            else:
                msg += " Feel free to ask a different question or upload a new PDF."
            return msg

        if not docs and active_file:
            print(f"--- [FALLBACK] Still no docs retrieved for '{active_file}' after 0.0 retry. ---")
            msg = get_fallback_suggestion_message(active_file)
            self._cache_store(query_embedding, msg, domain, active_file)
            if auto_switch_message:
                yield auto_switch_message
            yield msg
            return

        meta_domain  = docs[0].metadata.get("domain", "general")
        final_domain = domain or meta_domain

        t_gen = time.perf_counter()

        if auto_switch_message:
            yield auto_switch_message

        first_token = False
        full_response = ""

        async for chunk in self.llm_service.generate_response(
            standalone_query, context_text, domain=final_domain, intent=response_intent, history=history
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

        if "This information was not found in the selected document" in full_response:
            append_msg = get_fallback_suggestion_message(active_file)
            suggestion_only = append_msg.replace("This information was not found in the selected document.", "").strip()
            if suggestion_only:
                yield "\n\n" + suggestion_only

    async def process_new_pdf(self, file_path: str, session_id: str = "default") -> int:
        """Indexes a single document into its isolated FAISS index. Supports PDF, DOCX, XLSX, TXT, CSV, PPTX."""
        try:
            filename = os.path.basename(file_path).lower()
            clean_doc = "".join(c for c in filename if c.isalnum() or c in (".", "_", "-")).lower()

            dest_dir = os.path.join(Config.DATA_DIR, "indexes", session_id, clean_doc)
            os.makedirs(dest_dir, exist_ok=True)

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self.doc_service.process_pdf, file_path)

            session_vector_service = VectorService(store_dir=dest_dir)
            await loop.run_in_executor(None, session_vector_service.add_documents, chunks)

            self.doc_service.mark_as_indexed(filename)
            print(f"--- Indexed {len(chunks)} chunks from {filename} into {dest_dir} ---")

            if chunks:
                await self.generate_and_save_metadata(filename, chunks)

            if session_id != "default":
                default_dir = os.path.join(Config.DATA_DIR, "indexes", "default", clean_doc)
                os.makedirs(default_dir, exist_ok=True)
                for f in os.listdir(dest_dir):
                    shutil.copy2(os.path.join(dest_dir, f), os.path.join(default_dir, f))
                print(f"--- Cached index for {filename} into default path: {default_dir} ---")

            return len(chunks)
        except Exception as e:
            print(f"--- Error processing {file_path}: {e} ---")
            return 0

    async def generate_and_save_metadata(self, filename: str, chunks: List):
        """Generates document-level summary, keywords, and embeddings, then saves to JSON."""
        import json
        try:
            filename = filename.lower()
            text_sample = "\n\n".join([chunk.page_content for chunk in chunks[:3]])
            metadata = await self.llm_service.generate_document_metadata(filename, text_sample)

            filename_no_ext = filename.replace(".pdf", "")
            filename_embedding = await self.vector_service.aembed_query(filename_no_ext)
            summary_embedding = await self.vector_service.aembed_query(metadata["summary"])
            keywords_str = ", ".join(metadata["topics"])
            keywords_embedding = await self.vector_service.aembed_query(keywords_str)

            metadata_file = os.path.join(Config.DATA_DIR, f"document_metadata_{self.platform}.json")
            metadata_dict = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata_dict = json.load(f)
                except Exception as e:
                    print(f"--- [METADATA] Error loading metadata file: {e} ---")

            metadata_dict[filename] = {
                "document_name": filename,
                "summary": metadata["summary"],
                "topics": metadata["topics"],
                "filename_embedding": filename_embedding,
                "summary_embedding": summary_embedding,
                "keywords_embedding": keywords_embedding
            }

            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata_dict, f, indent=2)
                print(f"--- [METADATA] Successfully saved metadata for {filename} on platform '{self.platform}' ---")
            except Exception as e:
                print(f"--- [METADATA] Error saving metadata file: {e} ---")
        except Exception as e:
            print(f"--- [METADATA] Error generating metadata for {filename}: {e} ---")

    async def ensure_all_metadata_exists(self):
        """Ensures all platform files have metadata. Backfills if missing."""
        import json
        indexed_files = self.get_indexed_files()
        if not indexed_files:
            return

        metadata_file = os.path.join(Config.DATA_DIR, f"document_metadata_{self.platform}.json")
        metadata_dict = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata_dict = json.load(f)
            except Exception as e:
                print(f"--- [METADATA] Error loading metadata: {e} ---")

        missing = [f for f in indexed_files if f.lower() not in metadata_dict]
        if not missing:
            return

        print(f"--- [METADATA] Backfilling missing metadata for: {missing} ---")
        for filename in missing:
            upload_dir = self.doc_service.upload_dir
            file_path = os.path.join(upload_dir, filename)
            if not os.path.exists(file_path):

                for f in os.listdir(upload_dir):
                    if f.lower() == filename.lower():
                        file_path = os.path.join(upload_dir, f)
                        break
            if os.path.exists(file_path):
                try:
                    loop = asyncio.get_event_loop()
                    chunks = await loop.run_in_executor(None, self.doc_service.process_pdf, file_path)
                    if chunks:
                        await self.generate_and_save_metadata(filename, chunks)
                except Exception as e:
                    print(f"--- [METADATA] Backfill failed for {filename}: {e} ---")

    async def _load_metadata_cached(self) -> Dict:
        """Loads document metadata from disk with in-memory caching (5s TTL)."""
        import json
        import time
        metadata_file = os.path.join(Config.DATA_DIR, f"document_metadata_{self.platform}.json")
        if not os.path.exists(metadata_file):
            return {}
        file_mtime = os.path.getmtime(metadata_file)
        if self._metadata_cache and file_mtime <= self._metadata_cache_ts:
            return self._metadata_cache
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                self._metadata_cache = json.load(f)
                self._metadata_cache_ts = file_mtime
        except Exception as e:
            print(f"--- [ROUTER] Error loading metadata: {e} ---")
        return self._metadata_cache

    async def route_query(self, query: str, platform_files: List[str]) -> Dict:
        """
        Intelligently matches the user query to the most semantically relevant document.
        Returns routing details and confidence.
        """
        import re
        if not platform_files:
            return {"best_doc": None, "confidence": "LOW", "similarities": {}}

        await self.ensure_all_metadata_exists()

        metadata_dict = await self._load_metadata_cached()

        query_embedding = await self.vector_service.aembed_query(query)
        q_lower = query.lower()
        q_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 3]

        scores = {}
        similarities = {}

        for filename in platform_files:
            doc_meta = metadata_dict.get(filename.lower())
            if not doc_meta:
                scores[filename] = 0.0
                similarities[filename] = 0.0
                continue

            sim_filename = float(np.dot(query_embedding, doc_meta["filename_embedding"]))
            sim_summary = float(np.dot(query_embedding, doc_meta["summary_embedding"]))
            sim_keywords = float(np.dot(query_embedding, doc_meta["keywords_embedding"]))

            base_score = max(sim_filename, sim_summary, sim_keywords)

            filename_no_ext = filename.lower().replace(".pdf", "")
            heuristic_boost = 0.0

            if filename_no_ext in q_lower or q_lower in filename_no_ext:
                heuristic_boost += 0.35
            else:
                matches = [w for w in q_words if w in filename_no_ext]
                if matches:
                    heuristic_boost += 0.15 * len(matches)

            final_score = base_score + heuristic_boost

            scores[filename] = final_score
            similarities[filename] = base_score

            print(f"--- [ROUTER DEBUG] Platform: {self.platform} | Doc: {filename} | Base: {base_score:.4f} | Boost: {heuristic_boost:.2f} | Final: {final_score:.4f} ---")

        if not scores:
            return {"best_doc": None, "confidence": "LOW", "similarities": {}}

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_doc, best_score = sorted_docs[0]

        confidence = "LOW"

        if len(sorted_docs) == 1:
            confidence = "HIGH"
        else:
            second_doc, second_score = sorted_docs[1]
            diff = best_score - second_score

            if best_score >= 0.45 and diff >= 0.12:
                confidence = "HIGH"

            elif best_score >= 0.30 and diff < 0.12:
                confidence = "MEDIUM"

            elif best_score >= 0.35 and diff >= 0.10:
                confidence = "HIGH"
            elif best_score >= 0.25:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

        print(f"--- [ROUTER RESULT] Platform: {self.platform} | Best Doc: {best_doc} | Confidence: {confidence} | Score: {best_score:.4f} ---")
        return {
            "best_doc": best_doc,
            "confidence": confidence,
            "similarities": similarities,
            "scores": scores
        }

    async def sync_data_folder(self) -> int:
        """Parallel indexing of all un-indexed supported files in this platform's uploads folder."""
        from app.services.document_service import SUPPORTED_EXTENSIONS
        upload_dir = self.doc_service.upload_dir
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
            return 0
        indexed_files = self.doc_service.get_indexed_files()
        pending = [
            os.path.join(upload_dir, f)
            for f in os.listdir(upload_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS and f.lower() not in indexed_files
        ]
        if not pending:
            return 0
        print(f"--- [{self.platform}] Parallel Indexing {len(pending)} pending file(s)... ---")
        results = await asyncio.gather(*[self.process_new_pdf(f, session_id="default") for f in pending])
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
