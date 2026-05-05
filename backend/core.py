import os
import sys
import shutil
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Robust path handling for both script and sidecar/executable modes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If running as a PyInstaller bundle, use the executable directory
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)

# Search for .env in current file dir or executable dir
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# Setup paths relative to the base directory
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
INDEXED_FILES_LOG = os.path.join(DATA_DIR, "indexed_files.txt")

print(f"--- Backend Initializing ---")
print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class PDFChatBot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1) 
        self.vector_store = None
        self._indexed_files_cache = None
        self.load_index()

    async def detect_person_dynamically(self, query: str):
        """Uses LLM to identify the subject of the question."""
        if any(greet in query.lower() for greet in ["hi", "hello", "hey"]):
            return None

        prompt = f"Identify the person's name this query is about: '{query}'. Respond with ONLY the name (e.g., 'Aritra' or 'John'). If no specific person is mentioned, respond 'None'."
        try:
            response = await self.llm.ainvoke(prompt)
            name = response.content.strip().lower()
            return name if name != "none" else None
        except:
            return None

    # -------------------------------
    # PROCESS PDF
    # -------------------------------
    def get_indexed_files(self):
        """Returns a set of already indexed filenames from cache or file."""
        if self._indexed_files_cache is not None:
            return self._indexed_files_cache
            
        if not os.path.exists(INDEXED_FILES_LOG):
            self._indexed_files_cache = set()
            return self._indexed_files_cache
            
        with open(INDEXED_FILES_LOG, "r", encoding="utf-8") as f:
            self._indexed_files_cache = set(line.strip().lower() for line in f if line.strip())
            return self._indexed_files_cache

    def mark_file_as_indexed(self, filename: str):
        """Logs a filename as indexed and updates cache."""
        with open(INDEXED_FILES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{filename.lower()}\n")
        
        if self._indexed_files_cache is not None:
            self._indexed_files_cache.add(filename.lower())
        else:
            self.get_indexed_files() # Initialize cache if needed

    def process_pdf(self, file_path: str, force_reindex: bool = False):
        file_name = os.path.basename(file_path).lower()
        indexed_files = self.get_indexed_files()

        if file_name in indexed_files and not force_reindex:
            print(f"--- Skipping {file_name} (Already Indexed) ---")
            return 0

        print(f"--- Indexing: {file_name} ---")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name   

        # Load existing index or create new one
        if self.vector_store is None:
            if os.path.exists(FAISS_INDEX_PATH):
                self.load_index()
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        
        self.vector_store.save_local(FAISS_INDEX_PATH)
        self.mark_file_as_indexed(file_name)
        return len(chunks)

    def sync_folder(self):
        """Scans the data directory and indexes any new PDF files using optimized scandir."""
        indexed_files = self.get_indexed_files()
        new_files_count = 0
        total_chunks = 0

        print(f"--- Syncing Folder: {DATA_DIR} ---")
        try:
            with os.scandir(DATA_DIR) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(".pdf"):
                        if entry.name.lower() not in indexed_files:
                            file_path = entry.path
                            try:
                                chunks = self.process_pdf(file_path)
                                new_files_count += 1
                                total_chunks += chunks
                            except Exception as e:
                                print(f"Error processing {entry.name}: {e}")
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        if new_files_count > 0:
            print(f"--- Sync Complete: {new_files_count} new files, {total_chunks} total chunks added ---")
        
        return new_files_count, total_chunks

    # -------------------------------
    def load_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except:
                return False
        return False

    async def contextualize_query(self, query: str, history: list):
        """Robustly rephrases follow-up questions into standalone queries."""
        if not history:
            return query
        
        history_str = ""
        for msg in history[-5:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content', '')}\n"
            
        prompt = f"""You are a query contextualizer. Given a conversation history and a follow-up question, your goal is to rephrase the follow-up into a standalone question that mentions the correct subjects and entities.
        
        RULES:
        - Replace pronouns (he, she, they, it) with the actual name or entity mentioned in history.
        - If the follow-up says "this project" or "that project", replace it with the specific project name from the history.
        - Keep the output as a concise, direct question.
        
        HISTORY:
        {history_str}
        
        FOLLOW-UP: {query}
        
        STANDALONE QUESTION:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            standalone = response.content.strip()
            return standalone if standalone else query
        except:
            return query

    # -------------------------------
    # MAIN QUESTION FUNCTION
    # -------------------------------
    async def ask_question(self, query: str, history: Optional[List[Dict[str, str]]] = None, active_file: Optional[str] = None):
        if history is None:
            history = []
            
        # 0. Ensure index exists or try to sync if empty
        if not self.vector_store:
            if not self.load_index():
                print("--- Index empty, attempting to sync from data folder... ---")
                self.sync_folder()
                if not self.vector_store:
                    yield "I couldn't find any documents in the database. Please upload some PDFs first."
                    return

        # 1. & 2. Parallel Pre-processing (Contextualization & Person Detection)
        # We run these in parallel to save time.
        standalone_task = self.contextualize_query(query, history)
        
        # Skip person detection for very short queries to save one LLM call
        if len(query.split()) < 3:
            standalone_query = await standalone_task
            target_person = None
        else:
            # Detect person on the raw query to start early
            person_task = self.detect_person_dynamically(query)
            standalone_query, target_person = await asyncio.gather(standalone_task, person_task)

        print(f"--- Original: {query} | Standalone: {standalone_query} ---")
        
        search_kwargs = {
            "k": 15,  # Reduced from 25 for speed (15 is enough for most resumes/projects)
            "fetch_k": 50,
            "lambda_mult": 0.5
        } 
        
        filter_file = None
        if target_person:
            print(f"--- TARGET PERSON DETECTED: {target_person} ---")
            filter_file = target_person
        elif active_file:
            print(f"--- USING ACTIVE FILE HINT: {active_file} ---")
            filter_file = active_file.lower()

        # Try both filtered and unfiltered search to ensure we don't miss anything
        # while still giving priority to the 'active' or 'targeted' file.
        all_docs = []
        
        # 1. Search everything (Broad Search)
        print(f"--- Performing broad database search ---")
        broad_docs = self.vector_store.search(
            standalone_query, 
            search_type="mmr",
            **search_kwargs
        )
        all_docs.extend(broad_docs)

        # 2. If there's a filter/active file, do a targeted search and merge
        if filter_file:
            print(f"--- Performing targeted search for: {filter_file} ---")
            filtered_kwargs = search_kwargs.copy()
            filtered_kwargs["k"] = 10 # Get a few specific chunks from the target
            filtered_kwargs["filter"] = lambda metadata: filter_file in metadata.get("source_file", "").lower()
            
            target_docs = self.vector_store.search(
                standalone_query, 
                search_type="mmr",
                **filtered_kwargs
            )
            # Add to the beginning to give priority
            all_docs = target_docs + all_docs

        # 3. Deduplicate by content/source to stay within token limits
        seen_contents = set()
        docs = []
        for d in all_docs:
            content_hash = hash(d.page_content)
            if content_hash not in seen_contents:
                docs.append(d)
                seen_contents.add(content_hash)
            if len(docs) >= 20: # Limit total context to 20 high-quality chunks
                break

        # 4. Final Safety Sync: If STILL no docs at all, check for new files
        if not docs:
            print("--- NO RESULTS FOUND. Checking for new files in data folder... ---")
            new_files, _ = self.sync_folder()
            if new_files > 0:
                print(f"--- Found {new_files} new files. Retrying search... ---")
                docs = self.vector_store.search(standalone_query, search_type="mmr", **search_kwargs)

        if not docs:
            yield "I don't know based on the provided context."
            return

        sorted_docs = sorted(
            docs,
            key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0))
        )

        context_text = "\n\n".join([f"--- SOURCE: {d.metadata.get('source_file')} ---\n{d.page_content}" for d in sorted_docs])

        template = """You are a high-precision document extraction and analysis AI.

        Your task is to extract, analyze, and present ALL relevant information from the given context.

        CORE RULES:

        1. NO HALLUCINATION
        - Use ONLY the provided context.
        - You MAY perform logical deductions based on dates (e.g., graduation year 2027 means currently a student).

        2. FULL COMPLETENESS (CRITICAL)
        - Extract EVERY relevant item related to the query.
        - Analyze the entire context for the requested data.

        3. STRUCTURED OUTPUT (MANDATORY)
        - Organize output into clear sections using Markdown.
        - Use bold headers and bullet points.
        - For projects, always list the Name, Technologies, Description, and Developer/Candidate (if found in context) separately.
        - Always identify the person the information belongs to if a name is present in the context (e.g., at the top of a resume).

        4. MISSING INFORMATION HANDLING
        - If the requested data is absolutely not in the context:
        → respond exactly with: "I don't know based on the provided context."

        5. NO REPETITION OF THE QUESTION
        - Start your answer immediately with the data. Do NOT restate or repeat the user's question.

        --------------------------------------------------

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context_text, question=standalone_query)

        full_response = ""
        async for chunk in self.llm.astream(final_prompt):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        print(f"\n--- AI RESPONSE ---\n{full_response}\n-------\n")