import os
import sys
import shutil
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

print(f"--- Backend Initializing ---")
print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class PDFChatBot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1) # Lower temperature for higher precision
        self.vector_store = None

    async def detect_person_dynamically(self, query: str):
        """Uses LLM to identify the subject of the question."""
        if any(greet in query.lower() for greet in ["hi", "hello", "hey"]):
            return None

        prompt = f"Identify the person's name this query is about: '{query}'. Respond with ONLY the first name. If no specific person is mentioned, respond 'None'."
        try:
            response = await self.llm.ainvoke(prompt)
            name = response.content.strip().lower()
            return name if name != "none" else None
        except:
            return None

    # -------------------------------
    # PROCESS PDF
    # -------------------------------
    def process_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        file_name = os.path.basename(file_path).lower()

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name   

        if os.path.exists(FAISS_INDEX_PATH):
            try:
                shutil.rmtree(FAISS_INDEX_PATH)
            except:
                pass
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(FAISS_INDEX_PATH)
        return len(chunks)

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
    async def ask_question(self, query: str, history: list = [], active_file: str = None):
        if not self.vector_store:
            if not self.load_index():
                yield "No PDF uploaded yet."
                return

        # 1. Contextualize the query
        standalone_query = await self.contextualize_query(query, history)
        print(f"--- Original: {query} | Standalone: {standalone_query} ---")

        target_person = await self.detect_person_dynamically(standalone_query)
        
        search_kwargs = {
            "k": 15,
            "fetch_k": 40,
            "lambda_mult": 0.6
        } 
        
        filter_file = None
        if target_person:
            print(f"--- TARGET DETECTED: {target_person} (Filtering Search) ---")
            filter_file = target_person
        elif active_file:
            print(f"--- NO TARGET: Using Active File Fallback: {active_file} ---")
            filter_file = active_file.lower()

        if filter_file:
            search_kwargs["filter"] = lambda metadata: filter_file in metadata.get("source_file", "").lower()

        docs = self.vector_store.search(
            standalone_query, 
            search_type="mmr",
            **search_kwargs
        )

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
        - For projects, always list the Name, Technologies, and Description separately.

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