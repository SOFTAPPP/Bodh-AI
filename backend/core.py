import os
import sys
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
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        self.vector_store = None

    # ---------------------------------------------------------
    # DYNAMIC: Detect person name using AI (Production Ready)
    # ---------------------------------------------------------
    async def detect_person_dynamically(self, query: str):
        """Uses LLM to identify the subject of the question."""
        # Simple, fast check for greetings to avoid unnecessary LLM calls
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

        # Create / update FAISS
        if os.path.exists(FAISS_INDEX_PATH):
            existing_vs = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            existing_vs.add_documents(chunks)
            self.vector_store = existing_vs
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        self.vector_store.save_local(FAISS_INDEX_PATH)
        return len(chunks)

    # -------------------------------
    def load_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return False

    # -------------------------------
    # MAIN QUESTION FUNCTION
    # -------------------------------
    async def ask_question(self, query: str):

        greetings = ["hi", "hello", "hey", "who are you"]
        if query.lower().strip() in greetings:
            yield "Hello! I am your PDF Assistant. How can I help you today?"
            return

        if not self.vector_store:
            if not self.load_index():
                yield "No PDF uploaded yet."
                return

        # ---------------------------------------------------------
        # OPTIMIZED RETRIEVAL: Use Metadata Filtering + MMR
        # ---------------------------------------------------------
        # DYNAMIC RETRIEVAL: Auto-detect person and filter
        # ---------------------------------------------------------
        target_person = await self.detect_person_dynamically(query)
        
        # Increase k to ensure we catch all relevant chunks in a multi-page document
        search_kwargs = {"k": 15} 
        
        # If we detect a person, strictly filter to their file only
        if target_person:
            print(f"--- TARGET DETECTED: {target_person} (Filtering Search) ---")
            search_kwargs["filter"] = lambda metadata: target_person in metadata.get("source_file", "").lower()

        # Use similarity search for better recall of all items
        docs = self.vector_store.similarity_search(
            query, 
            **search_kwargs
        )

        relevant_docs = docs
        if not relevant_docs:
            yield f"I don't know based on these files. I couldn't find any relevant sections matching your query: **{query[0:50]}...**"
            return

        # -------------------------------
        # BUILD CONTEXT
        # -------------------------------
        # Sort by source and then chunk index to maintain logical order (Introduction -> Projects -> Skills)
        sorted_docs = sorted(
            relevant_docs,
            key=lambda x: (x.metadata.get("source_file", ""), x.metadata.get("chunk_index", 0))
        )

        context_parts = []
        for doc in sorted_docs:
            filename = doc.metadata.get("source_file", "unknown")
            context_parts.append(f"--- SOURCE: {filename} ---\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)

        # -------------------------------
        # REFINED PRECISION PROMPT
        # -------------------------------
        template = """You are a high-precision document extraction AI.

        Your task is to extract and present ALL relevant information from the given context with ZERO loss, ZERO hallucination, and MAXIMUM detail.

        CORE RULES:

        1. NO HALLUCINATION
        - Use ONLY the provided context.
        - Do NOT add, assume, or infer anything beyond the text.
        - If something is not explicitly written, do NOT include it.

        2. FULL COMPLETENESS (CRITICAL)
        - Extract EVERY relevant item related to the user’s query.
        - Do NOT summarize unless explicitly asked.
        - Do NOT skip repeated or similar items.
        - If 10 items exist, output all 10.

        3. EXACT EXTRACTION
        - Preserve original wording as much as possible.
        - Do NOT rephrase key information (names, technologies, numbers, boards, years).
        - Maintain factual integrity exactly as written.

        4. STRUCTURED OUTPUT (MANDATORY)
        - Organize output into clear sections using Markdown.
        - Use bold headers like:
        **Projects**
        **Education**
        **Technical Skills**
        **Experience**
        - Under each section, use bullet points.
        - For projects:
        - Show project name as **bold**
        - Add details as bullet points under it

        5. MULTI-ENTITY SAFETY
        - If multiple individuals exist in the context:
        - ONLY extract data for the correct person (based on the query)
        - NEVER mix information between individuals

        6. MISSING INFORMATION HANDLING
        - If the requested data is not found:
        → respond exactly with:
        "I don't know based on the provided context."

        7. NO GENERIC TEXT
        - Do NOT add explanations like "based on the document"
        - Do NOT add filler or conversational text
        - Output only structured extracted data

        8. PRIORITY LOGIC
        - Accuracy > Completeness > Formatting
        - Never sacrifice correctness for formatting

        --------------------------------------------------

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context_text, question=query)

        # -------------------------------
        # STREAM RESPONSE
        # -------------------------------
        full_response = ""
        async for chunk in self.llm.astream(final_prompt):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        print(f"\n--- AI RESPONSE ---\n{full_response}\n-------\n")