import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import Config

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_pdf(self, file_path: str):
        """Loads and splits a PDF into optimized chunks."""
        # PyMuPDF is faster and handles complex layouts better than PyPDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        file_name = os.path.basename(file_path).lower()
        
        # Enrich metadata for domain-specific filtering
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name
            # You can add logic here to detect "domain" (Legal, Resume, etc.) 
            # based on keywords in the first few pages.
            
        return chunks

    def get_indexed_files(self):
        """Returns the list of already indexed files."""
        if not os.path.exists(Config.INDEXED_FILES_LOG):
            return set()
        with open(Config.INDEXED_FILES_LOG, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())

    def mark_as_indexed(self, filename: str):
        """Logs a filename as indexed."""
        with open(Config.INDEXED_FILES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{filename.lower()}\n")
