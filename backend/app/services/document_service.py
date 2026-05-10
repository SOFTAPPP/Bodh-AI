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

    def get_domain_from_content(self, text: str) -> str:
        """Heuristic-based domain detection for high-speed categorization."""
        text = text.lower()
        if any(kw in text for kw in ["court", "plaintiff", "defendant", "judgment", "legal", "petitioner"]):
            return "legal"
        if any(kw in text for kw in ["patient", "diagnosis", "symptoms", "medical", "clinical", "hospital"]):
            return "medical"
        if any(kw in text for kw in ["experience", "education", "skills", "cv", "resume", "work history"]):
            return "hr"
        if any(kw in text for kw in ["invoice", "bill", "payment", "tax", "financial", "pricing"]):
            return "financial"
        if any(kw in text for kw in ["methodology", "abstract", "references", "research", "conclusion"]):
            return "academic"
        return "general"

    def process_pdf(self, file_path: str):
        """Loads and splits a PDF into optimized chunks with automatic domain detection."""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        file_name = os.path.basename(file_path).lower()
        
        # Detect domain based on first 2000 chars
        sample_text = "".join([d.page_content for d in documents[:2]])
        detected_domain = self.get_domain_from_content(sample_text)
        
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name
            chunk.metadata["domain"] = detected_domain
            
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
