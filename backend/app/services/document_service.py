import os
import uuid
from datetime import datetime, timezone
from langchain_community.document_loaders import PyMuPDFLoader
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import Config

class DocumentService:
    def __init__(self, upload_dir: Optional[str] = None, indexed_files_log: Optional[str] = None):
        self.upload_dir       = upload_dir       or Config.UPLOAD_DIR
        self.indexed_files_log = indexed_files_log or Config.INDEXED_FILES_LOG

        self.default_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def get_domain_from_content(self, text: str) -> str:
        """Heuristic-based domain detection for high-speed categorization."""
        text = text.lower()
        if any(kw in text for kw in ["court", "plaintiff", "defendant", "judgment", "legal", "petitioner", "advocate", "tribunal", "appeal", "verdict"]):
            return "legal"
        if any(kw in text for kw in ["patient", "diagnosis", "symptoms", "medical", "clinical", "hospital", "prescription", "dosage", "therapy", "diagnosis"]):
            return "medical"
        if any(kw in text for kw in ["experience", "education", "skills", "cv", "resume", "work history", "professional summary", "certification"]):
            return "hr"
        if any(kw in text for kw in ["invoice", "bill", "payment", "tax", "financial", "pricing", "revenue", "balance sheet", "audit"]):
            return "financial"
        if any(kw in text for kw in ["methodology", "abstract", "references", "research", "conclusion", "hypothesis", "literature review"]):
            return "academic"
        return "general"

    def _get_text_splitter_for_domain(self, domain: str) -> RecursiveCharacterTextSplitter:
        """Returns a text splitter configured for the given domain."""
        domain_cfg = Config.get_domain_config(domain)
        return RecursiveCharacterTextSplitter(
            chunk_size=domain_cfg["chunk_size"],
            chunk_overlap=domain_cfg["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_pdf(self, file_path: str):
        """Loads and splits a PDF into domain-optimized chunks."""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        file_name = os.path.basename(file_path).lower()

        # detect domain
        sample_text = "".join([d.page_content for d in documents[:2]])
        detected_domain = self.get_domain_from_content(sample_text)

        text_splitter = self._get_text_splitter_for_domain(detected_domain)
        chunks = text_splitter.split_documents(documents)

        domain_cfg = Config.get_domain_config(detected_domain)
        doc_id = str(uuid.uuid4())  # Unique identifier for this indexing run
        upload_ts = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name
            chunk.metadata["domain"] = detected_domain
            chunk.metadata["document_id"] = doc_id
            chunk.metadata["upload_timestamp"] = upload_ts
            # store metadata
            chunk.metadata["top_k"] = domain_cfg["top_k"]
            chunk.metadata["similarity_threshold"] = domain_cfg["similarity_threshold"]

        print(f"--- DocumentService: {len(chunks)} chunks created for '{file_name}' "
              f"[domain={detected_domain}, chunk_size={domain_cfg['chunk_size']}] ---")
        return chunks

    def get_indexed_files(self):
        """Returns the set of already indexed files for this platform."""
        if not os.path.exists(self.indexed_files_log):
            return set()
        with open(self.indexed_files_log, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())

    def mark_as_indexed(self, filename: str):
        """Logs a filename as indexed for this platform."""
        with open(self.indexed_files_log, "a", encoding="utf-8") as f:
            f.write(f"{filename.lower()}\n")
