import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import Config

class DocumentService:
    def __init__(self):
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

        # Detect domain based on first 2000 chars
        sample_text = "".join([d.page_content for d in documents[:2]])
        detected_domain = self.get_domain_from_content(sample_text)

        text_splitter = self._get_text_splitter_for_domain(detected_domain)
        chunks = text_splitter.split_documents(documents)

        domain_cfg = Config.get_domain_config(detected_domain)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = file_name
            chunk.metadata["domain"] = detected_domain
            # Store domain-specific retrieval config in metadata for vector search
            chunk.metadata["top_k"] = domain_cfg["top_k"]
            chunk.metadata["similarity_threshold"] = domain_cfg["similarity_threshold"]

        print(f"--- DocumentService: {len(chunks)} chunks created for '{file_name}' "
              f"[domain={detected_domain}, chunk_size={domain_cfg['chunk_size']}] ---")
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
