import shutil
import os

# Script to clear vector store and logs to trigger re-indexing with new chunk settings
VECTOR_STORE_DIR = "data/vector_store"
INDEXED_FILES_LOG = "data/indexed_files.txt"

def clear_system():
    if os.path.exists(VECTOR_STORE_DIR):
        print(f"Clearing {VECTOR_STORE_DIR}...")
        shutil.rmtree(VECTOR_STORE_DIR)
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    if os.path.exists(INDEXED_FILES_LOG):
        print(f"Clearing {INDEXED_FILES_LOG}...")
        os.remove(INDEXED_FILES_LOG)
    
    print("System cleared. Restart the backend and re-upload your PDFs to index them with the new chunk settings.")

if __name__ == "__main__":
    clear_system()
