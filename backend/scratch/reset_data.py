import os
import shutil

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def reset_system():
    print(f"Clearing all contents in {DATA_DIR}...")
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    print(f"Deleted file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted directory: {item}")
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
                
    # Re-create base directories needed by config
    os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "vector_store"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "indexes"), exist_ok=True)
    
    print("\nSystem fully cleared! All uploaded PDFs, FAISS indexes, JSON metadata, and session histories have been removed.")
    print("Please restart your backend (Ctrl+C and run python run.py again) to apply the clean slate.")

if __name__ == "__main__":
    reset_system()
