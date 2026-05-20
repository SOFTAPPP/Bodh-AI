import uvicorn
import os
import sys

# bootstrap env
backend_dir = os.path.dirname(os.path.abspath(__file__))

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    os.chdir(backend_dir)
    
    print(f"--- Starting BodhAI Backend ---")
    print(f"Working Directory: {os.getcwd()}")
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
