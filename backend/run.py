import uvicorn
import os
import sys

# Get the absolute path of the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Add the backend directory to sys.path so 'app' can be found
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    # Change to the backend directory to ensure relative paths for 'data/' work correctly
    os.chdir(backend_dir)
    
    print(f"--- Starting BodhAI Backend ---")
    print(f"Working Directory: {os.getcwd()}")
    
    # Use the 'app.main:app' syntax which refers to app/main.py -> app object
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
