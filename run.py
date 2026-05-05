import subprocess
import os
import sys
import time

def run_app():
    # 1. Path to the backend python inside the venv
    backend_python = os.path.join("backend", "venv", "Scripts", "python.exe")
    
    # 2. Check if venv exists
    if not os.path.exists(backend_python):
        print("Error: Virtual environment not found. Please run the setup first.")
        return

    print("Starting PDF Chatbot...")

    # 3. Start Backend
    print("Starting Backend (FastAPI)...")
    backend_process = subprocess.Popen(
        [backend_python, "-m", "uvicorn", "main:app", "--reload"],
        cwd="backend"
    )

    # 4. Start Frontend
    print("Starting Frontend (Vite)...")
    frontend_process = subprocess.Popen(
        ["npm.cmd", "run", "dev"],
        cwd="frontend",
        shell=True
    )

    print("\n Both servers are starting up!")
    print("Press Ctrl+C to stop both servers at once.\n")

    try:
        # Keep the script running while the processes are active
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    run_app()
