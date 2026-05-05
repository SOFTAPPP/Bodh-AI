import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from core import PDFChatBot

app = FastAPI(title="BodhAI API")

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our Bot logic
bot = PDFChatBot()

@app.on_event("startup")
async def startup_event():
    print("--- Startup: Syncing PDF Database in background ---")
    # Run sync in a separate thread/task to avoid blocking
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, bot.sync_folder)

# Data models
class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    active_file: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "PDF Chatbot Backend is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save file temporarily
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    upload_path = os.path.join("data", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process PDF (Extract, Chunk, Embed)
        num_chunks = bot.process_pdf(upload_path)
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_created": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(bot.ask_question(request.message, request.history, request.active_file), media_type="text/plain")

@app.post("/sync")
async def sync_database():
    try:
        new_files, chunks = bot.sync_folder()
        return {
            "status": "success",
            "new_files_indexed": new_files,
            "total_chunks_added": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
