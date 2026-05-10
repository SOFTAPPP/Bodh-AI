import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.services.bot_service import PDFChatBot
from app.core.config import Config

router = APIRouter()
bot = PDFChatBot()

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    active_file: Optional[str] = None

@router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    upload_path = os.path.join(Config.UPLOAD_DIR, file.filename)
    
    # Save file instantly
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Check if already indexed
    indexed_files = bot.doc_service.get_indexed_files()
    if file.filename.lower() in indexed_files:
        return {
            "filename": file.filename,
            "status": "already_indexed",
            "message": "File is already in the knowledge base."
        }

    # Parallel Background Ingestion
    background_tasks.add_task(bot.process_new_pdf, upload_path)
    
    return {
        "filename": file.filename,
        "status": "processing",
        "message": "File upload complete. Indexing in background..."
    }

@router.get("/files")
async def get_files():
    files = list(bot.doc_service.get_indexed_files())
    return {"files": files}

@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        bot.ask(request.message, request.history, request.active_file),
        media_type="text/plain"
    )

@router.post("/sync")
async def sync():
    new_files = await bot.sync_data_folder()
    return {"status": "success", "new_files_indexed": new_files}
