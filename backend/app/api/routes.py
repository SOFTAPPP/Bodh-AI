import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
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
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    upload_path = os.path.join(Config.UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = await bot.process_new_pdf(upload_path)
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_created": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
