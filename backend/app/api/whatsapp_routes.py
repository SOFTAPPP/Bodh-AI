from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse
from app.services.whatsapp_service import WhatsAppService
from app.api.routes import bot
import asyncio

router = APIRouter()
wa_service = WhatsAppService()

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """Webhook verification for Meta."""
    if hub_mode == "subscribe" and hub_verify_token == wa_service.verify_token:
        print("--- Webhook verified! ---")
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")

@router.post("/webhook")
async def webhook_events(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming messages and events from WhatsApp."""
    try:
        body = await request.json()
    except Exception:
        return {"status": "ok"}
    
    # Check if it's a WhatsApp status update (message delivered/read, etc.)
    if "entry" in body and body["entry"]:
        entry = body["entry"][0]
        if "changes" in entry and entry["changes"]:
            change = entry["changes"][0]
            value = change.get("value", {})
            
            # Extract messages
            if "messages" in value and value["messages"]:
                message = value["messages"][0]
                phone_number = message.get("from")
                
                # Background processing so we can return 200 OK immediately to Meta
                background_tasks.add_task(process_whatsapp_message, phone_number, message)

    # Always return 200 OK to Meta
    return {"status": "ok"}

async def process_whatsapp_message(phone_number: str, message: dict):
    msg_type = message.get("type")
    
    if msg_type == "text":
        text = message["text"]["body"]
        await handle_text_message(phone_number, text)
        
    elif msg_type == "document":
        document = message["document"]
        media_id = document.get("id")
        filename = document.get("filename", f"wa_doc_{media_id}.pdf")
        
        if not filename.endswith(".pdf"):
            await wa_service.send_message(phone_number, "Sorry, I can only process PDF documents right now.")
            return
            
        await wa_service.send_message(phone_number, f"Downloading '{filename}'...")
        filepath = await wa_service.download_media(media_id, filename)
        
        if filepath:
            await wa_service.send_message(phone_number, "Document downloaded. Indexing... Please wait.")
            # Run index
            chunks = await bot.process_new_pdf(filepath)
            if chunks > 0:
                await wa_service.set_user_file(phone_number, filename)
                await wa_service.send_message(phone_number, f"✅ Successfully indexed '{filename}'. You can now ask questions about it.")
            else:
                await wa_service.send_message(phone_number, "❌ Failed to process the document. It might be already indexed or corrupted.")
        else:
            await wa_service.send_message(phone_number, "❌ Failed to download the document.")

async def handle_text_message(phone_number: str, text: str):
    # Load user history and active file
    history = wa_service.get_user_history(phone_number)
    active_file = wa_service.get_user_file(phone_number)
    
    await wa_service.add_user_history(phone_number, "user", text)
    
    # Process RAG Query via bot
    # Note: bot.ask returns an AsyncIterable[str]. We accumulate the chunks.
    response_chunks = []
    try:
        async for chunk in bot.ask(query=text, history=history, active_file=active_file):
            response_chunks.append(chunk)
            
        full_response = "".join(response_chunks)
        if not full_response.strip():
            full_response = "I couldn't generate a response based on the document."
            
    except Exception as e:
        print(f"--- [ERROR] RAG generation failed: {e} ---")
        full_response = "An error occurred while analyzing the document."
        
    await wa_service.add_user_history(phone_number, "assistant", full_response)
    
    # Send chunks if response is too long, or just one message
    # WhatsApp max message length is 4096 chars.
    chunk_size = 4000
    for i in range(0, len(full_response), chunk_size):
        await wa_service.send_message(phone_number, full_response[i:i+chunk_size])
