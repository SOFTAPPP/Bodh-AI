from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse
from app.services.whatsapp_service import WhatsAppService
from app.services.bot_service import PDFChatBot
import asyncio

router = APIRouter()
wa_service = WhatsAppService()
# Completely isolated bot — uses wa_uploads/ and wa_vector_store/, invisible to the web app
wa_bot = PDFChatBot(platform="whatsapp")

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
    import time
    from datetime import datetime
    t_web_received = time.perf_counter()
    
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
                
                print(f"=== [WHATSAPP WEBHOOK RECEIVED] Time: {datetime.utcnow().isoformat()} | Phone: {phone_number} ===")
                # Background processing so we can return 200 OK immediately to Meta
                background_tasks.add_task(process_whatsapp_message, phone_number, message, t_web_received)

    # Always return 200 OK to Meta
    return {"status": "ok"}
user_locks = {}

def get_user_lock(phone_number: str) -> asyncio.Lock:
    if phone_number not in user_locks:
        user_locks[phone_number] = asyncio.Lock()
    return user_locks[phone_number]

async def process_whatsapp_message(phone_number: str, message: dict, t_web_received: float):
    lock = get_user_lock(phone_number)
    async with lock:
        import time
        t_bg_start = time.perf_counter()
        print(f"--- [WHATSAPP BG DISPATCH] Time since Webhook received: {t_bg_start - t_web_received:.4f}s ---")
    
        msg_type = message.get("type")
        
        if msg_type == "text":
            text = message["text"]["body"]
            await handle_text_message(phone_number, text, t_web_received)
        
        elif msg_type == "document":
            document = message["document"]
            media_id = document.get("id")
            filename = document.get("filename", f"wa_doc_{media_id}.pdf")
            
            if not filename.endswith(".pdf"):
                await wa_service.send_message(phone_number, "Sorry, I can only process PDF documents right now.")
                return
                
            print(f"--- [WHATSAPP PDF UPLOAD DETECTED] File: {filename} ---")
            
            try:
                t_parallel_start = time.perf_counter()
                # Run sending of confirmation message and media download in parallel!
                confirm_task = wa_service.send_message(
                    phone_number, 
                    f"📥 Received '{filename}'. I am downloading and indexing it for you now—this will take just a few seconds. I will let you know the moment it is ready!"
                )
                download_task = wa_service.download_media(media_id, filename)
                
                _, filepath = await asyncio.gather(confirm_task, download_task)
                print(f"--- [PERF] Concurrently sent confirmation and downloaded media in: {time.perf_counter() - t_parallel_start:.4f}s ---")
                
                if not filepath:
                    await wa_service.send_message(phone_number, f"❌ Failed to download '{filename}'. Please try uploading it again.")
                    return
                    
                t_idx_start = time.perf_counter()
                chunks = await wa_bot.process_new_pdf(filepath)
                print(f"--- [WHATSAPP PDF indexed] Time taken to index embeddings: {time.perf_counter() - t_idx_start:.4f}s | Chunks: {chunks} ---")
                
                if chunks > 0:
                    await wa_service.set_user_file(phone_number, filename)
                    t_success_send = time.perf_counter()
                    await wa_service.send_message(
                        phone_number, 
                        f"✅ '{filename}' has been fully indexed and is now active! You can start asking questions about it."
                    )
                    print(f"--- [WHATSAPP PDF success msg sent] Time taken to send success message: {time.perf_counter() - t_success_send:.4f}s ---")
                    
                    caption = document.get("caption")
                    if caption:
                        print(f"--- [WHATSAPP PDF caption detected] Processing caption: {caption} ---")
                        await handle_text_message(phone_number, caption, t_web_received)
                else:
                    await wa_service.send_message(phone_number, f"❌ Failed to extract text from '{filename}'. It might be corrupted or protected.")
            except Exception as e:
                print(f"--- [ERROR] Background document processing failed: {e} ---")
                await wa_service.send_message(phone_number, f"❌ An error occurred while processing '{filename}'.")
                
            print(f"=== [WHATSAPP BG COMPLETED] Total processing time: {time.perf_counter() - t_bg_start:.4f}s ===")

async def handle_text_message(phone_number: str, text: str, t_web_received: float):
    import time
    t_llm_start = time.perf_counter()
    print(f"--- [WHATSAPP TEXT ROUTE] Loading history & active context... ---")

    # Load history & active file BEFORE saving the new user message
    # (history snapshot used for LLM context, saved together with response below)
    history = wa_service.get_user_history(phone_number)
    active_file = wa_service.get_user_file(phone_number)

    web_files = sorted(list(wa_bot.doc_service_web.get_indexed_files()))
    wa_files = sorted(list(wa_bot.doc_service_wa.get_indexed_files()))
    
    seen = set()
    all_files = []
    for f in (web_files + wa_files):
        if f not in seen:
            seen.add(f)
            all_files.append(f)
            
    prepend_msg = ""
    
    if not active_file and all_files:
        text_clean = text.strip()
        selected_file = None
        if text_clean.isdigit():
            idx = int(text_clean) - 1
            if 0 <= idx < len(all_files):
                selected_file = all_files[idx]
        else:
            # Check for exact or partial filename match (case-insensitive)
            matches = [f for f in all_files if text_clean.lower() in f]
            if len(matches) == 1:
                selected_file = matches[0]
                
        if selected_file:
            active_file = selected_file
            await wa_service.set_user_file(phone_number, selected_file)
            text = "what is this document all about ??"

    print(f"--- [WHATSAPP TEXT ROUTE] Starting LLM ask() generator... ---")
    # Process RAG Query via bot
    # Note: bot.ask returns an AsyncIterable[str]. We accumulate the chunks.
    response_chunks = []
    if prepend_msg:
        response_chunks.append(prepend_msg)
        
    try:
        async for chunk in wa_bot.ask(query=text, history=history, active_file=active_file):
            response_chunks.append(chunk)

        full_response = "".join(response_chunks)
        if not full_response.strip():
            full_response = "I couldn't generate a response based on the document."

    except Exception as e:
        print(f"--- [ERROR] RAG generation failed: {e} ---")
        full_response = "An error occurred while analyzing the document."

    t_llm_done = time.perf_counter()
    print(f"--- [WHATSAPP TEXT ROUTE] LLM Answer generated! Time taken: {t_llm_done - t_llm_start:.4f}s ---")

    # Save both turns atomically in a single disk write
    await wa_service.add_conversation_turn(phone_number, text, full_response)

    # Send response — split if > 4096 chars (WhatsApp limit)
    chunk_size = 4000
    t_send_start = time.perf_counter()
    for i in range(0, len(full_response), chunk_size):
        await wa_service.send_message(phone_number, full_response[i:i+chunk_size])
    t_send_done = time.perf_counter()

    print(f"--- [WHATSAPP TEXT ROUTE] Message sent back to Meta! Time taken: {t_send_done - t_send_start:.4f}s ---")
    print(f"=== [WHATSAPP TEXT RTT COMPLETE] Total round-trip time since received: {time.perf_counter() - t_web_received:.4f}s ===")
