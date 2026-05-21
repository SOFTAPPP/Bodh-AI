from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse
from app.services.whatsapp_service import WhatsAppService
from app.services.bot_service import PDFChatBot
from app.services.session_service import SessionManager
import asyncio
import os

router = APIRouter()
wa_service = WhatsAppService()

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

    if "entry" in body and body["entry"]:
        entry = body["entry"][0]
        if "changes" in entry and entry["changes"]:
            change = entry["changes"][0]
            value = change.get("value", {})

            if "messages" in value and value["messages"]:
                message = value["messages"][0]
                phone_number = message.get("from")

                print(f"=== [WHATSAPP WEBHOOK RECEIVED] Time: {datetime.utcnow().isoformat()} | Phone: {phone_number} ===")

                background_tasks.add_task(process_whatsapp_message, phone_number, message, t_web_received)

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

        from app.core.config import Config
        os.makedirs(Config.WA_UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.WA_VECTOR_STORE_DIR, exist_ok=True)

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

                confirm_task = wa_service.send_message(
                    phone_number,
                    f"Received '{filename}'. I am downloading and indexing it for you now—this will take just a few seconds. I will let you know the moment it is ready!"
                )
                download_task = wa_service.download_media(media_id, filename)

                _, filepath = await asyncio.gather(confirm_task, download_task)
                print(f"--- [PERF] Concurrently sent confirmation and downloaded media in: {time.perf_counter() - t_parallel_start:.4f}s ---")

                if not filepath:
                    await wa_service.send_message(phone_number, f"Failed to download '{filename}'. Please try uploading it again.")
                    return

                t_idx_start = time.perf_counter()
                chunks = await wa_bot.process_new_pdf(filepath, session_id=phone_number)
                print(f"--- [WHATSAPP PDF indexed] Time taken to index embeddings: {time.perf_counter() - t_idx_start:.4f}s | Chunks: {chunks} ---")

                if chunks > 0:
                    await wa_service.set_user_file(phone_number, filename)

                    SessionManager.switch_document(
                        platform="whatsapp",
                        session_id=phone_number,
                        new_document=filename,
                        bot_instance=wa_bot
                    )
                    t_success_send = time.perf_counter()
                    await wa_service.send_message(
                        phone_number,
                        f"'{filename}' has been fully indexed and is now active! You can start asking questions about it."
                    )
                    print(f"--- [WHATSAPP PDF success msg sent] Time taken to send success message: {time.perf_counter() - t_success_send:.4f}s ---")

                    caption = document.get("caption")
                    if caption:
                        print(f"--- [WHATSAPP PDF caption detected] Processing caption: {caption} ---")
                        await handle_text_message(phone_number, caption, t_web_received)
                else:
                    await wa_service.send_message(phone_number, f"Failed to extract text from '{filename}'. It might be corrupted or protected.")
            except Exception as e:
                print(f"--- [ERROR] Background document processing failed: {e} ---")
                await wa_service.send_message(phone_number, f"An error occurred while processing '{filename}'.")

            print(f"=== [WHATSAPP BG COMPLETED] Total processing time: {time.perf_counter() - t_bg_start:.4f}s ===")

async def handle_text_message(phone_number: str, text: str, t_web_received: float):
    import time
    t_llm_start = time.perf_counter()
    print(f"--- [WHATSAPP TEXT ROUTE] Loading history & active context... ---")

    history = wa_service.get_user_history(phone_number)
    active_file = wa_service.get_user_file(phone_number)
    previous_active_file = active_file

    session = SessionManager.switch_document(
        platform="whatsapp",
        session_id=phone_number,
        new_document=active_file,
        bot_instance=wa_bot
    )

    last_assistant_msg = ""
    for turn in reversed(history):
        if turn.get("role") in ("assistant", "bot"):
            last_assistant_msg = turn.get("content", "")
            break

    is_not_found_fallback = "This information was not found in the selected document" in last_assistant_msg
    is_ambiguous_fallback = "Which document are you referring to?" in last_assistant_msg
    is_fallback_selection = is_not_found_fallback or is_ambiguous_fallback

    all_files = wa_bot.get_indexed_files()

    if is_fallback_selection and active_file:
        if is_not_found_fallback:
            target_list = [f for f in all_files if f.lower() != active_file.lower()]
        else:
            target_list = all_files
        active_file = None
        await wa_service.set_user_file(phone_number, None)
        SessionManager.switch_document(
            platform="whatsapp",
            session_id=phone_number,
            new_document=None,
            bot_instance=wa_bot
        )
    else:
        target_list = all_files

    prepend_msg = ""
    is_selection_retry = False

    if is_fallback_selection and target_list:
        text_clean = text.strip()
        selected_file = None

        if text_clean.isdigit():
            idx = int(text_clean) - 1
            if 0 <= idx < len(target_list):
                selected_file = target_list[idx]
                is_selection_retry = True
        else:
            matches = [f for f in target_list if text_clean.lower() in f.lower()]
            if len(matches) == 1:
                selected_file = matches[0]
                is_selection_retry = True

            if not selected_file and len(target_list) > 1:
                q_words = set(text_clean.lower().split())
                for f in target_list:
                    f_name_lower = f.lower()
                    f_name_no_ext = f_name_lower.replace(".pdf", "")
                    if any(w in f_name_lower for w in q_words if len(w) > 3) or text_clean.lower() in f_name_no_ext:
                        selected_file = f
                        break

        if selected_file:
            active_file = selected_file
            await wa_service.set_user_file(phone_number, selected_file)
            session = SessionManager.switch_document(
                platform="whatsapp",
                session_id=phone_number,
                new_document=active_file,
                bot_instance=wa_bot
            )

            if is_selection_retry:
                original_query = None
                for turn in reversed(history):
                    if turn.get("role") == "user":
                        content = turn.get("content", "").strip()
                        if not content.isdigit() and len(content) > 3:
                            original_query = content
                            break

                if original_query:
                    text = original_query
                else:
                    text = "what is this document all about ?"

                prepend_msg = f"**{active_file}** selected. \n\n"

    if not is_selection_retry and all_files:
        intent = await wa_bot.llm_service.analyze_query(text, history)
        standalone_query = intent.get("standalone_query", text)

        route_res = await wa_bot.route_query(standalone_query, all_files)
        confidence = route_res["confidence"]
        best_doc = route_res["best_doc"]

        if confidence == "HIGH":
            if not active_file or active_file.lower() != best_doc.lower():
                prepend_msg = f"*(Auto-switched context to {best_doc})*\n\n"
                print(f"--- [ROUTE SYNCHRONIZER] Auto-switching WhatsApp from {active_file} to {best_doc} ---")
                active_file = best_doc
                await wa_service.set_user_file(phone_number, active_file)
                session = SessionManager.switch_document(
                    platform="whatsapp",
                    session_id=phone_number,
                    new_document=active_file,
                    bot_instance=wa_bot
                )
        elif confidence == "MEDIUM" or (confidence == "LOW" and not active_file):
            print(f"--- [ROUTE SYNCHRONIZER] WhatsApp Ambiguity or Low Confidence ({confidence}) detected. Prompting user. ---")
            msg = "Which document are you referring to?\n\n"
            for idx, f in enumerate(all_files, 1):
                msg += f"{idx}. **{f}**\n"
            msg += "\nPlease select a document by replying with its number or name."
            await wa_service.set_user_file(phone_number, None)
            SessionManager.switch_document(
                platform="whatsapp",
                session_id=phone_number,
                new_document=None,
                bot_instance=wa_bot
            )
            await wa_service.add_conversation_turn(phone_number, text, msg)
            await wa_service.send_message(phone_number, msg)
            return

    print(f"--- [WHATSAPP TEXT ROUTE] Starting LLM ask() generator... ---")

    response_chunks = []
    if prepend_msg:
        response_chunks.append(prepend_msg)

    effective_history = history
    if active_file and previous_active_file and active_file.lower() != previous_active_file.lower():
        effective_history = []

    try:
        async for chunk in wa_bot.ask(
            query=text,
            history=effective_history,
            active_file=active_file,
            is_selection_retry=is_selection_retry,
            session_id=phone_number
        ):
            response_chunks.append(chunk)

        full_response = "".join(response_chunks)
        if not full_response.strip():
            full_response = "I couldn't generate a response based on the document."

    except Exception as e:
        print(f"--- [ERROR] RAG generation failed: {e} ---")
        full_response = "An error occurred while analyzing the document."

    t_llm_done = time.perf_counter()
    print(f"--- [WHATSAPP TEXT ROUTE] LLM Answer generated! Time taken: {t_llm_done - t_llm_start:.4f}s ---")

    await wa_service.add_conversation_turn(phone_number, text, full_response)

    session.memory = wa_service.get_user_history(phone_number)

    chunk_size = 4000
    t_send_start = time.perf_counter()
    for i in range(0, len(full_response), chunk_size):
        await wa_service.send_message(phone_number, full_response[i:i+chunk_size])
    t_send_done = time.perf_counter()

    print(f"--- [WHATSAPP TEXT ROUTE] Message sent back to Meta! Time taken: {t_send_done - t_send_start:.4f}s ---")
    print(f"=== [WHATSAPP TEXT RTT COMPLETE] Total round-trip time since received: {time.perf_counter() - t_web_received:.4f}s ===")
