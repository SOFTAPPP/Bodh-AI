import os
import json
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.services.bot_service import PDFChatBot
from app.core.config import Config

router = APIRouter()
bot = PDFChatBot()

# In-memory analytics store (replace with DB for production)
_query_log: List[dict] = []
_leads_store: List[dict] = []
_sessions: dict = {}

DEMO_CONTEXTS = {
    "legal": {
        "niche": "legal",
        "label": "Law Firm — Service Agreement",
        "context": """
--- SOURCE: ABC_Law_Firm_Service_Agreement.pdf | Page 1 ---
SERVICE AGREEMENT
This Service Agreement ("Agreement") is entered into as of January 15, 2024, between ABC Law Firm LLP ("Firm") and TechCorp Inc. ("Client").

1. SCOPE OF SERVICES
The Firm agrees to provide the following legal services: contract review, intellectual property protection, corporate governance advisory, and litigation support as requested by the Client.

2. CONFIDENTIALITY OBLIGATIONS
Both parties agree to maintain strict confidentiality regarding all information exchanged under this Agreement. The Client's proprietary data, business strategies, and legal matters shall not be disclosed to any third party without prior written consent. This obligation survives termination of this Agreement for a period of five (5) years.

--- SOURCE: ABC_Law_Firm_Service_Agreement.pdf | Page 2 ---
3. TERMINATION CLAUSES
Either party may terminate this Agreement with thirty (30) days written notice. The Firm may terminate immediately upon non-payment of fees exceeding 60 days. Upon termination, all outstanding invoices become due immediately.

4. LIABILITY AND INDEMNIFICATION
The Firm's total liability shall not exceed the total fees paid in the three (3) months preceding the claim. The Client agrees to indemnify the Firm against any third-party claims arising from the Client's business operations.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York. Any disputes shall be resolved through binding arbitration under AAA rules.

6. PARTIES AND SIGNATURES
ABC Law Firm LLP — Represented by: James Carter, Managing Partner
TechCorp Inc. — Represented by: Sarah Mitchell, CEO
Effective Date: January 15, 2024 | Expiry Date: January 14, 2025
""",
        "sample_questions": [
            "What are the confidentiality obligations in this contract?",
            "Who are the liable parties and what are the penalties?",
            "Summarize all termination clauses in a table",
        ],
    },
    "medical": {
        "niche": "medical",
        "label": "Medical Clinic — Patient Report",
        "context": """
--- SOURCE: Patient_Report_John_Doe.pdf | Page 1 ---
PATIENT CLINICAL REPORT
Patient: John Doe | DOB: 1978-03-14 | MRN: 48291 | Date: 2024-02-20

PRESENTING COMPLAINTS
Patient presents with chest pain (7/10 severity), shortness of breath on exertion, and fatigue for the past 3 weeks.

VITAL SIGNS
Blood Pressure: 152/94 mmHg (elevated ⚠️)
Heart Rate: 88 bpm | Respiratory Rate: 18/min | SpO2: 96% | Temperature: 37.1°C

DIAGNOSES
1. Hypertension Stage 2 (ICD-10: I10)
2. Suspected Coronary Artery Disease — pending stress test
3. Dyslipidemia (ICD-10: E78.5)

--- SOURCE: Patient_Report_John_Doe.pdf | Page 2 ---
MEDICATIONS PRESCRIBED
1. Amlodipine 10mg — once daily (antihypertensive)
2. Atorvastatin 40mg — once daily at bedtime (lipid-lowering)
3. Aspirin 81mg — once daily (antiplatelet)

TREATMENT PLAN
Cardiac stress test scheduled for 2024-03-05. Low-sodium diet recommended. 30 min moderate exercise daily. Follow-up in 4 weeks. Refer to cardiologist if stress test abnormal.

LAB RESULTS
Total Cholesterol: 248 mg/dL ⚠️ | LDL: 162 mg/dL ⚠️ | HDL: 42 mg/dL | Triglycerides: 198 mg/dL ⚠️
Blood Glucose (fasting): 108 mg/dL (borderline)
""",
        "sample_questions": [
            "What are the patient's key diagnoses and conditions?",
            "List all prescribed medications and dosages",
            "Summarize treatment plan and follow-up instructions",
        ],
    },
    "realestate": {
        "niche": "general",
        "label": "Real Estate — Property Listings",
        "context": """
--- SOURCE: Miami_Property_Listings_Q1_2024.pdf | Page 1 ---
PREMIUM PROPERTY LISTINGS — MIAMI, FL — Q1 2024

PROPERTY #1: Brickell City Tower — Unit 1204
Type: 2 Bedroom, 2 Bathroom Apartment
Rent: $2,800/month | Size: 1,050 sq ft | Floor: 12th
Amenities: Gym, Rooftop Pool, Concierge, Valet Parking
Pet Policy: Small pets allowed (under 25 lbs) | Available: March 1, 2024
Lease Terms: 12-month minimum | Security Deposit: 2 months rent

PROPERTY #2: Wynwood Arts Loft — Unit 3B
Type: 1 Bedroom, 1 Bathroom Loft
Rent: $2,100/month | Size: 750 sq ft | Floor: 3rd
Amenities: Art Gallery Lobby, Rooftop Terrace, Co-working Space
Pet Policy: No pets | Available: February 15, 2024
Lease Terms: 6 or 12-month | Security Deposit: 1.5 months rent

--- SOURCE: Miami_Property_Listings_Q1_2024.pdf | Page 2 ---
PROPERTY #3: Coconut Grove Villa — 4BR House
Type: 4 Bedroom, 3 Bathroom Single Family Home
Rent: $5,500/month | Size: 2,800 sq ft | Garden & Pool
Amenities: Private Pool, 2-Car Garage, Smart Home System
Pet Policy: All pets welcome | Available: April 1, 2024
Lease Terms: 12-month minimum | Security Deposit: 3 months rent

PROPERTY #4: Downtown Miami Studio — Unit 807
Type: Studio Apartment
Rent: $1,650/month | Size: 420 sq ft | Floor: 8th
Amenities: Gym, Business Center, 24/7 Security
Pet Policy: No pets | Available: Immediately
Lease Terms: 3, 6, or 12-month | Security Deposit: 1 month rent

PROPERTY #5: Coral Gables Executive Suite — Unit 22A
Type: 2 Bedroom, 2 Bathroom
Rent: $3,200/month | Size: 1,200 sq ft | Floor: 22nd
Amenities: Concierge, Tennis Courts, Spa, Valet
Pet Policy: Cats only | Available: March 15, 2024
Lease Terms: 12-month minimum | Security Deposit: 2 months rent
""",
        "sample_questions": [
            "What properties are available under $3000/month?",
            "List all lease terms and renewal conditions",
            "Summarize all obligations of the tenant vs landlord",
        ],
    },
    "financial": {
        "niche": "financial",
        "label": "Finance — Q4 2023 Audit Report",
        "context": """
--- SOURCE: TechCorp_Q4_2023_Financial_Report.pdf | Page 1 ---
FINANCIAL AUDIT REPORT — Q4 2023
Company: TechCorp Inc. | Auditor: Sterling & Associates CPA | Date: January 30, 2024

INCOME STATEMENT SUMMARY (Oct–Dec 2023)
Total Revenue: $4,820,000
  - Product Sales: $3,150,000
  - Service Revenue: $1,420,000
  - Licensing Fees: $250,000

Total Expenses: $3,940,000
  - Cost of Goods Sold: $1,680,000
  - Salaries & Benefits: $1,420,000
  - Marketing: $380,000
  - R&D: $290,000
  - Administrative: $170,000

Net Income: $880,000 (18.3% margin)

--- SOURCE: TechCorp_Q4_2023_Financial_Report.pdf | Page 2 ---
ANOMALIES AND FLAGS IDENTIFIED ⚠️
1. Marketing Expense Spike: $380,000 in Q4 vs $140,000 average in Q1-Q3. No supporting campaign invoices provided.
2. Unreconciled Transaction: $47,500 wire transfer on Nov 14 to vendor "Global Supplies Ltd" — no PO or contract on file.
3. Accounts Receivable: $320,000 outstanding > 90 days from 3 clients. Risk of write-off.

OUTSTANDING INVOICES
- Client A (Acme Corp): $125,000 — Due Oct 15, 2023 (overdue 107 days)
- Client B (Nexus LLC): $98,000 — Due Nov 1, 2023 (overdue 90 days)
- Client C (Vertex Ltd): $97,000 — Due Nov 10, 2023 (overdue 81 days)

AUDITOR RECOMMENDATION
Overall risk rating: MEDIUM. Immediate clarification required on marketing spend and unreconciled wire transfer before filing annual report.
""",
        "sample_questions": [
            "Identify any financial anomalies or red flags",
            "Summarize total revenue, expenses, and net profit",
            "List all outstanding invoices and payment terms",
        ],
    },
}


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    active_file: Optional[str] = None
    niche: Optional[str] = None
    demo_mode: Optional[bool] = False
    session_id: Optional[str] = None


class LeadRequest(BaseModel):
    name: str
    email: str
    business_type: str
    message: Optional[str] = None

from fastapi import Form

@router.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    upload_path = os.path.join(bot.doc_service.upload_dir, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    indexed_files = bot.doc_service.get_indexed_files()
    if file.filename.lower() in indexed_files:
        return {"filename": file.filename.lower(), "status": "already_indexed",
                "message": "File is already in the knowledge base."}

    await bot.process_new_pdf(upload_path)
    
    # Auto-switch active document for the session
    if session_id:
        if session_id not in _sessions:
            _sessions[session_id] = {"active_file": None}
        _sessions[session_id]["active_file"] = file.filename.lower()

    return {"filename": file.filename.lower(), "status": "ready",
            "message": f"'{file.filename}' has been fully indexed and is now active! You can start asking questions about it."}


@router.get("/files")
async def get_files():
    """Returns only web-platform indexed documents. WhatsApp files are strictly isolated."""
    return {"files": bot.get_indexed_files()}


@router.post("/chat")
async def chat(request: ChatRequest):
    import time
    from datetime import datetime
    t_start = time.perf_counter()
    print(f"=== [WEB APP /CHAT START] Time: {datetime.utcnow().isoformat()} | Query: {request.message} ===")

    # Fast-path for simple chitchat/greetings
    q = request.message.lower().strip().strip("?").strip("!").strip(".")
    chitchat_phrases = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how's it going", "howdy", "who are you", "what is your name", "what can you do",
        "thanks", "thank you", "bye", "goodbye", "help", "what's up", "sup"
    }
    if q in chitchat_phrases or (len(q.split()) <= 2 and any(w in q for w in ["hi", "hello", "hey", "thanks", "thank", "bye"])):
        async def chitchat_stream():
            print(f"--- [WEB APP CHITCHAT] Bypassing FAISS. Starting generation stream... ---")
            t_gen_start = time.perf_counter()
            first = True
            prompt_doc = (request.active_file is None)
            async for chunk in bot.llm_service.generate_chitchat_response(request.message, request.history, prompt_for_document=prompt_doc):
                if first:
                    print(f"--- [WEB APP CHITCHAT] First Token: {time.perf_counter() - t_gen_start:.4f}s ---")
                    first = False
                yield chunk
            print(f"=== [WEB APP /CHAT END] Total Time: {time.perf_counter() - t_start:.4f}s ===")
        return StreamingResponse(chitchat_stream(), media_type="text/plain")

    # Check if the last assistant message in history was asking to choose another document
    last_assistant_msg = ""
    for turn in reversed(request.history):
        if turn.get("role") in ("assistant", "bot"):
            last_assistant_msg = turn.get("content", "")
            break

    is_not_found_fallback = "I could not find this information in the selected document" in last_assistant_msg
    is_ambiguous_fallback = "I found multiple documents" in last_assistant_msg
    is_fallback_selection = is_not_found_fallback or is_ambiguous_fallback

    # Resolve session and active file using ONLY web-platform files (strict isolation)
    session_id = request.session_id or "default"
    if session_id not in _sessions:
        _sessions[session_id] = {"active_file": None}

    all_files = bot.get_indexed_files()  # web-only

    # If the UI actively passes an active_file, we MIGHT sync it, but prompt says DO NOT trust frontend-only state.
    # However, since the user wants to avoid manual switching, we rely on _sessions.
    previous_active_file = _sessions[session_id].get("active_file")
    active_file = previous_active_file

    if is_fallback_selection and active_file:
        if is_not_found_fallback:
            target_list = [f for f in all_files if f.lower() != active_file.lower()]
        else:
            target_list = all_files
        active_file = None  # Clear to force resolution
        _sessions[session_id]["active_file"] = None
    else:
        target_list = all_files
        
    prepend_msg = ""
    is_selection_retry = False
    
    if not active_file and target_list:
        text_clean = request.message.strip()
        selected_file = None
        
        # 1. Check numeric selection
        if text_clean.isdigit():
            idx = int(text_clean) - 1
            if 0 <= idx < len(target_list):
                selected_file = target_list[idx]
                is_selection_retry = True
        else:
            # 2. Check for exact or partial filename match (case-insensitive)
            matches = [f for f in target_list if text_clean.lower() in f.lower()]
            if len(matches) == 1:
                selected_file = matches[0]
                is_selection_retry = True
            
            # 3. Smart Document Detection based on query keywords
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
            _sessions[session_id]["active_file"] = active_file
            
            if is_selection_retry:
                # Find user's original query before the selection prompt
                original_query = None
                for turn in reversed(request.history):
                    if turn.get("role") == "user":
                        content = turn.get("content", "").strip()
                        if not content.isdigit() and len(content) > 3:
                            original_query = content
                            break
                
                if original_query:
                    request.message = original_query
                else:
                    request.message = "what is this document all about ?"
                    
                prepend_msg = f"✅ **{active_file}** selected. \n\n"


    # Pass active niche hint if present to skip heuristic classification
    _query_log.append({
        "ts": datetime.utcnow().isoformat(),
        "niche": request.niche or "auto",
        "query": request.message,
        "demo": False,
    })
    
    # FIX: Clear old conversation chain if document changed
    effective_history = request.history
    if active_file and previous_active_file and active_file.lower() != previous_active_file.lower():
        print(f"--- [SESSION] Active document changed from {previous_active_file} to {active_file}. Clearing history context. ---")
        effective_history = []

    async def bot_stream():
        t_gen_start = time.perf_counter()
        if prepend_msg:
            yield prepend_msg
        first = True
        async for chunk in bot.ask(
            request.message,
            effective_history,
            active_file,
            niche_hint=request.niche,
            is_selection_retry=is_selection_retry,
        ):
            if first:
                print(f"--- [WEB APP RAG] First Token: {time.perf_counter() - t_gen_start:.4f}s ---")
                first = False
            yield chunk
        print(f"=== [WEB APP /CHAT END] Total Time: {time.perf_counter() - t_start:.4f}s ===")

    # Determine the effective active_file for the response header
    # (includes auto-selected single-doc case handled inside bot.ask())
    effective_file = active_file
    if not effective_file and len(all_files) == 1:
        effective_file = all_files[0]
        _sessions[session_id]["active_file"] = effective_file

    headers = {}
    if effective_file:
        headers["X-Active-File"] = effective_file

    return StreamingResponse(bot_stream(), media_type="text/plain", headers=headers)


@router.get("/demo/{niche}")
async def get_demo_context(niche: str):
    """Returns sample questions and metadata for a given niche demo."""
    if niche not in DEMO_CONTEXTS:
        raise HTTPException(status_code=404, detail=f"No demo for niche '{niche}'")
    demo = DEMO_CONTEXTS[niche]
    return {
        "niche": niche,
        "label": demo["label"],
        "sample_questions": demo["sample_questions"],
    }


@router.post("/leads")
async def capture_lead(lead: LeadRequest):
    """Stores a lead from the demo contact form."""
    entry = {
        "id": len(_leads_store) + 1,
        "ts": datetime.utcnow().isoformat(),
        "name": lead.name,
        "email": lead.email,
        "business_type": lead.business_type,
        "message": lead.message or "",
    }
    _leads_store.append(entry)

    # Also persist to a JSON file so leads survive server restart
    leads_file = os.path.join(Config.DATA_DIR, "leads.json")
    try:
        existing = []
        if os.path.exists(leads_file):
            with open(leads_file, "r") as f:
                existing = json.load(f)
        existing.append(entry)
        with open(leads_file, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"--- Lead save error: {e} ---")

    return {"status": "received", "id": entry["id"], "message": "Thank you! We'll be in touch."}


@router.get("/leads")
async def get_leads():
    """Returns all captured leads (admin use)."""
    leads_file = os.path.join(Config.DATA_DIR, "leads.json")
    if os.path.exists(leads_file):
        with open(leads_file, "r") as f:
            return {"leads": json.load(f), "total": len(json.load(open(leads_file)))}
    return {"leads": _leads_store, "total": len(_leads_store)}


@router.get("/analytics")
async def get_analytics():
    """Returns query stats — most popular niches, total queries, demo vs real."""
    total = len(_query_log)
    niche_counts: dict = {}
    demo_count = 0
    for q in _query_log:
        n = q.get("niche", "unknown")
        niche_counts[n] = niche_counts.get(n, 0) + 1
        if q.get("demo"):
            demo_count += 1

    return {
        "total_queries": total,
        "demo_queries": demo_count,
        "real_queries": total - demo_count,
        "by_niche": niche_counts,
        "recent": _query_log[-10:][::-1],
    }


@router.post("/sync")
async def sync():
    new_files = await bot.sync_data_folder()
    return {"status": "success", "new_files_indexed": new_files}
