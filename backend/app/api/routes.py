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

# ── In-memory analytics store (replace with DB for production) ────────────────
_query_log: List[dict] = []
_leads_store: List[dict] = []

# ── Sample demo contexts per niche (no PDF upload required) ───────────────────
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


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    active_file: Optional[str] = None
    niche: Optional[str] = None       # NEW: pre-seed domain from frontend
    demo_mode: Optional[bool] = False  # NEW: use hardcoded demo context


class LeadRequest(BaseModel):
    name: str
    email: str
    business_type: str
    message: Optional[str] = None


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    upload_path = os.path.join(Config.UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    indexed_files = bot.doc_service.get_indexed_files()
    if file.filename.lower() in indexed_files:
        return {"filename": file.filename, "status": "already_indexed",
                "message": "File is already in the knowledge base."}

    background_tasks.add_task(bot.process_new_pdf, upload_path)
    return {"filename": file.filename, "status": "processing",
            "message": "File upload complete. Indexing in background..."}


# ── Files ─────────────────────────────────────────────────────────────────────

@router.get("/files")
async def get_files():
    files = list(bot.doc_service.get_indexed_files())
    return {"files": files}


# ── Chat (with niche + demo_mode support) ─────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    # Demo mode: inject hardcoded context instead of FAISS search
    if request.demo_mode and request.niche and request.niche in DEMO_CONTEXTS:
        demo = DEMO_CONTEXTS[request.niche]
        async def demo_stream():
            async for chunk in bot.llm_service.generate_response(
                request.message,
                demo["context"],
                domain=demo["niche"],
                intent="fact_extraction",
            ):
                yield chunk

        # Log analytics
        _query_log.append({
            "ts": datetime.utcnow().isoformat(),
            "niche": request.niche,
            "query": request.message,
            "demo": True,
        })

        return StreamingResponse(demo_stream(), media_type="text/plain")

    # Normal RAG mode — pass niche hint to skip heuristic classification
    _query_log.append({
        "ts": datetime.utcnow().isoformat(),
        "niche": request.niche or "auto",
        "query": request.message,
        "demo": False,
    })

    return StreamingResponse(
        bot.ask(
            request.message,
            request.history,
            request.active_file,
            niche_hint=request.niche,
        ),
        media_type="text/plain",
    )


# ── Demo Context ──────────────────────────────────────────────────────────────

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


# ── Lead Capture ──────────────────────────────────────────────────────────────

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


# ── Analytics ─────────────────────────────────────────────────────────────────

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


# ── Sync ──────────────────────────────────────────────────────────────────────

@router.post("/sync")
async def sync():
    new_files = await bot.sync_data_folder()
    return {"status": "success", "new_files_indexed": new_files}
