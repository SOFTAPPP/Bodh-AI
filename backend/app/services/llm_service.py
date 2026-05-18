import re
import json
import time
from typing import List, Dict, AsyncIterable, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import Config


class LLMService:
    """
    Groq-powered LLM service with:
    - llama-3.3-70b-versatile  → primary generation (ultra-low latency)
    - deepseek-r1-distill-llama-70b → complex reasoning
    - gemma2-9b-it             → rate-limit fallback + HyDE generation
    - Token streaming for immediate output
    - Inline query analysis (zero extra LLM call for simple queries)
    - HyDE (Hypothetical Document Embeddings) generation
    - Enhanced domain-specific response protocols
    """

    def __init__(self):
        common = dict(
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0,
            streaming=True,
        )
        self.generation_llm = ChatGroq(model_name=Config.GENERATION_MODEL, **common)
        self.reasoning_llm  = ChatGroq(model_name=Config.REASONING_MODEL,  **common)
        self.fallback_llm   = ChatGroq(model_name=Config.FALLBACK_MODEL,   **common)

        # Non-streaming LLM for HyDE generation (faster without streaming overhead)
        hyde_common = dict(
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0,
            streaming=False,
        )
        self.hyde_llm = ChatGroq(model_name=Config.HYDE_MODEL, **hyde_common)

        print(f"--- LLMService: Groq engine ready [{Config.GENERATION_MODEL}] ---")

    async def generate_hypothetical_answer(self, query: str, domain: str) -> Optional[str]:
        """
        Generates a short hypothetical answer for the query.
        This is embedded and used for FAISS search instead of the raw query,
        bridging the semantic gap between short queries and long document chunks.

        Uses gemma2-9b-it (lightweight) for sub-100ms generation.
        """
        if not Config.HYDE_ENABLED:
            return None

        hyde_prompt = (
            f"You are a {domain} document analyst. Given a user query, write a short "
            f"hypothetical paragraph that would appear in a relevant document answering "
            f"this query. Write it as if it's an excerpt from the actual document.\n\n"
            f"Query: {query}\n\n"
            f"Hypothetical document excerpt (2-3 sentences, factual tone):"
        )

        try:
            resp = await self.hyde_llm.ainvoke(
                [HumanMessage(content=hyde_prompt)],
                config={"max_tokens": 150},
            )
            hyde_text = resp.content.strip()
            print(f"--- HyDE: Generated hypothetical answer ({len(hyde_text)} chars) ---")
            return hyde_text
        except Exception as e:
            print(f"--- HyDE generation error: {e}. Skipping HyDE. ---")
            return None

    async def analyze_query(self, query: str, history: List[Dict[str, str]]) -> Dict:
        """
        Ultra-fast query analysis using Groq (gemma2-9b-it for minimal latency).
        Returns structured intent + rephrased standalone query + domain.

        Skips the LLM entirely for single-turn queries — uses heuristics instead.
        """
        # ── Fast-path: no history → classify via heuristics (0ms overhead) ──
        if not history:
            return self._heuristic_classify(query)

        system_prompt = (
            "You are a RAG query classifier. Output ONLY valid JSON, no markdown fences:\n"
            '{"standalone_query": "...", "domain": "legal|medical|hr|financial|academic|general", '
            '"intent": "fact_extraction|complex_analysis"}\n\n'
            "RULES:\n"
            "1. standalone_query: incorporate conversation context so it can be searched alone.\n"
            "2. intent: fact_extraction = simple lookup; complex_analysis = reasoning/comparison/why/how.\n"
            "3. domain: primary subject of the query."
        )

        last = history[-1]
        human_msg = f"PREV: {last['content'][:100]}\nQ: {query}"

        try:
            # Use smallest model for sub-100ms analysis
            resp = await self.fallback_llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)],
                config={"max_tokens": 120},
            )
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"--- analyze_query error: {e}. Using heuristics. ---")

        return self._heuristic_classify(query)

    def _heuristic_classify(self, query: str) -> Dict:
        """Zero-latency heuristic classifier (no LLM call)."""
        q = query.lower()
        complex_kw = {"why", "how", "compare", "contrast", "analyze", "evaluate",
                      "difference", "impact", "implication", "relationship", "contradict"}
        intent = "complex_analysis" if any(kw in q for kw in complex_kw) else "fact_extraction"

        domain = "general"
        if any(kw in q for kw in ["court", "plaintiff", "defendant", "judgment", "legal", "law", "clause", "section", "advocate", "appeal", "verdict"]):
            domain = "legal"
        elif any(kw in q for kw in ["patient", "diagnosis", "treatment", "clinical", "medical", "hospital", "drug", "prescription", "dosage", "therapy"]):
            domain = "medical"
        elif any(kw in q for kw in ["resume", "cv", "hire", "skills", "candidate", "experience", "job", "certification"]):
            domain = "hr"
        elif any(kw in q for kw in ["revenue", "invoice", "tax", "profit", "financial", "audit", "balance sheet", "transaction"]):
            domain = "financial"
        elif any(kw in q for kw in ["research", "methodology", "abstract", "hypothesis", "study", "findings", "literature"]):
            domain = "academic"

        return {"standalone_query": query, "domain": domain, "intent": intent}

    def _get_domain_protocol(self, domain: str, intent: str = "fact_extraction") -> str:
        """
        Returns an enhanced, structured domain-specific system prompt.
        Each domain has a precise output structure for maximum precision.
        """
        protocols = {
            "legal": (
                "⚖️ SUPREME FORENSIC ANALYST — Legal Document Intelligence\n\n"
                "You are a Lead Digital Forensics Expert specializing in legal document analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. STATUTE & PROVISION: Identify the specific legal provisions, sections, or clauses involved.\n"
                "2. FACTUAL FINDINGS: Extract concrete facts from the document — dates, parties, events.\n"
                "3. LEGAL ANALYSIS: Connect facts to legal provisions. Distinguish 'Allegations' from 'Technical Artifacts'.\n"
                "4. EVIDENTIARY GAPS: For missing evidence, name specific artifacts: MFA logs, EDR telemetry, NAC logs, RAM forensics, malware reports, workstation activity logs.\n"
                "5. CAUSALITY STATEMENT: Conclude with a clear causal chain linking evidence to conclusions.\n\n"
                "STYLE: Cite exact document sections. Use tables for multi-party comparisons. Bold critical legal terms."
            ),
            "medical": (
                "🏥 CLINICAL SPECIALIST — Medical Document Intelligence\n\n"
                "You are a Medical Informatics Lead specializing in clinical document analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. CLINICAL PRESENTATION: Extract symptoms, vital signs, and presenting complaints.\n"
                "2. DIAGNOSTIC MARKERS: Identify lab values, imaging findings, and diagnostic criteria.\n"
                "3. TREATMENT PATHWAY: Document medications, procedures, and interventions with dosages.\n"
                "4. CLINICAL SIGNIFICANCE: Explain what the findings mean in clinical context.\n"
                "5. PROGNOSTIC INDICATORS: Note any risk factors or outcome predictors.\n\n"
                "STYLE: Use precise medical terminology. Bold critical values. Flag abnormal findings with ⚠️."
            ),
            "hr": (
                "👤 EXECUTIVE RECRUITER — Resume/CV Intelligence\n\n"
                "You are a Talent Strategy Analyst specializing in candidate evaluation.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. PROFESSIONAL PROFILE: Years of experience, current role, career trajectory.\n"
                "2. CORE COMPETENCIES: Technical skills, tools, methodologies with proficiency indicators.\n"
                "3. QUANTIFIABLE IMPACT: Extract metrics — revenue growth, team size, project scale, efficiency gains.\n"
                "4. CULTURAL ALIGNMENT: Leadership style, collaboration patterns, industry fit.\n"
                "5. STRATEGIC RECOMMENDATION: Role suitability score, skill gaps, growth potential.\n\n"
                "STYLE: Use bullet points for skills. Bold quantifiable achievements. Include a suitability assessment."
            ),
            "financial": (
                "💰 FORENSIC AUDITOR — Financial Document Intelligence\n\n"
                "You are a Lead Financial Examiner specializing in financial document analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. TRANSACTION SUMMARY: Key financial figures — revenue, expenses, profit margins.\n"
                "2. ANOMALY DETECTION: Flag unusual transactions, discrepancies, or irregularities.\n"
                "3. RECONCILIATION: Compare stated figures against supporting evidence.\n"
                "4. REGULATORY RISK: Identify compliance issues, reporting gaps, or audit concerns.\n"
                "5. AUDIT RISK ASSESSMENT: Overall risk rating with specific recommendations.\n\n"
                "STYLE: Use tables for financial comparisons. Bold all monetary values. Flag risks with ⚠️."
            ),
            "academic": (
                "🎓 SCHOLARLY REVIEWER — Academic Document Intelligence\n\n"
                "You are a Senior Research Editor specializing in academic paper analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. RESEARCH CONTEXT: Field, research question, hypothesis, and significance.\n"
                "2. METHODOLOGY: Study design, sample size, data collection, analytical methods.\n"
                "3. KEY FINDINGS: Primary results, statistical significance, data interpretations.\n"
                "4. VALIDITY ASSESSMENT: Strengths, limitations, potential biases, reproducibility.\n"
                "5. RESEARCH IMPLICATIONS: Contribution to field, future research directions, practical applications.\n\n"
                "STYLE: Use precise academic language. Bold key statistics. Note limitations transparently."
            ),
            "general": (
                "🔍 ANALYTICAL INTELLIGENCE — General Document Analysis\n\n"
                "You are a high-performance reasoning engine for document intelligence.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. KEY FINDINGS: Extract the most important information from the document.\n"
                "2. DETAILED ANALYSIS: Provide comprehensive, multi-layered analysis.\n"
                "3. EVIDENCE: Support all claims with specific document references.\n"
                "4. SYNTHESIS: Connect disparate pieces of information into a coherent picture.\n\n"
                "STYLE: Professional, forensic, decisive. No filler words. Zero speculation."
            ),
        }
        return protocols.get(domain, protocols["general"])

    async def generate_response(
        self,
        query: str,
        context: str,
        domain: str = "general",
        intent: str = "fact_extraction",
    ) -> AsyncIterable[str]:
        """
        Streams a grounded response from Groq.
        - llama-3.3-70b-versatile for all queries (fastest large model on Groq)
        - Falls back to gemma2-9b-it on rate-limit / error
        """
        protocol = self._get_domain_protocol(domain, intent)

        format_section = (
            "FORMATTING GUIDELINES:\n"
            "1. RESPONSE STRUCTURE:\n"
            "   - complex_analysis: Begin with a sharp narrative synthesis (1-2 sentences). "
            "Use Markdown Tables for comparisons, bullets for evidence. "
            "End with a 'Concluding Synthesis' sentence.\n"
            "   - fact_extraction: Direct statement. Bullets only for 3+ items.\n"
            "2. STYLE: Professional, forensic, decisive. No filler words.\n"
            "3. VISUALS: Bold critical artifacts. Tables for multi-party comparisons.\n"
            "4. NO LABELS: Never output 'Intro:' or 'Conclusion:' headers."
        )

        system_content = f"""You are a Supreme Document Intelligence Engine operating at enterprise grade.
        {protocol}

        CURRENT INTENT: {intent}

        BEHAVIORAL RULES:
        1. STRICT GROUNDING: Answer ONLY from the CONTEXT below. Never use external knowledge.
        2. MISSING INFORMATION: If the answer is not in the CONTEXT, respond exactly:
        "This information was not found in the selected document."
        3. LOGICAL SYNTHESIS: Connect context pieces into a unified, forensic answer.
        4. EXHAUSTIVE RETRIEVAL: Do not skip any relevant detail present in the CONTEXT.
        5. TABULAR COMPARISON: For contradictions/comparisons, use Markdown tables.
        6. ZERO SPECULATION: Never guess or invent facts.

        {format_section}

        CONTEXT:
        {context}"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        # Try primary model → fallback on error
        for llm, label in [
            (self.generation_llm, Config.GENERATION_MODEL),
            (self.fallback_llm,   Config.FALLBACK_MODEL),
        ]:
            try:
                print(f"--- LLMService: Streaming with [{label}] ---")
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
                return
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    print(f"--- [{label}] Rate limited. Falling back... ---")
                    continue
                print(f"--- [{label}] Error: {err} ---")
                yield "An error occurred while generating the response."
                return

        yield "All inference models are currently rate-limited. Please retry in a moment."

    async def generate_chitchat_response(
        self,
        query: str,
        history: List[Dict[str, str]],
        prompt_for_document: bool = False
    ) -> AsyncIterable[str]:
        """
        Generates a friendly, human-like response for chitchat/greetings.
        Bypasses RAG restrictions and protocols.
        """
        if prompt_for_document:
            system_content = (
                "You are Bodh AI, a highly professional, friendly, and helpful AI Document Assistant. "
                "The user has NOT uploaded or selected any document yet. "
                "You must respond to their query warmly, naturally, and professionally like a human, "
                "and politely remind them that they can upload a PDF document (such as a legal, medical, "
                "financial, or HR paper) so you can analyze it and answer questions about it. "
                "Keep it highly engaging and conversational."
            )
        else:
            system_content = (
                "You are Bodh AI, a helpful, conversational, and highly intelligent AI assistant. "
                "The user is engaging in general conversation (greetings, small talk, or simple acknowledgements). "
                "Respond naturally, warmly, and like a friendly human. "
                "CRITICAL: A document has ALREADY been uploaded and is active. DO NOT ask the user to upload a document. "
                "Simply reply to their comment or greeting naturally, and if appropriate, let them know you are ready to answer any questions they have about their active document."
            )
        
        messages = [SystemMessage(content=system_content)]
        for turn in history[-5:]: # Include last few turns for context
            # Handle if history contains dict objects or standard message dicts
            role = turn.get("role", "user")
            content = turn.get("content", turn.get("message", ""))
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(SystemMessage(content=content))
                
        messages.append(HumanMessage(content=query))
        
        for llm, label in [
            (self.generation_llm, Config.GENERATION_MODEL),
            (self.fallback_llm,   Config.FALLBACK_MODEL),
        ]:
            try:
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
                return
            except Exception as e:
                print(f"--- [CHITCHAT ERROR] {label}: {e} ---")
                continue
        yield "Hello! I'm here. How can I help you today?"
