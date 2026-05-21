import re
import json
import time
from typing import List, Dict, AsyncIterable, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import SecretStr
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
        groq_api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
        
        self.generation_llm = ChatGroq(
            model=Config.GENERATION_MODEL,
            api_key=groq_api_key,
            temperature=0.0,
            streaming=True
        )
        self.reasoning_llm  = ChatGroq(
            model=Config.REASONING_MODEL,
            api_key=groq_api_key,
            temperature=0.0,
            streaming=True
        )
        self.fallback_llm   = ChatGroq(
            model=Config.FALLBACK_MODEL,
            api_key=groq_api_key,
            temperature=0.0,
            streaming=True
        )

        # setup llm
        self.hyde_llm = ChatGroq(
            model=Config.HYDE_MODEL,
            api_key=groq_api_key,
            temperature=0.0,
            streaming=False
        )

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
                max_tokens=150,
            )
            hyde_text = str(resp.content).strip()
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
        # quick classify
        if not history:
            return self._heuristic_classify(query)

        system_prompt = (
            "You are a RAG query classifier. Output ONLY valid JSON, no markdown fences:\n"
            '{"standalone_query": "...", "domain": "legal|medical|hr|financial|academic|general", '
            '"intent": "fact_extraction|complex_analysis"}\n\n'
            "RULES:\n"
            "1. standalone_query: MUST be a self-contained search query. Replace pronouns (it, this, he) with the subject from PREV. If Q asks for a general summary (e.g., 'What is this about?'), output 'Summarize the document'. NEVER output meta-commands like 'context reset'.\n"
            "2. intent: fact_extraction = simple lookup; complex_analysis = reasoning/comparison/why/how.\n"
            "3. domain: primary subject of the query."
        )

        last = history[-1]
        human_msg = f"PREV: {last['content'][:100]}\nQ: {query}"

        try:
            # query analyzer
            resp = await self.fallback_llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)],
                max_tokens=120,
            )
            match = re.search(r"\{.*\}", str(resp.content), re.DOTALL)
            if match:
                res = json.loads(match.group())
                sq = res.get("standalone_query", "").lower().strip()
                if not sq or sq in ["context reset", "reset", "none", "n/a", "null"]:
                    res["standalone_query"] = query
                return res
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
        If intent is fact_extraction, returns a clean, direct lookup prompt to prevent rigid formatting.
        If intent is complex_analysis, returns the structured forensic analysis protocol.
        """
        if intent == "fact_extraction":
            return (
                "You are an expert document assistant. Your task is to answer the user's question "
                "directly, naturally, and concisely based ONLY on the provided context.\n"
                "Do NOT use any rigid structured headers, points, or templates. Answer like a friendly, helpful expert."
            )

        protocols = {
            "legal": (
                "SUPREME FORENSIC ANALYST — Legal Document Intelligence\n\n"
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
                "CLINICAL SPECIALIST — Medical Document Intelligence\n\n"
                "You are a Medical Informatics Lead specializing in clinical document analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. CLINICAL PRESENTATION: Extract symptoms, vital signs, and presenting complaints.\n"
                "2. DIAGNOSTIC MARKERS: Identify lab values, imaging findings, and diagnostic criteria.\n"
                "3. TREATMENT PATHWAY: Document medications, procedures, and interventions with dosages.\n"
                "4. CLINICAL SIGNIFICANCE: Explain what the findings mean in clinical context.\n"
                "5. PROGNOSTIC INDICATORS: Note any risk factors or outcome predictors.\n\n"
                "STYLE: Use precise medical terminology. Bold critical values. Flag abnormal findings."
            ),
            "hr": (
                "EXECUTIVE RECRUITER — Resume/CV Intelligence\n\n"
                "You are a Talent Strategy Analyst specializing in candidate evaluation.\n\n"
                "ASSUMPTION RULE: You must assume that all experiences, projects, skills, certifications, and achievements "
                "listed in the resume belong to the candidate (whose name is at the top of the resume), unless explicitly stated otherwise.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. PROFESSIONAL PROFILE: Years of experience, current role, career trajectory.\n"
                "2. CORE COMPETENCIES: Technical skills, tools, methodologies with proficiency indicators.\n"
                "3. QUANTIFIABLE IMPACT: Extract metrics — revenue growth, team size, project scale, efficiency gains.\n"
                "4. CULTURAL ALIGNMENT: Leadership style, collaboration patterns, industry fit.\n"
                "5. STRATEGIC RECOMMENDATION: Role suitability score, skill gaps, growth potential.\n\n"
                "STYLE: Use bullet points for skills. Bold quantifiable achievements. Include a suitability assessment."
            ),
            "financial": (
                "FORENSIC AUDITOR — Financial Document Intelligence\n\n"
                "You are a Lead Financial Examiner specializing in financial document analysis.\n\n"
                "RESPONSE STRUCTURE:\n"
                "1. TRANSACTION SUMMARY: Key financial figures — revenue, expenses, profit margins.\n"
                "2. ANOMALY DETECTION: Flag unusual transactions, discrepancies, or irregularities.\n"
                "3. RECONCILIATION: Compare stated figures against supporting evidence.\n"
                "4. REGULATORY RISK: Identify compliance issues, reporting gaps, or audit concerns.\n"
                "5. AUDIT RISK ASSESSMENT: Overall risk rating with specific recommendations.\n\n"
                "STYLE: Use tables for financial comparisons. Bold all monetary values. Flag risks."
            ),
            "academic": (
                "SCHOLARLY REVIEWER — Academic Document Intelligence\n\n"
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
                "ANALYTICAL INTELLIGENCE — General Document Analysis\n\n"
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

        # run model
        for llm, label in [
            (self.generation_llm, Config.GENERATION_MODEL),
            (self.fallback_llm,   Config.FALLBACK_MODEL),
        ]:
            try:
                print(f"--- LLMService: Streaming with [{label}] ---")
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            yield chunk.content
                        else:
                            yield str(chunk.content)
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
        
        messages: List[BaseMessage] = [SystemMessage(content=system_content)]
        for turn in history[-5:]: # trim history
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
                        if isinstance(chunk.content, str):
                            yield chunk.content
                        else:
                            yield str(chunk.content)
                return
            except Exception as e:
                print(f"--- [CHITCHAT ERROR] {label}: {e} ---")
                continue
        yield "Hello! I'm here. How can I help you today?"

    async def generate_document_metadata(self, filename: str, text_sample: str) -> dict:
        """
        Generates a short summary and keywords for the document text sample.
        """
        prompt = (
            "You are an AI document classifier. Analyze the following excerpt from a document "
            f"named '{filename}' and extract its main topics/keywords and a brief 2-sentence summary.\n\n"
            "Output ONLY a valid JSON object with the following structure (no markdown formatting, no fences):\n"
            '{\n  "summary": "A 2-sentence summary of the document contents and purpose.",\n  "topics": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]\n}\n\n'
            f"Document Excerpt:\n{text_sample[:4000]}"
        )
        try:
            resp = await self.fallback_llm.ainvoke([HumanMessage(content=prompt)], max_tokens=200)
            content = str(resp.content).strip()
            # clean output
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"--- Error generating document metadata: {e} ---")
        
        # handle fallback
        return {
            "summary": f"This document contains information related to {filename}.",
            "topics": [filename.replace(".pdf", ""), "document"]
        }

