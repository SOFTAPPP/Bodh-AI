import re
import json
import time
from typing import List, Dict, AsyncIterable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import Config


class LLMService:
    """
    Groq-powered LLM service with:
    - llama-3.3-70b-versatile  → primary generation (ultra-low latency)
    - deepseek-r1-distill-llama-70b → complex reasoning
    - gemma2-9b-it             → rate-limit fallback
    - Token streaming for immediate output
    - Inline query analysis (zero extra LLM call for simple queries)
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

        print(f"--- LLMService: Groq engine ready [{Config.GENERATION_MODEL}] ---")

    # ─── Query Analysis ──────────────────────────────────────────────────────

    async def analyze_query(self, query: str, history: List[Dict[str, str]]) -> Dict:
        """
        Ultra-fast query analysis using Groq (gemma2-9b-it for minimal latency).
        Returns structured intent + rephrased standalone query + domain.

        Skips the LLM entirely for single-turn queries — uses heuristics instead.
        """
        # ── Fast-path: no history → classify via heuristics (0ms overhead) ──
        if not history:
            return self._heuristic_classify(query)

        # ── Follow-up: use lightweight Groq call to rephrase + classify ──────
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
        # Intent detection
        complex_kw = {"why", "how", "compare", "contrast", "analyze", "evaluate",
                      "difference", "impact", "implication", "relationship", "contradict"}
        intent = "complex_analysis" if any(kw in q for kw in complex_kw) else "fact_extraction"

        # Domain detection
        domain = "general"
        if any(kw in q for kw in ["court", "plaintiff", "defendant", "judgment", "legal", "law", "clause", "section"]):
            domain = "legal"
        elif any(kw in q for kw in ["patient", "diagnosis", "treatment", "clinical", "medical", "hospital", "drug"]):
            domain = "medical"
        elif any(kw in q for kw in ["resume", "cv", "hire", "skills", "candidate", "experience", "job"]):
            domain = "hr"
        elif any(kw in q for kw in ["revenue", "invoice", "tax", "profit", "financial", "audit", "balance sheet"]):
            domain = "financial"
        elif any(kw in q for kw in ["research", "methodology", "abstract", "hypothesis", "study", "findings"]):
            domain = "academic"

        return {"standalone_query": query, "domain": domain, "intent": intent}

    # ─── Domain Protocol ─────────────────────────────────────────────────────

    def _get_domain_protocol(self, domain: str) -> str:
        protocols = {
            "legal": (
                "⚖️ SUPREME FORENSIC ANALYST: You are a Lead Digital Forensics Expert. "
                "Distinguish 'Allegations' from 'Technical Artifacts'. For missing evidence, "
                "name specific artifacts: MFA logs, EDR telemetry, NAC logs, RAM forensics, "
                "malware reports, workstation activity logs. Conclude with a 'Causality Statement'."
            ),
            "medical": (
                "🏥 CLINICAL SPECIALIST: You are a Medical Informatics Lead. Focus on diagnostic "
                "markers, lab values, clinical pathways. Conclude with 'Clinical Significance'."
            ),
            "hr": (
                "👤 EXECUTIVE RECRUITER: You are a Talent Strategy Analyst. Focus on skill gaps, "
                "cultural alignment, quantifiable impact. Conclude with 'Strategic Recommendation'."
            ),
            "financial": (
                "💰 FORENSIC AUDITOR: You are a Lead Financial Examiner. Focus on transaction "
                "anomalies, ledger reconciliation, regulatory risk. Conclude with 'Audit Risk Assessment'."
            ),
            "academic": (
                "🎓 SCHOLARLY REVIEWER: You are a Senior Research Editor. Focus on methodology, "
                "validity, contribution to field. Conclude with 'Research Implications'."
            ),
            "general": (
                "🔍 ANALYTICAL INTELLIGENCE: You are a high-performance reasoning engine. "
                "Provide deep, multi-layered analysis with zero fluff."
            ),
        }
        return protocols.get(domain, protocols["general"])

    # ─── Response Generation ─────────────────────────────────────────────────

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
        protocol = self._get_domain_protocol(domain)

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
   "I don't know based on the provided context."
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
                return  # success — exit generator
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    print(f"--- [{label}] Rate limited. Falling back... ---")
                    continue
                print(f"--- [{label}] Error: {err} ---")
                yield "An error occurred while generating the response."
                return

        # All models exhausted
        yield "All inference models are currently rate-limited. Please retry in a moment."
