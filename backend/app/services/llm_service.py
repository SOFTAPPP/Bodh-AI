from typing import List, Dict, Optional, AsyncIterable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import Config
import time

class LLMService:
    def __init__(self):
        # Unified Model Strategy (GPT-4o-mini)
        self.llm = ChatOpenAI(
            model_name=Config.REASONING_MODEL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY,
            streaming=True
        )
        
        # We use the same instance for both to save memory/resources
        self.reasoning_llm = self.llm
        self.generation_llm = self.llm

    async def analyze_query(self, query: str, history: List[Dict[str, str]]) -> Dict:
        """Optimized query analysis with ultra-low latency."""
        is_followup = len(history) > 0
        
        system_prompt = """You are a high-precision RAG Analysis Engine. Output JSON ONLY.
        {
        "standalone_query": "The rephrased query for standalone context",
        "domain": "legal|medical|hr|financial|academic|general",
        "intent": "fact_extraction|complex_analysis"
        }
        
        CRITICAL RULES:
        1. REPHRASE: If this is a followup, incorporate necessary context from history into the standalone_query.
        2. INTENT CLASSIFICATION:
           - 'fact_extraction': Simple lookups (What, Who, When, Where, List of X).
           - 'complex_analysis': Reasoning, Comparison, Evaluation, Impact, Why, How, Contradictions, or Judicial-style questions.
        3. DOMAIN: Detect the primary subject matter of the query."""

        history_context = ""
        if is_followup:
            last = history[-1]
            history_context = f"PREV: {last['content'][:80]}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"{history_context}Q: {query}")
        ])

        try:
            # Limit tokens for instant JSON generation
            response = await self.reasoning_llm.ainvoke(prompt, config={"max_tokens": 100})
            import json, re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"standalone_query": query, "domain": "general", "intent": "fact_extraction"}
        except Exception:
            return {"standalone_query": query, "domain": "general", "intent": "fact_extraction"}

    def _get_domain_protocol(self, domain: str) -> str:
        protocols = {
            "legal": "⚖️ SUPREME FORENSIC ANALYST: You are a Lead Digital Forensics Expert. Distinguish between 'Allegations' and 'Technical Artifacts'. When asked for missing evidence, avoid generic procedural reports; instead, mandate specific technical artifacts: Multi-Factor Authentication (MFA) logs, Endpoint Detection & Response (EDR) telemetry, Network Access Control (NAC) logs, RAM forensics, Malware investigation reports, and Employee Workstation activity logs. Always conclude with a 'Causality Statement' explaining how this evidence resolves the conflict between intent and exploit.",
            "medical": "🏥 CLINICAL SPECIALIST: You are a Medical Informatics Lead. Focus on diagnostic markers, lab values, and clinical pathways. Be extremely precise. Conclude with the 'Clinical Significance' of the findings.",
            "hr": "👤 EXECUTIVE RECRUITER: You are a Talent Strategy Analyst. Focus on skill gap analysis, cultural alignment, and quantifiable impact. Conclude with a 'Strategic Recommendation'.",
            "financial": "💰 FORENSIC AUDITOR: You are a Lead Financial Examiner. Focus on transaction anomalies, ledger reconciliation, and regulatory risk. Conclude with an 'Audit Risk Assessment'.",
            "academic": "🎓 SCHOLARLY REVIEWER: You are a Senior Research Editor. Focus on methodology, validity, and contribution to the field. Conclude with 'Research Implications'.",
            "general": "🔍 ANALYTICAL INTELLIGENCE: You are a high-performance reasoning engine. Provide deep, multi-layered analysis with zero fluff."
        }
        return protocols.get(domain, protocols["general"])

    async def generate_response(self, query: str, context: str, domain: str = "general", intent: str = "fact_extraction") -> AsyncIterable[str]:
        """Generates a hybrid response: Narrative Synthesis + Forensic Grounding."""
        protocol = self._get_domain_protocol(domain)
        
        format_section = (
            "FORMATTING GUIDELINES:\n"
            "1. RESPONSE STRUCTURE:\n"
            "   - IF intent='complex_analysis': Begin with a sharp narrative synthesis (1-2 sentences). Use Markdown Tables for comparisons or structured bullets for evidence. ALWAYS end with a final 'Concluding Synthesis' sentence that ties the analysis together.\n"
            "   - IF intent='fact_extraction': Provide a direct statement. Use bullets ONLY for lists of 3+ items.\n"
            "2. STYLE: Professional, forensic, and decisive. No filler.\n"
            "3. VISUALS: Use bold for critical artifacts. Use tables for multi-party comparisons.\n"
            "4. NO LABELS: Never use 'Intro:' or 'Conclusion:' labels. Start and end immediately."
        )

        system_prompt = f"""You are a Supreme Document Intelligence Engine.
        {protocol}
        
        CURRENT INTENT: {intent}
        
        BEHAVIORAL RULES:
        1. LOGICAL SYNTHESIS: Connect context pieces into a unified forensic answer.
        2. TABULAR COMPARISON: If the user asks for contradictions, differences, or a comparison between parties, use a Markdown Table with columns like 'Point of Conflict', 'Prosecution/Party A', and 'Defense/Party B'.
        3. EXHAUSTIVE RETRIEVAL: Do not miss any relevant details found in the text.
        4. STRICT GROUNDING: Stay 100% within the provided CONTEXT. If a fact (e.g., 'Genomic Data', 'Wearables') is not in the text, you MUST NOT include it.
        5. INTELLIGENT APPLICATION: You ARE allowed to synthesize and apply the facts found in the document to answer reasoning, hypothetical, or design questions. However, your logic must be built solely using the 'building blocks' (technologies, data, and outcomes) mentioned in the context.
        6. ZERO SPECULATION: Never guess or invent information that isn't supported by the context's logic.
        
        {format_section}

        CONTEXT:
        {{context}}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Build the chain
        chain = prompt | self.generation_llm
        
        try:
            async for chunk in chain.astream({"query": query, "context": context}):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"--- LLM Error: {str(e)} ---")
            yield "An error occurred while generating the response."
            raise e
