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

    async def contextualize_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """Rephrases follow-up questions for RAG. Optimized for long-chat latency."""
        if not history or len(query.split()) < 3:
            return query
            
        # Aggressive Pruning: Only take the last 2 exchanges to minimize latency in long chats
        trimmed_history = []
        for m in history[-2:]:
            content = m['content'][:200] + "..." if len(m['content']) > 200 else m['content']
            trimmed_history.append(f"{m['role'].capitalize()}: {content}")
        
        history_str = "\n".join(trimmed_history)
        
        system_prompt = (
            "Given the conversation history and a follow-up question, rephrase it into a standalone question. "
            "Be extremely concise. Output ONLY the question."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"HISTORY:\n{history_str}\n\nFOLLOW-UP: {query}")
        ])
        
        try:
            response = await self.reasoning_llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            return query

    async def detect_intent_and_entities(self, query: str) -> Dict[str, Optional[str]]:
        """Identifies names, professional entities, or specific files mentioned and classifies the domain."""
        system_prompt = (
            "You are an expert intent classifier. Analyze the query and extract entities and domain in JSON format.\n"
            "DOMAINS:\n"
            "- 'legal': Contracts, case law, regulations, statutes.\n"
            "- 'medical': Clinical records, lab reports, medical journals.\n"
            "- 'hr': Resumes, job descriptions, employee handbooks.\n"
            "- 'general': Everything else.\n\n"
            "EXTRACT:\n"
            "{'person_name': str|null, 'topic': str|null, 'domain': 'legal'|'medical'|'hr'|'general'}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", query)
        ])
        
        try:
            response = await self.reasoning_llm.ainvoke(prompt)
            import json
            import re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Default to general if not specified
                if 'domain' not in data:
                    data['domain'] = 'general'
                return data
            return {"person_name": None, "topic": None, "domain": "general"}
        except Exception:
            return {"person_name": None, "topic": None, "domain": "general"}

    async def generate_response(self, query: str, context: str, domain: str = "general") -> AsyncIterable[str]:
        """Generates the final answer using GPT-4o-mini with Strict RAG enforcement."""
        
        # Best RAG Prompt (Production Ready) + Strict Rules
        system_prompt = """You are a Retrieval-Augmented Generation (RAG) system.

RULES:
- Use ONLY provided context
- Do NOT use external knowledge
- Do NOT hallucinate or expand
- If answer is not in context, respond ONLY with: "Not mentioned in the provided context."
- Keep answers concise and factual
- "Only use phrases present in retrieved context. Do NOT expand or explain beyond source."
- Strict mode: no external knowledge, no inference expansion, no extra bullet points.
- "Each answer must be grounded in one or more exact retrieved chunks. Do not merge unrelated chunks unless explicitly relevant."
- "Do not reuse identical evidence text across multiple answers unless necessary."

OUTPUT FORMAT:
Answer:
<final answer>

Evidence:
<exact supporting text from context>

CONTEXT:
{context}"""
        
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
