import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.whatsapp_routes import router as wa_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # warmup embeddings
    from app.api.routes import bot
    asyncio.create_task(asyncio.to_thread(bot.vector_service.warmup))
    yield

app = FastAPI(
    title="Bodh AI — Groq Ultra-Low-Latency RAG Engine",
    description="Enterprise-grade PDF intelligence powered by Groq + LLaMA 3.3 70B + FAISS",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Active-File"],
)

app.include_router(router)
app.include_router(wa_router, prefix="/whatsapp", tags=["whatsapp"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "engine": "Groq LLaMA-3.3-70B + FAISS RAG",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "inference": "Groq API — ultra-low latency",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
