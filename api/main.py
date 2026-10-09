import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import get_collection
from api.routers import documents, ask, trends, files, oasis, qa
from api.schemas import HealthResponse

app = FastAPI(title="Local RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(trends.router)
app.include_router(files.router)
app.include_router(oasis.router)
app.include_router(qa.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    ollama_ok = False
    try:
        ollama.list()
        ollama_ok = True
    except Exception:
        pass

    chroma_ok = False
    try:
        get_collection()
        chroma_ok = True
    except Exception:
        pass

    return HealthResponse(ollama=ollama_ok, chroma=chroma_ok)
