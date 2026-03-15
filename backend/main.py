"""
BabyLLM — A community-built, self-learning knowledge agent.
Production-ready FastAPI backend with FAISS vector memory and Groq LLM.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DB_PATH = os.getenv("DB_PATH", "/data/baby_faiss")
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("babyllm")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
embedding = None
db = None
metadata_store: dict = {}  # doc_id -> metadata
META_PATH = os.path.join(os.path.dirname(DB_PATH), "metadata.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata():
    global metadata_store
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata_store = json.load(f)

def _save_metadata():
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(metadata_store, f, indent=2)

def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _init_db():
    global db, embedding
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
        logger.info(f"Loaded existing FAISS index from {DB_PATH}")
    else:
        seed = Document(
            page_content="BabyLLM is a self-learning knowledge agent that grows from community interactions.",
            metadata={"source": "system", "created_at": datetime.now(timezone.utc).isoformat(), "type": "seed"}
        )
        db = FAISS.from_documents([seed], embedding)
        _save_db()
        logger.info("Created new FAISS index with seed document")
    _load_metadata()

def _save_db():
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    db.save_local(DB_PATH)

def _build_system_prompt(memory: str, memory_count: int) -> str:
    return f"""You are BabyLLM — a community-built, self-learning AI assistant.

CORE RULES:
1. You ONLY know what the community has taught you through your memory.
2. If the memory below does NOT contain information relevant to the question, say:
   "I haven't been taught about that yet. You can teach me by switching to Teach mode!"
3. NEVER fabricate, hallucinate, or use general knowledge outside your memory.
4. Be honest about what you know and don't know.
5. When answering, cite which memory/fact you're drawing from.
6. Be friendly, concise, and helpful.

YOUR MEMORY ({memory_count} relevant facts retrieved):
---
{memory}
---

Remember: If none of the memories above are relevant to the question, admit you don't know."""


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BabyLLM",
    description="A community-built, self-learning knowledge agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TeachRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000, description="Fact or knowledge to teach")
    source: str = Field(default="user", description="Source/contributor name")
    category: str = Field(default="general", description="Category tag")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)

class FeedbackRequest(BaseModel):
    question: str
    original_answer: str
    correct_answer: Optional[str] = None
    rating: str = Field(default="neutral", pattern="^(good|bad|correct)$")

class BulkTeachRequest(BaseModel):
    facts: list[str]
    source: str = "bulk_upload"
    category: str = "general"

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/teach")
async def teach(req: TeachRequest):
    """Teach BabyLLM a new fact."""
    doc_id = _doc_id(req.text)
    doc = Document(
        page_content=req.text,
        metadata={
            "id": doc_id,
            "source": req.source,
            "category": req.category,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "taught",
        }
    )
    db.add_documents([doc])
    _save_db()

    metadata_store[doc_id] = {
        "text_preview": req.text[:100],
        "source": req.source,
        "category": req.category,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_metadata()

    logger.info(f"Learned new fact: {req.text[:80]}...")
    return {"status": "learned", "id": doc_id, "preview": req.text[:100]}


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Ask BabyLLM a question."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured. Set it in .env file.")

    # Retrieve relevant memories with scores
    docs_and_scores = db.similarity_search_with_score(req.question, k=req.top_k)

    # Filter by similarity threshold (lower score = more similar in FAISS L2)
    relevant_docs = [(doc, score) for doc, score in docs_and_scores if score < (1.0 / SIMILARITY_THRESHOLD)]

    if not relevant_docs:
        memory_text = "(No relevant memories found)"
        memory_count = 0
    else:
        memory_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "unknown")
            memory_parts.append(f"[Memory #{i} | source: {source} | relevance: {1/(1+score):.0%}]\n{doc.page_content}")
        memory_text = "\n\n".join(memory_parts)
        memory_count = len(relevant_docs)

    system_prompt = _build_system_prompt(memory_text, memory_count)

    # Call Groq
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": req.question},
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq API error: {e.response.text}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=502, detail="Failed to reach LLM API")

    answer = data["choices"][0]["message"]["content"]
    sources_used = [doc.metadata.get("source", "unknown") for doc, _ in relevant_docs]

    return {
        "answer": answer,
        "memories_used": memory_count,
        "sources": sources_used,
    }


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """Provide feedback or correction on an answer."""
    if req.rating == "correct" and req.correct_answer:
        correction_text = f"Q: {req.question}\nA: {req.correct_answer}"
        doc = Document(
            page_content=correction_text,
            metadata={
                "id": _doc_id(correction_text),
                "source": "community_correction",
                "category": "correction",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "correction",
                "original_answer": req.original_answer[:200],
            }
        )
        db.add_documents([doc])
        _save_db()
        return {"status": "corrected", "message": "Memory updated with correction. Thank you!"}
    elif req.rating == "good":
        # Reinforce — store the Q&A pair as verified knowledge
        verified_text = f"Q: {req.question}\nA: {req.original_answer}"
        doc = Document(
            page_content=verified_text,
            metadata={
                "id": _doc_id(verified_text),
                "source": "community_verified",
                "category": "verified",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "verified",
            }
        )
        db.add_documents([doc])
        _save_db()
        return {"status": "reinforced", "message": "Good answer reinforced in memory!"}
    else:
        return {"status": "noted", "message": "Feedback noted. Consider providing a correction!"}


@app.post("/api/bulk-teach")
async def bulk_teach(req: BulkTeachRequest):
    """Teach multiple facts at once."""
    taught = 0
    for fact in req.facts:
        fact = fact.strip()
        if len(fact) < 3:
            continue
        doc = Document(
            page_content=fact,
            metadata={
                "id": _doc_id(fact),
                "source": req.source,
                "category": req.category,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "taught",
            }
        )
        db.add_documents([doc])
        taught += 1
    _save_db()
    return {"status": "bulk_learned", "count": taught}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a text file to teach BabyLLM."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    # Split into chunks by paragraphs or double newlines
    chunks = [c.strip() for c in text.split("\n\n") if c.strip() and len(c.strip()) > 10]

    if not chunks:
        # Fallback: split by single newlines
        chunks = [c.strip() for c in text.split("\n") if c.strip() and len(c.strip()) > 10]

    if not chunks:
        raise HTTPException(status_code=400, detail="No teachable content found in file")

    taught = 0
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "id": _doc_id(chunk),
                "source": f"file:{file.filename}",
                "category": "uploaded",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "taught",
            }
        )
        db.add_documents([doc])
        taught += 1

    _save_db()
    return {"status": "file_learned", "filename": file.filename, "chunks_taught": taught}


@app.get("/api/stats")
async def stats():
    """Get memory statistics."""
    total_docs = db.index.ntotal if db else 0
    categories = {}
    for meta in metadata_store.values():
        cat = meta.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_memories": total_docs,
        "tracked_entries": len(metadata_store),
        "categories": categories,
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/api/memories")
async def list_memories(limit: int = 50, offset: int = 0):
    """List recent memories."""
    items = sorted(metadata_store.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)
    page = items[offset:offset + limit]
    return {
        "memories": [{"id": k, **v} for k, v in page],
        "total": len(metadata_store),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "db_loaded": db is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "templates", "index.html"))
