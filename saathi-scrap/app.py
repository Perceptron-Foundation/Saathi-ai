import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pinecone import Pinecone
from pydantic import BaseModel

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI()
load_dotenv()

# =====================================================
# ENABLE CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIG
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "t1d-rag")
TOP_K = int(os.getenv("TOP_K", "5"))

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1"),
    )
    if GEMINI_API_KEY
    else None
)

pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(PINECONE_INDEX) if pc else None


def _is_model_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "not found" in msg and "model" in msg


def _candidate_embedding_models() -> list[str]:
    candidates = [
        EMBEDDING_MODEL,
        "gemini-embedding-2",
        "gemini-embedding-001",
        "text-embedding-004",
        "embedding-001",
    ]
    unique: list[str] = []
    seen = set()
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)
    return unique

# =====================================================
# REQUEST MODEL
# =====================================================
class QueryRequest(BaseModel):
    query: str
    user_id: str
    glucose: Optional[float] = None
    iob: Optional[float] = None

# =====================================================
# HOME ROUTE
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Saathi AI backend running successfully"
    }

# =====================================================
# FETCH LAST 5 MEALS
# =====================================================
def fetch_last_5_meals(user_id):

    # =================================================
    # TODO:
    # Replace this mock data with actual API call
    # =================================================

    meals = [
        {
            "meal_name": "Rice",
            "carbs": 60,
            "time": "2 PM"
        },
        {
            "meal_name": "Banana",
            "carbs": 20,
            "time": "11 AM"
        },
        {
            "meal_name": "Roti Sabzi",
            "carbs": 45,
            "time": "9 AM"
        },
        {
            "meal_name": "Milk",
            "carbs": 15,
            "time": "7 AM"
        },
        {
            "meal_name": "Apple",
            "carbs": 25,
            "time": "Yesterday 9 PM"
        }
    ]

    return meals

# =====================================================
# SEARCH FUNCTION
# =====================================================
def search(query, k=5):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    if not index:
        raise RuntimeError("PINECONE_API_KEY or PINECONE_INDEX is not configured")

    query_vec = None
    last_exc = None
    for model_name in _candidate_embedding_models():
        try:
            emb_resp = gemini_client.models.embed_content(
                model=model_name,
                contents=[query],
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            query_vec = emb_resp.embeddings[0].values
            break
        except Exception as exc:
            if _is_model_not_found_error(exc):
                last_exc = exc
                continue
            raise

    if query_vec is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("No embedding model candidates available")

    results = index.query(
        vector=query_vec,
        top_k=k,
        include_metadata=True,
    )
    return results.get("matches", [])

# =====================================================
# GENERATE ANSWER
# =====================================================
def generate_answer(query, patient_data):
    matches = search(query, TOP_K)
    docs = [m.get("metadata", {}).get("text", "") for m in matches]
    docs = [d for d in docs if d]
    context = "\n\n".join(docs)

    prompt = f"""
You are a medically safe Type 1 Diabetes assistant.

STRICT RULES:
- Answer ONLY from provided context
- Do NOT hallucinate
- If unsure say:
  "I don't have enough information"
- Do NOT give unsafe insulin advice
- Keep answers concise and safe

Retrieved Medical Context:
{context}

Patient Data:
{patient_data}

User Question:
{query}

Answer:
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return {
        "answer": response.text or "",
        "sources": [
            {
                "source": m.get("metadata", {}).get("source", "unknown"),
                "chunk_id": m.get("id", ""),
                "score": round(float(m.get("score", 0.0)), 4),
            }
            for m in matches
        ],
    }

# =====================================================
# ASK ROUTE
# =====================================================
@app.post("/ask")
def ask(req: QueryRequest):

    meals = fetch_last_5_meals(
        req.user_id
    )

    patient_data = {
        "glucose": req.glucose,
        "iob": req.iob,
        "last_5_meals": meals
    }

    result = generate_answer(
        req.query,
        patient_data
    )

    return result
