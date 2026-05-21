from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from dotenv import load_dotenv
import os
load_dotenv()

import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI()

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
# GEMINI CONFIG
# =====================================================
# Replace with your actual Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash-lite"
)

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================
print("Loading embedding model...")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")

# =====================================================
# LOAD CHROMA DB
# =====================================================
print("Loading Chroma DB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="t1d_rag"
)

print("Chroma DB loaded successfully")

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
    #
    # Example:
    #
    # import requests
    #
    # response = requests.get(
    #     f"http://your-api.com/meals/{user_id}"
    # )
    #
    # meals = response.json()
    #
    # return meals
    # =================================================

    # MOCK DATA FOR NOW
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

    query_embedding = embed_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results

# =====================================================
# GENERATE ANSWER
# =====================================================
def generate_answer(query, patient_data):

    results = search(query)

    docs = results["documents"][0]

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
- Mention when medical supervision is needed

Retrieved Medical Context:
{context}

Patient Data:
{patient_data}

User Question:
{query}

Answer:
"""

    response = gemini_model.generate_content(
        prompt
    )

    return {
        "answer": response.text,
        "sources": results["metadatas"][0]
    }

# =====================================================
# ASK ROUTE
# =====================================================
@app.post("/ask")
def ask(req: QueryRequest):

    # ================================================
    # FETCH LAST 5 MEALS
    # ================================================
    meals = fetch_last_5_meals(
        req.user_id
    )

    # ================================================
    # BUILD PATIENT DATA
    # ================================================
    patient_data = {
        "glucose": req.glucose,
        "iob": req.iob,
        "last_5_meals": meals
    }

    # ================================================
    # GENERATE AI RESPONSE
    # ================================================
    result = generate_answer(
        req.query,
        patient_data
    )

    return result