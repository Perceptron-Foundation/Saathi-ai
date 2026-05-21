import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
	PINECONE_API_KEY,
	PINECONE_INDEX,
	GOOGLE_API_KEY,
	HF_TOKEN,
	GEMINI_MODEL,
	TOP_K,
	LLM_TEMPERATURE,
	LLM_MAX_TOKENS,
)
from constants import SYSTEM_PROMPT
from query import retrieve, build_prompt, build_references


logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
	question: str = Field(..., min_length=1, description="User query for PDF RAG")
	top_k: int = Field(default=TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
	success: bool
	answer: str
	citations: list[str]
	safety_notice: str | None
	insufficient_context: bool
	references: list[dict[str, Any]]

class HealthResponse(BaseModel):
    status: str

app = FastAPI(title="Saathi RAG API", version="1.0.0")

@app.on_event("startup")
def startup_event() -> None:
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY")
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN")

    app.state.embed_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    app.state.index = pc.Index(PINECONE_INDEX)

    app.state.llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/query", response_model=QueryResponse)
def query_loop(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        matches = retrieve(
            question,
            app.state.embed_client,
            app.state.index,
            top_k=request.top_k,
        )

        prompt = build_prompt(question, matches)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        llm_response = app.state.llm.invoke(messages)
        raw_content = llm_response.content.strip()

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON content; using fallback envelope")
            parsed = {
                "success": True,
                "answer": raw_content,
                "citations": [],
                "safety_notice": None,
                "insufficient_context": False,
            }

        refs = build_references(matches)

        return QueryResponse(
            success=bool(parsed.get("success", True)),
            answer=str(parsed.get("answer", "")),
            citations=list(parsed.get("citations", [])),
            safety_notice=parsed.get("safety_notice"),
            insufficient_context=bool(parsed.get("insufficient_context", False)),
            references=refs,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query pipeline failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc