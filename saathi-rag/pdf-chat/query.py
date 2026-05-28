"""
query.py — interactive console RAG query loop
================================================
    user query  →  HF inference embed  →  Pinecone cosine top-K
              →  build prompt  →  Gemini LLM  →  answer + references

Run:
    python query.py

Optional:  filter results to a specific PDF source:
    python query.py
"""
import json
import os
import math

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
    EMBEDDING_MODEL,
    TOP_K,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

from constants import SYSTEM_PROMPT

def build_prompt(query: str, context_passages: list[dict]) -> str:
    """Construct the final user message with numbered context blocks."""
    blocks = []
    for i, p in enumerate(context_passages, start=1):
        source = p["metadata"].get("source", "unknown")
        page   = p["metadata"].get("page", "?")
        text   = p["metadata"].get("text", "")
        blocks.append(f"[{i}] (source: {source}, page: {page})\n{text}")

    context_str = "\n\n".join(blocks)
    return (
        f"Context passages:\n\n{context_str}\n\n"
        f"---\nQuestion: {query}\n\nAnswer:"
    )


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, embed_client: InferenceClient, index, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and fetch top-K similar vectors from Pinecone."""
    raw_embedding = embed_client.feature_extraction(
        text=query,
        model=EMBEDDING_MODEL,
    )

    # Normalize to match cosine retrieval behavior used during ingestion.
    q_vec = [float(v) for v in raw_embedding]
    norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
    q_vec = [v / norm for v in q_vec]

    response = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return response.get("matches", [])


# ── reference builder ─────────────────────────────────────────────────────────

def build_references(matches: list[dict]) -> list[dict]:
    """Return a clean reference list from retrieved matches."""
    refs = []
    for m in matches:
        meta = m.get("metadata", {})
        refs.append({
            "chunk_id"    : meta.get("chunk_id", m["id"]),
            "source"      : meta.get("source", "unknown"),
            "page"        : meta.get("page", "?"),
            "date"        : meta.get("date", "?"),
            "score"       : round(m.get("score", 0.0), 4),
        })
    return refs


def print_references(refs: list[dict]) -> None:
    print("\n References:")
    for i, r in enumerate(refs, start=1):
        print(f"  [{i}] {r['source']}  |  page {r['page']}  |  similarity {r['score']}  |  chunk: {r['chunk_id']}")


# ── query loop ────────────────────────────────────────────────────────────────

def query_loop(embed_client: InferenceClient, index, llm):
    print("Type your question and press Enter \n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        # 1. Retrieve context
        matches = retrieve(query, embed_client, index)

        # 2. Build prompt
        prompt = build_prompt(query, matches)

        # 3. Call Gemini
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer   = response.content.strip()

        # 4. Display
        print(answer)

        # 5. References
        refs = build_references(matches)
        print("\n" + "-"*50)
        print_references(refs)
        print()


def main():
    print(f"[init] QUERY MODE")

    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN")

    embed_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )

    query_loop(embed_client, index, llm)

if __name__ == "__main__":
    main()