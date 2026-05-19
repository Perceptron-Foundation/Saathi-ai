"""
query.py — interactive console RAG query loop
================================================
  user query  →  BiomedBERT embed  →  Pinecone cosine top-K
              →  build prompt  →  Gemini LLM  →  answer + references

Run:
    python query.py

Optional:  filter results to a specific PDF source:
    python query.py
"""
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    METADATA_FILE,
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

def retrieve(query: str, embed_model: SentenceTransformer, index, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and fetch top-K similar vectors from Pinecone."""
    q_vec = embed_model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    response = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return response.get("matches", [])


# ── reference builder ─────────────────────────────────────────────────────────

def load_metadata_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": []}


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

def query_loop(embed_model: SentenceTransformer, index, llm):
    print("Type your question and press Enter \n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        # 1. Retrieve context
        matches = retrieve(query, embed_model, index)

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

    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )

    query_loop(embed_model, index, llm)

if __name__ == "__main__":
    main()