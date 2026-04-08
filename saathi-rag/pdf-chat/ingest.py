"""
ingest.py — PDF ingestion pipeline
=====================================
  PDF  →  pypdf loader  →  LangChain splitter  →  BiomedBERT embeddings
       →  Pinecone upsert  +  chunks_metadata.json update

Run:
    python ingest.py --pdf path/to/file.pdf
    python ingest.py --pdf path/to/file.pdf --pdf path/to/other.pdf
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_METRIC,
    METADATA_FILE,
)

def load_metadata_store(path: str) -> dict:
    """Load existing metadata JSON, or return empty store."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": []}


def save_metadata_store(store: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    print(f"[metadata] saved → {path}  (total chunks: {len(store['chunks'])})")


def get_pinecone_index(pc: Pinecone):
    """Create index if it doesn't exist, then return the Index object."""
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric=SIMILARITY_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"[pinecone] index created.")
    else:
        print(f"[pinecone] using existing index '{PINECONE_INDEX}'.")
    return pc.Index(PINECONE_INDEX)


def ingest_pdf(pdf_path: str, embed_model: SentenceTransformer, index, store: dict) -> int:
    """
    Full ingestion for a single PDF.
    Returns the number of new chunks upserted.
    """
    pdf_path = str(Path(pdf_path).resolve())
    source_name = Path(pdf_path).name
    date_chunked = datetime.now(timezone.utc).isoformat()

    # ── 1. Load PDF ───────────────────────────────────────────────────────────
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,            # character-based approximation
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    print(f"[chunk] {len(chunks)} chunks created  (size≈{CHUNK_SIZE} tok, overlap≈{CHUNK_OVERLAP} tok).")

    if not chunks:
        print("[warn] no chunks produced — skipping.")
        return 0

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    texts = [c.page_content for c in chunks]
    print(f"[embed] encoding with {EMBEDDING_MODEL} …")
    vectors = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"[embed] done — shape {vectors.shape}.")

    # ── 4. Build records & upsert to Pinecone ─────────────────────────────────
    pinecone_records = []
    metadata_records = []

    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        chunk_id   = f"{source_name}__chunk_{i}__{uuid.uuid4().hex[:8]}"
        page_num   = chunk.metadata.get("page", "unknown")

        pinecone_records.append({
            "id"    : chunk_id,
            "values": vec.tolist(),
            "metadata": {
                "chunk_id"    : chunk_id,
                "source"      : source_name,
                "page"        : page_num,
                "date"        : date_chunked,
                "text"        : chunk.page_content[:1000],
            },
        })

        metadata_records.append({
            "chunk_id"    : chunk_id,
            "source"      : source_name,
            "page"        : page_num,
            "date"        : date_chunked,
            "char_count"  : len(chunk.page_content),
        })

    batch_size = 100
    for start in range(0, len(pinecone_records), batch_size):
        batch = pinecone_records[start : start + batch_size]
        index.upsert(vectors=batch)

    store["chunks"].extend(metadata_records)

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG pipeline.")
    parser.add_argument("--pdf", action="append", required=True, help="Path to a PDF file. Repeat for multiple files.")
    args = parser.parse_args()

    print(f"[init] loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = get_pinecone_index(pc)

    store = load_metadata_store(METADATA_FILE)

    total_chunks = 0
    for pdf_path in args.pdf:
        if not os.path.isfile(pdf_path):
            print(f"[warn] file not found: {pdf_path} — skipping.")
            continue
        n = ingest_pdf(pdf_path, embed_model, index, store)
        total_chunks += n

    save_metadata_store(store, METADATA_FILE)
    print(f"\n ingestion complete — {total_chunks} chunks upserted across {len(args.pdf)} file(s).")


if __name__ == "__main__":
    main()