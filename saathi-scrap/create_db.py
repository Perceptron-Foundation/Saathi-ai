import json
import os
import time
from typing import Iterable

import gdown
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATASET_FILE_ID = "1xBi6wggC2qWJBFzkrKgiTisqvJK44pY6"
DATASET_PATH = "./data/t1d_big_dataset.jsonl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "t1d-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "3"))
RETRY_MAX_SECONDS = float(os.getenv("RETRY_MAX_SECONDS", "120"))
PROGRESS_PATH = os.getenv("PROGRESS_PATH", "./data/ingest_progress.json")


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_dataset() -> None:
    os.makedirs("./data", exist_ok=True)
    if os.path.exists(DATASET_PATH):
        return
    print("Downloading dataset from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={DATASET_FILE_ID}",
        DATASET_PATH,
        quiet=False,
    )


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _is_model_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "not found" in msg and "model" in msg


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "quota" in msg
    )


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


def load_progress() -> dict:
    if not os.path.exists(PROGRESS_PATH):
        return {"next_index": 0, "upserted": 0, "model": None}
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(next_index: int, upserted: int, model: str | None) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"next_index": next_index, "upserted": upserted, "model": model},
            f,
            indent=2,
        )


def clear_progress() -> None:
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


def embed_with_fallback(gemini_client: genai.Client, texts: list[str]) -> tuple[str, list[list[float]]]:
    last_exc: Exception | None = None

    # Pass explicit content objects so each input text maps to one embedding.
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )
        for text in texts
    ]

    for model_name in _candidate_embedding_models():
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                emb = gemini_client.models.embed_content(
                    model=model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                return model_name, [item.values for item in emb.embeddings]
            except Exception as exc:
                if _is_model_not_found_error(exc):
                    print(f"Embedding model unavailable: {model_name} (trying next)")
                    last_exc = exc
                    break

                if _is_quota_error(exc) and attempt < MAX_RETRIES:
                    sleep_seconds = min(
                        RETRY_MAX_SECONDS,
                        RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                    )
                    print(
                        f"Quota/rate limit hit on {model_name}. "
                        f"Retrying in {sleep_seconds:.1f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})..."
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError("No embedding model candidates configured")


def ensure_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"Created Pinecone index: {PINECONE_INDEX}")
    return pc.Index(PINECONE_INDEX)


def main() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1"),
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = ensure_index(pc)

    ensure_dataset()
    rows = load_rows()
    print(f"Loaded dataset rows: {len(rows)}")
    print(f"Preferred embedding model: {EMBEDDING_MODEL}")

    progress = load_progress()
    start_index = int(progress.get("next_index", 0))
    upserted = int(progress.get("upserted", 0))
    if start_index > 0:
        print(
            f"Resuming from row index {start_index} "
            f"(already upserted: {upserted})"
        )

    active_model: str | None = None
    total_rows = len(rows)

    for batch_start in range(start_index, total_rows, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        texts = []
        ids = []
        metadatas = []

        for row in batch:
            text = (row.get("chunk") or "").strip()
            if not text:
                continue
            ids.append(row["id"])
            texts.append(text)
            metadatas.append(
                {
                    "url": row.get("url", ""),
                    "topic": "type1_diabetes",
                    "source": row.get("url", "dataset"),
                    "text": text[:1000],
                }
            )

        if not texts:
            continue

        used_model, vectors = embed_with_fallback(gemini_client, texts)
        if active_model is None:
            active_model = used_model
            print(f"Using embedding model: {active_model}")

        if len(vectors) != len(ids):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(vectors)} vectors for "
                f"{len(ids)} texts in batch starting at row {batch_start}."
            )

        records = []
        for _id, vec, meta in zip(ids, vectors, metadatas):
            records.append({"id": _id, "values": vec, "metadata": meta})

        index.upsert(vectors=records)
        upserted += len(records)
        save_progress(batch_start + len(batch), upserted, active_model)
        print(f"Upserted {upserted}/{len(rows)}")
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    clear_progress()
    print("Pinecone ingestion completed successfully")


if __name__ == "__main__":
    main()
