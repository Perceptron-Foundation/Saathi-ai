import os
import json
import chromadb
import gdown

from tqdm import tqdm

# =====================================================
# CREATE DATA FOLDER
# =====================================================
os.makedirs("./data", exist_ok=True)

# =====================================================
# GOOGLE DRIVE FILE IDS
# =====================================================

# Replace with your actual Google Drive file IDs

DATASET_FILE_ID = "1xBi6wggC2qWJBFzkrKgiTisqvJK44pY6"
EMBEDDING_FILE_ID = "1h6shC3j7X817kiB7zfQxZW7VHX4QjO9l"

# =====================================================
# FILE PATHS
# =====================================================
DATASET_PATH = "./data/t1d_big_dataset.jsonl"
EMBEDDING_PATH = "./data/t1d_embeddings.jsonl"

# =====================================================
# DOWNLOAD DATASET IF NOT EXISTS
# =====================================================
if not os.path.exists(DATASET_PATH):

    print("Downloading dataset from Google Drive...")

    gdown.download(
        f"https://drive.google.com/uc?id={DATASET_FILE_ID}",
        DATASET_PATH,
        quiet=False
    )

# =====================================================
# DOWNLOAD EMBEDDINGS IF NOT EXISTS
# =====================================================
if not os.path.exists(EMBEDDING_PATH):

    print("Downloading embeddings from Google Drive...")

    gdown.download(
        f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}",
        EMBEDDING_PATH,
        quiet=False
    )

# =====================================================
# INIT CHROMA DB
# =====================================================
print("Initializing Chroma DB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="t1d_rag"
)

# =====================================================
# LOAD EMBEDDINGS
# =====================================================
print("Loading embeddings...")

emb_data = []

with open(EMBEDDING_PATH, "r", encoding="utf-8") as f:
    for line in f:
        emb_data.append(json.loads(line))

print(f"Loaded embeddings: {len(emb_data)}")

# =====================================================
# LOAD ORIGINAL DATASET
# =====================================================
print("Loading original dataset...")

original_map = {}

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        original_map[item["id"]] = item

print(f"Loaded dataset: {len(original_map)}")

# =====================================================
# INSERT INTO CHROMA DB
# =====================================================
print("Creating vector database...")

BATCH_SIZE = 100

for i in tqdm(range(0, len(emb_data), BATCH_SIZE)):

    batch = emb_data[i:i+BATCH_SIZE]

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for item in batch:

        original = original_map.get(item["id"])

        if not original:
            continue

        text = original.get("chunk", "")

        if not text:
            continue

        ids.append(item["id"])
        embeddings.append(item["embedding"])
        documents.append(text)

        metadatas.append({
            "url": original.get("url", ""),
            "topic": "type1_diabetes"
        })

    if len(ids) > 0:

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

print("✅ Chroma DB Created Successfully")
print("✅ Backend is ready to use")