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

DATASET_FILE_ID = "1xBi6wggC2qWJBFzkrKgiTisqvJK44pY6"
EMBEDDING_FILE_ID = "1h6shC3j7X817kiB7zfQxZW7VHX4QjO9l"

# =====================================================
# FILE PATHS
# =====================================================
DATASET_PATH = "./data/t1d_big_dataset.jsonl"

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
# LOAD DATASET
# =====================================================
print("Loading dataset...")

data = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded dataset: {len(data)}")

# =====================================================
# INSERT INTO CHROMA
# =====================================================
print("Creating vector database...")

BATCH_SIZE = 100

for i in tqdm(range(0, len(data), BATCH_SIZE)):

    batch = data[i:i+BATCH_SIZE]

    ids = []
    documents = []
    metadatas = []

    for item in batch:

        text = item.get("chunk", "")

        if not text:
            continue

        ids.append(item["id"])
        documents.append(text)

        metadatas.append({
            "url": item.get("url", ""),
            "topic": "type1_diabetes"
        })

    if len(ids) > 0:

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

print("✅ Chroma DB Created Successfully")