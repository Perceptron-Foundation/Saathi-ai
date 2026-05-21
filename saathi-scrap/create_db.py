import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# LOAD EMBEDDING MODEL
# =========================
print("Loading embedding model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# =========================
# INIT CHROMA DB
# =========================
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="t1d_rag"
)

# =========================
# LOAD DATASET
# =========================
DATA_FILE = "./data/t1d_big_dataset.jsonl"

data = []

with open(DATA_FILE, "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} chunks")

# =========================
# INSERT INTO CHROMA
# =========================
BATCH_SIZE = 100

for i in tqdm(range(0, len(data), BATCH_SIZE)):

    batch = data[i:i+BATCH_SIZE]

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for item in batch:

        text = item["chunk"]

        embedding = model.encode(text).tolist()

        ids.append(item["id"])
        documents.append(text)
        embeddings.append(embedding)

        metadatas.append({
            "url": item.get("url", "")
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

print("✅ Chroma DB Created Successfully")