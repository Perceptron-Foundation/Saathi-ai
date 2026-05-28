import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv() 

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

embeddings = client.feature_extraction(
    text="Type 1 diabetes insulin management",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print(len(embeddings))
print(embeddings)