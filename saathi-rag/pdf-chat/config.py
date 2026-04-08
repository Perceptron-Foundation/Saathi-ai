from dotenv import load_dotenv
import os

load_dotenv() 

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = "type1-diabetes-pdf-rag"
PINECONE_CLOUD     = "aws"        
PINECONE_REGION    = "us-east-1"            

EMBEDDING_MODEL    = "neuml/pubmedbert-base-embeddings"
EMBEDDING_DIM      = 768

CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 64

TOP_K              = 5
SIMILARITY_METRIC  = "cosine"

GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL       = "gemini-2.5-flash" 
LLM_TEMPERATURE    = 0.2
LLM_MAX_TOKENS     = 1024

METADATA_FILE      = "chunks_metadata.json"