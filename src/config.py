import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "OfflineCursor_Qdrant"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_CODEBASE = os.getenv("COLLECTION_CODEBASE", "codebase_hybrid_v1")
COLLECTION_MEMORY = os.getenv("COLLECTION_MEMORY", "chat_memory_v1")

# Dense embeddings (semantic)
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_VECTOR_SIZE = int(os.getenv("DENSE_VECTOR_SIZE", "384"))

# LLM
LLM_TYPE = os.getenv("LLM_TYPE", "gemini").lower()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL= os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Retrieval
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "6"))
TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "6"))
TOP_K_MEMORY = int(os.getenv("TOP_K_MEMORY", "3"))

# Fusion
RRF_K = int(os.getenv("RRF_K", "60"))

# Prompt limits
MAX_CODE_CHARS_PER_CHUNK = int(os.getenv("MAX_CODE_CHARS_PER_CHUNK", "1800"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "9000"))
