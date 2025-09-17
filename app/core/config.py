import os
from dotenv import load_dotenv

load_dotenv()

# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.100.3:11434")
# # OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
# NEO4J_URL = os.getenv("NEO4J_URL", "bolt://10.50.0.101:7687")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin.admin")

UPLOAD_DIR = "uploads"
ENGLISH_DIR = os.path.join(UPLOAD_DIR, "english")
INDONESIAN_DIR = os.path.join(UPLOAD_DIR, "indonesian")

VLLM_EMBEDDINGS_URL = os.getenv("VLLM_EMBEDDINGS_URL", "http://192.168.100.136:8010/v1/embeddings")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "qwen3-embed")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

# vLLM Chat/LLM Service  
VLLM_CHAT_URL = os.getenv("VLLM_CHAT_URL", "http://192.168.100.136:8006/v1")
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", "mistral7b-4bit")
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "90"))

# Neo4j Configuration (you can import from your core.config)
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://192.168.100.136:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin.admin")

os.makedirs(ENGLISH_DIR, exist_ok=True)
os.makedirs(INDONESIAN_DIR, exist_ok=True)
