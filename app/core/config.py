import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.100.3:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://192.168.100.3:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin.admin")

# Base upload directory
UPLOAD_DIR = "uploads"
ENGLISH_DIR = os.path.join(UPLOAD_DIR, "english")
INDONESIAN_DIR = os.path.join(UPLOAD_DIR, "indonesian")

# Create directories if they don't exist
os.makedirs(ENGLISH_DIR, exist_ok=True)
os.makedirs(INDONESIAN_DIR, exist_ok=True)
