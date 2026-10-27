import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LEGAL_DIR = BASE_DIR / "data" / "legal"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "legal.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",
)
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "900"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
