import os
import json
from pathlib import Path

# Base directory (where this file is located)
BASE_DIR = Path(__file__).parent.absolute()

# Default Settings Path
SETTINGS_FILE = BASE_DIR / "settings.json"

def load_settings():
    defaults = {
        "PROJECT_NAME": "IndexGPT",
        "RESEARCH_FIELD": "General Science",
        "BACKEND_PORT": 8176,
        "BASE_MODEL_ID": "",
        "BGE_MODEL_PATH": "",
        "API_KEY": os.environ.get("API_KEY", ""),
        "API_BASE_URL": os.environ.get("API_BASE_URL", "https://api.deepseek.com"),
        "API_MODEL_ID": "deepseek-chat",
        "CHAT_TOPK": 3,
        "COMPARE_TOPK": 12,
        "CHAT_MAX_TOKENS": 1200,
        "CHAT_HISTORY_ROUNDS": 3,
        "SFT_MODEL_SOURCE": "local",  # "local" or "api"
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                user_settings = json.load(f)
                defaults.update(user_settings)
        except:
            pass
    return defaults

settings = load_settings()

# Apply Settings
PROJECT_NAME = settings["PROJECT_NAME"]
RESEARCH_FIELD = settings["RESEARCH_FIELD"]
BACKEND_PORT = settings["BACKEND_PORT"]
BASE_MODEL_ID = settings["BASE_MODEL_ID"]
BGE_MODEL_PATH = Path(settings["BGE_MODEL_PATH"])
API_KEY = settings["API_KEY"]
API_BASE_URL = settings["API_BASE_URL"]
API_MODEL_ID = settings["API_MODEL_ID"]
CHAT_TOPK = settings["CHAT_TOPK"]
COMPARE_TOPK = settings["COMPARE_TOPK"]
CHAT_MAX_TOKENS = settings["CHAT_MAX_TOKENS"]
CHAT_HISTORY_ROUNDS = settings["CHAT_HISTORY_ROUNDS"]
SFT_MODEL_SOURCE = settings["SFT_MODEL_SOURCE"]

# Data and workspace directories
WORKSPACE_DIR = BASE_DIR / "workspace"
PAPERS_DIR = WORKSPACE_DIR / "papers"
RAG_INDEX_DIR = WORKSPACE_DIR / "rag_index"
PARSE_DIR = RAG_INDEX_DIR / "parse"
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
LOGS_DIR = WORKSPACE_DIR / "logs"

CHATS_DIR = WORKSPACE_DIR / "chats"

# Ensure all directories exist
for d in [WORKSPACE_DIR, PAPERS_DIR, RAG_INDEX_DIR, PARSE_DIR, DATA_DIR, OUTPUTS_DIR, LOGS_DIR, CHATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Fixed paths derived from outputs
LORA_MODEL_PATH = OUTPUTS_DIR / "lora"
SFT_DATA_PATH = DATA_DIR / "sft.jsonl"
TRAIN_LOG_PATH = LOGS_DIR / "train.log"

# RAG configuration
CHUNK_MAX_WORDS = 300
RAG_TOPK = 3
