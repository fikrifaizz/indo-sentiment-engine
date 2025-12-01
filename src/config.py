import os

# Base Directory Project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "lazada_reviews"

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
LORA_ADAPTER_PATH = os.path.join(MODEL_DIR, "indobert-lora-finetuned")
BASE_MODEL_NAME = os.path.join(MODEL_DIR, "indobert-base-p1") 

# Hyperparameters
MAX_LEN = 128