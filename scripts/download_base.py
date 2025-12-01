import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "models", "indobert-base-p1")

MODEL_NAME = "indobenchmark/indobert-base-p1"

def download_model():
    print(f"Mengunduh model base '{MODEL_NAME}' dari HuggingFace...")
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("Downloading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Downloading Base Model Weights...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_PATH)
    print("Downloading Classification Config...")
    cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    cls_model.save_pretrained(SAVE_PATH)

    print(f"Selesai! Model tersimpan di: {SAVE_PATH}")

if __name__ == "__main__":
    download_model()