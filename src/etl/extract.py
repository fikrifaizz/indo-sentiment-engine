import os
from kaggle.api.kaggle_api_extended import KaggleApi

# --- KONFIGURASI ---
DATASET_ID = "grikomsn/lazada-indonesian-reviews"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def extract_data():
    print("Memulai proses ekstraksi data...")
    
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Membuat folder: {RAW_DATA_DIR}")
    api = KaggleApi()
    try:
        api.authenticate()
        print("Autentikasi Kaggle berhasil.")
    except Exception as e:
        print(f"Gagal Autentikasi Kaggle. Pastikan 'kaggle.json' sudah benar.")
        print(f"Detail Error: {e}")
        return

    target_file = os.path.join(RAW_DATA_DIR, "20191002-reviews.csv")
    
    if os.path.exists(target_file):
        print(f"Data sudah ditemukan di: {target_file}")
        print("Mewatikan proses download.")
        return target_file

    print(f"Mengunduh dataset '{DATASET_ID}'...")
    try:
        api.dataset_download_files(DATASET_ID, path=RAW_DATA_DIR, unzip=True)
        print(f"Dataset berhasil didownload dan diekstrak ke: {RAW_DATA_DIR}")
        
        files = os.listdir(RAW_DATA_DIR)
        print(f"File tersedia: {files}")
        
    except Exception as e:
        print(f"Gagal mengunduh dataset: {e}")

if __name__ == "__main__":
    extract_data()