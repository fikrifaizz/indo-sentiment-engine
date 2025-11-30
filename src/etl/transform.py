import pandas as pd
import re
import os

SLANG_DICT = {
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "tdk": "tidak", "enggak": "tidak",
    "yg": "yang", "dgn": "dengan", "dr": "dari", "karna": "karena",
    "tp": "tapi", "tpi": "tapi", "utk": "untuk", "sm": "sama",
    "sy": "saya", "aq": "aku",
    "bgt": "banget", "udh": "sudah", "sdh": "sudah", "dah": "sudah",
    "blm": "belum", "trus": "terus", "skrg": "sekarang",
    "aja": "saja",
    "brg": "barang", "brng": "barang",
    "bgs": "bagus",
    "pake": "pakai",
    "sampe": "sampai", "nyampe": "sampai",
    "dtg": "datang",
    "pesen": "pesan",
    "cepet": "cepat",
    "rapih": "rapi",
    "rmh": "rumah",
    "tgl": "tanggal",
    "thanks": "terima kasih", "thank": "terima kasih",
    "good": "bagus", "best": "terbaik",
    "fast": "cepat", "slow": "lambat",
    "packing": "pengemasan", "seller": "penjual"
}

def normalize_slang(text):
    words = text.split()
    norm_words = [SLANG_DICT.get(word, word) for word in words]
    return " ".join(norm_words)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    if text == "nan":
        return ""
    
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = normalize_slang(text)

    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def map_label(rating):
    if rating <= 2:
        return 0 # Negative
    elif rating == 3:
        return 1 # Neutral
    else:
        return 2 # Positive

def run_transform(input_path, output_path):
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Total data awal: {initial_count}")

    df = df.dropna(subset=['reviewContent', 'rating'])
    df = df[df['reviewContent'].astype(str) != 'nan']

    df = df.drop_duplicates(subset=['reviewContent'])

    df['clean_text'] = df['reviewContent'].apply(clean_text)

    df['num_words'] = df['clean_text'].apply(lambda x: len(x.split()))
    df = df[df['num_words'] >= 3]

    df['label'] = df['rating'].apply(map_label)

    final_count = len(df)
    removed = initial_count - final_count
    
    final_df = df[['clean_text', 'label', 'rating']]

    print(f"\nSTATISTIK AKHIR:")
    print(f"Awal   : {initial_count}")
    print(f"Bersih : {final_count}")
    print(f"Sampah : {removed} ({removed/initial_count*100:.2f}%)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Data bersih tersimpan di: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "20191002-reviews.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "train_cleaned.parquet")
    
    run_transform(INPUT_FILE, OUTPUT_FILE)

    