import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "train_cleaned.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "final")

def run_load():
    print(f"Membaca data bersih dari {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} tidak ditemukan.")
        return

    df = pd.read_parquet(INPUT_FILE)

    print("Melakukan 3-Way Split (Train / Val / Test)...")
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5, 
        random_state=42,
        stratify=temp_df['label']
    )
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("LOAD SUKSES! (Skema 70:15:15)")
    print(f"Train Set : {len(train_df)} baris -> {train_path}")
    print(f"Val Set   : {len(val_df)} baris  -> {val_path}")
    print(f"Test Set  : {len(test_df)} baris  -> {test_path}")

if __name__ == "__main__":
    run_load()