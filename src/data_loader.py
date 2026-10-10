"""
PhishGuard - Data Acquisition Script
Downloads the public Phishing Email dataset from Hugging Face / Kaggle open mirror.
Dataset: Nazario Phishing Corpus + Enron Legitimate Email corpus compiled dataset.
"""

import os
import urllib.request
import pandas as pd
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)
DEST_PATH = os.path.join(DATA_DIR, "Phishing_Email.csv")

URL = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"

def download_dataset():
    if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) > 1000000:
        print(f"Dataset already exists at: {DEST_PATH} ({os.path.getsize(DEST_PATH):,} bytes)")
        return DEST_PATH

    print(f"Downloading phishing email dataset from {URL}...")
    start_time = time.time()
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}% ({count * block_size / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB)", end="")
    
    urllib.request.urlretrieve(URL, DEST_PATH, reporthook=reporthook)
    elapsed = time.time() - start_time
    print(f"\nDownload complete in {elapsed:.2f}s! File size: {os.path.getsize(DEST_PATH):,} bytes")
    
    # Quick verification
    df = pd.read_csv(DEST_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Class distribution:\n", df['Email Type'].value_counts(dropna=False))
    return DEST_PATH

if __name__ == "__main__":
    download_dataset()
