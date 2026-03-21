import numpy as np
import pandas as pd
import os
import requests
import zipfile
import joblib
from sklearn.preprocessing import MinMaxScaler

# ── Config
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 100))
RAW_DIR     = "data/raw"
PROC_DIR    = "data/processed"
os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── Download NASA SMAP data 
    url = "https://github.com/khundman/telemanom/archive/refs/heads/master.zip"
    zip_path = os.path.join(RAW_DIR, "telemanom.zip")
    if not os.path.exists(zip_path):
        print("Downloading NASA SMAP/MSL dataset...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Already downloaded.")

    extract_path = os.path.join(RAW_DIR, "telemanom-master")
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
    return extract_path

# ── Load .npy files
def load_npy_data(base_path):
    train_dir = os.path.join(base_path, "data", "train")
    test_dir  = os.path.join(base_path, "data", "test")

    def load_dir(d):
        arrays = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(d, f))
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
        return np.concatenate(arrays, axis=0)

    train = load_dir(train_dir)
    test  = load_dir(test_dir)
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test

# ── Normalize
def normalize(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled  = scaler.transform(test)
    joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))
    return train_scaled, test_scaled

# ── Sliding Window 
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

# ── Main
def main():
    base_path = download_data()
    train_raw, test_raw = load_npy_data(base_path)
    train_scaled, test_scaled = normalize(train_raw, test_raw)

    train_windows = create_windows(train_scaled, WINDOW_SIZE)
    test_windows  = create_windows(test_scaled,  WINDOW_SIZE)

    np.save(os.path.join(PROC_DIR, "train.npy"), train_windows)
    np.save(os.path.join(PROC_DIR, "test.npy"),  test_windows)
    np.save(os.path.join(PROC_DIR, "test_raw.npy"), test_scaled)

    print(f"Saved train.npy: {train_windows.shape}")
    print(f"Saved test.npy:  {test_windows.shape}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()