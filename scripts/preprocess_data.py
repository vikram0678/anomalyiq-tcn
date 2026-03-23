import numpy as np
import joblib
import os
import requests
import zipfile
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 100
RAW_DIR     = "data/raw"
PROC_DIR    = "data/processed"
os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── Auto detect Kaggle or Local ──────────────────────────
KAGGLE_PATH = "/kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data/data"
LOCAL_PATH  = "data/raw/nasa_data/data"

if os.path.exists(KAGGLE_PATH):
    print("Running on Kaggle ✅")
    train_dir = os.path.join(KAGGLE_PATH, "train")
    test_dir  = os.path.join(KAGGLE_PATH, "test")
else:
    print("Running locally — downloading data...")
    url      = "https://github.com/khundman/telemanom/archive/refs/heads/master.zip"
    zip_path = os.path.join(RAW_DIR, "data.zip")
    if not os.path.exists(zip_path):
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Download complete ✅")
    extract = os.path.join(RAW_DIR, "nasa_data")
    if not os.path.exists(extract):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract)
        print("Extracted ✅")

    # walk to find train/test
    train_dir = None
    test_dir  = None
    for root, dirs, files in os.walk(extract):
        npy = [f for f in files if f.endswith(".npy")]
        if npy:
            name = os.path.basename(root).lower()
            if "train" in name:
                train_dir = root
            elif "test" in name:
                test_dir = root

    if not train_dir or not test_dir:
        # use already processed data if exists
        if os.path.exists(os.path.join(PROC_DIR, "train.npy")):
            print("Processed data already exists ✅")
            exit()
        raise FileNotFoundError("Could not find train/test folders")

print(f"Train dir: {train_dir}")
print(f"Test  dir: {test_dir}")

# ── Load files ───────────────────────────────────────────
def load_dir(d, max_files=10):
    arrays = []
    files  = sorted([f for f in os.listdir(d) if f.endswith(".npy")])[:max_files]
    for f in files:
        arr = np.load(os.path.join(d, f))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
        print(f"  {f} → {arr.shape}")
    return arrays

def pad_and_concat(arrays):
    max_ch = max(a.shape[1] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[1] < max_ch:
            pad = np.zeros((a.shape[0], max_ch - a.shape[1]))
            a   = np.hstack([a, pad])
        padded.append(a)
    return np.concatenate(padded, axis=0)

train_raw = pad_and_concat(load_dir(train_dir))
test_raw  = pad_and_concat(load_dir(test_dir))
print(f"\nTrain shape : {train_raw.shape}")
print(f"Test  shape : {test_raw.shape}")

# ── Normalize ────────────────────────────────────────────
scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)
joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))

# ── Windows ──────────────────────────────────────────────
def create_windows(data, w):
    return np.array([data[i:i+w] for i in range(len(data)-w+1)])

train_windows = create_windows(train_scaled, WINDOW_SIZE)
test_windows  = create_windows(test_scaled,  WINDOW_SIZE)

np.save(os.path.join(PROC_DIR, "train.npy"),    train_windows)
np.save(os.path.join(PROC_DIR, "test.npy"),     test_windows)
np.save(os.path.join(PROC_DIR, "test_raw.npy"), test_scaled)

print(f"\ntrain.npy : {train_windows.shape}")
print(f"test.npy  : {test_windows.shape}")
print("Preprocessing complete ✅")
