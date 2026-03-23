# AnomalyIQ-TCN

An unsupervised time-series anomaly detector using a Temporal Convolutional Network (TCN) Autoencoder, trained on NASA SMAP/MSL sensor data to flag deviations via reconstruction error and statistical thresholding. Features an interactive Streamlit dashboard with Docker deployment and a complete ML pipeline covering preprocessing, training, and evaluation.

---

## 📌 Project Overview

- Train a TCN Autoencoder **only on normal data**
- Model learns to reconstruct normal patterns accurately
- When fed **anomalous data**, reconstruction error is high
- High reconstruction error = **anomaly detected**
- Requires **no labels** — fully unsupervised

---

## 📁 Project Structure
```
anomalyiq-tcn/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── submission.json
├── requirements.txt
├── README.md
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.npy           ← download from HuggingFace
│       ├── test.npy            ← download from HuggingFace
│       ├── test_raw.npy        ← download from HuggingFace
│       └── scaler.pkl          ← download from HuggingFace
├── models/
│   └── tcn_autoencoder.pth     ← download from HuggingFace
├── results/
│   ├── anomaly_scores.csv
│   ├── anomalies_percentile.csv
│   ├── anomalies_pot.csv
│   └── streamlit_report.json
└── docs/
    └── TCN_vs_LSTM.md
```

---

## 📥 Download Pre-trained Model and Data

Large files are hosted on Hugging Face (too large for GitHub):

👉 https://huggingface.co/datasets/vikram006/anomalyiq-tcn-data/tree/main

### Option 1 — Auto Download Script (Recommended)
```bash
# Install huggingface_hub
pip install huggingface_hub

# Run this to download all files into correct folders
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

files = [
    ('data/processed/train.npy',    'data/processed/train.npy'),
    ('data/processed/test.npy',     'data/processed/test.npy'),
    ('data/processed/test_raw.npy', 'data/processed/test_raw.npy'),
    ('data/processed/scaler.pkl',   'data/processed/scaler.pkl'),
    ('models/tcn_autoencoder.pth',  'models/tcn_autoencoder.pth'),
]

for repo_path, local_path in files:
    print(f'Downloading {local_path}...')
    hf_hub_download(
        repo_id='vikram006/anomalyiq-tcn-data',
        filename=repo_path,
        repo_type='dataset',
        local_dir='.'
    )
    print(f'Saved to {local_path} ✅')

print('All files downloaded and placed correctly ✅')
"
```

### Option 2 — Download All at Once
```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vikram006/anomalyiq-tcn-data',
    repo_type='dataset',
    local_dir='.'
)
print('All files downloaded ✅')
"
```

### After Download — Verify Files Exist
```bash
# Check all required files are in place
ls data/processed/
# Expected: train.npy  test.npy  test_raw.npy  scaler.pkl

ls models/
# Expected: tcn_autoencoder.pth
```

---

## 🚀 Quick Start — Docker (Recommended)
```bash
# 1. Clone the repo
git clone https://github.com/vikram0678/anomalyiq-tcn.git
cd anomalyiq-tcn

# 2. Download large files from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vikram006/anomalyiq-tcn-data',
    repo_type='dataset',
    local_dir='.'
)
"

# 3. Start the app
docker-compose up --build

# 4. Open dashboard
# http://localhost:8501
```

---

## 🔧 Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Step 1 — Download large files
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vikram006/anomalyiq-tcn-data',
    repo_type='dataset',
    local_dir='.'
)
"

# Step 2 — Preprocess data (or skip if downloaded above)
python scripts/preprocess_data.py

# Step 3 — Train model (or skip if downloaded above)
python scripts/train.py

# Step 4 — Evaluate
python scripts/evaluate.py

# Step 5 — Run dashboard
streamlit run app/main.py
```

---

## 🧠 Model — TCN Autoencoder
```
Encoder:
  Input (25 channels, 100 timesteps)
  → TCNBlock (dilation=1)
  → TCNBlock (dilation=2)
  → TCNBlock (dilation=4)
  → TCNBlock (dilation=8)
  → Latent Space (32 dims)

Decoder:
  Latent Space (32 dims)
  → TCNBlock (dilation=1)
  → TCNBlock (dilation=2)
  → TCNBlock (dilation=4)
  → TCNBlock (dilation=8)
  → Output (25 channels, 100 timesteps)
```

### Why TCN over LSTM?

| Property           | TCN      | LSTM   |
|--------------------|----------|--------|
| Training Speed     | Fast     | Slow   |
| Gradient Stability | High     | Medium |
| Parallelism        | Yes      | No     |
| Receptive Field    | Explicit | Hidden |

See full comparison → [`docs/TCN_vs_LSTM.md`](docs/TCN_vs_LSTM.md)

---

## 🎯 Anomaly Detection Methods

**Percentile** — Takes 99th percentile of reconstruction errors as threshold. Simple and reliable.

**POT (Peak Over Threshold)** — Fits a Generalized Pareto Distribution to the tail of errors. More statistically robust for rare extreme events.

---

## 📊 Dashboard Features

- **Anomaly Score Timeline** — Interactive chart with adjustable threshold slider
- **Signal Explorer** — View any combination of sensor channels
- **Channel Contribution** — Bar chart showing which channel caused the anomaly
- **Generate Full Report** — Downloads `results/streamlit_report.json`

---

## 📈 Results

| Metric                | Value         |
|-----------------------|---------------|
| Dataset               | NASA SMAP/MSL |
| Training windows      | 19,080        |
| Test windows          | 75,370        |
| Model parameters      | 918,041       |
| Training time         | 351.4s        |
| Final training loss   | 0.000013      |
| Percentile anomalies  | 754           |
| POT anomalies         | 750           |

---

## 📦 Dataset

**NASA SMAP and MSL Dataset**
- Real spacecraft telemetry from SMAP satellite and Curiosity Rover
- 82 unique telemetry channels, 496,444 total values
- Source: [Kaggle — patrickfleith](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

---

## 🌍 Environment Variables
```env
STREAMLIT_SERVER_PORT=8501
DATASET_NAME=NASA_SMAP
WINDOW_SIZE=100
LATENT_DIM=32
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=0.001
PERCENTILE_QUANTILE=0.99
POT_INITIAL_QUANTILE=0.95
```

---

## 🛠️ Tech Stack

| Tool         | Purpose               |
|--------------|-----------------------|
| PyTorch      | TCN Autoencoder model |
| Streamlit    | Interactive dashboard |
| Plotly       | Interactive charts    |
| Scikit-learn | Data normalization    |
| SciPy        | POT thresholding      |
| Docker       | Containerization      |
| NumPy/Pandas | Data processing       |

---

## 📚 References

- [TCN Original Paper](https://arxiv.org/abs/1803.01271)
- [NASA Telemanom Paper](https://arxiv.org/abs/1802.04431)
- [Streamlit Docs](https://docs.streamlit.io)
- [POT — Extreme Value Theory](https://en.wikipedia.org/wiki/Generalized_Pareto_distribution)

---

## 👤 Author

**Vikram** — [@vikram0678](https://github.com/vikram0678)