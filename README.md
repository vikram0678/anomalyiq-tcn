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
│       ├── train.npy
│       ├── test.npy
│       ├── test_raw.npy
│       └── scaler.pkl
├── models/
│   └── tcn_autoencoder.pth
├── results/
│   ├── anomaly_scores.csv
│   ├── anomalies_percentile.csv
│   ├── anomalies_pot.csv
│   └── streamlit_report.json
└── docs/
    └── TCN_vs_LSTM.md
```

---

## 🚀 Quick Start — Docker (Recommended)
```bash
# 1. Clone the repo
git clone https://github.com/vikram0678/anomalyiq-tcn.git
cd anomalyiq-tcn

# 2. Start the app
docker-compose up --build

# 3. Open dashboard
# http://localhost:8501
```

---

## 🔧 Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Step 1 — Preprocess data
python scripts/preprocess_data.py

# Step 2 — Train model
python scripts/train.py

# Step 3 — Evaluate
python scripts/evaluate.py

# Step 4 — Run dashboard
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

| Metric                | Value      |
|-----------------------|------------|
| Dataset               | NASA SMAP/MSL |
| Training windows      | 19,080     |
| Test windows          | 75,370     |
| Model parameters      | 918,041    |
| Training time         | 351.4s     |
| Final training loss   | 0.000013   |
| Percentile anomalies  | 754        |
| POT anomalies         | 750        |

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

| Tool         | Purpose                  |
|--------------|--------------------------|
| PyTorch      | TCN Autoencoder model    |
| Streamlit    | Interactive dashboard    |
| Plotly       | Interactive charts       |
| Scikit-learn | Data normalization       |
| SciPy        | POT thresholding         |
| Docker       | Containerization         |
| NumPy/Pandas | Data processing          |

---

