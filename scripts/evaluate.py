import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib, os
from scipy.stats import genpareto
from train import TCNAutoencoder, CausalConv1d, TCNBlock

# ── Config 
PROC_DIR   = "data/processed"
MODEL_DIR  = "models"
RES_DIR    = "results"
PERCENTILE = float(os.getenv("PERCENTILE_QUANTILE",  0.99))
POT_Q      = float(os.getenv("POT_INITIAL_QUANTILE", 0.95))
EMA_ALPHA  = 0.3
os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model
def load_model():
    ckpt = torch.load(os.path.join(MODEL_DIR, "tcn_autoencoder.pth"),
                      map_location=DEVICE)
    model = TCNAutoencoder(ckpt["in_channels"], ckpt["latent_dim"],
                           ckpt["tcn_layers"],  ckpt["kernel_size"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model

# ── Reconstruction error 
def get_errors(model, test_windows):
    data   = np.transpose(test_windows, (0, 2, 1)).astype(np.float32)
    tensor = torch.tensor(data).to(DEVICE)
    errors = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i+batch_size]
            recon = model(batch)
            err   = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(err)
    return np.array(errors)

# ── EMA smoothing
def ema_smooth(errors, alpha=EMA_ALPHA):
    smoothed = np.zeros_like(errors)
    smoothed[0] = errors[0]
    for i in range(1, len(errors)):
        smoothed[i] = alpha * errors[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

# ── Percentile threshold
def percentile_threshold(scores, q=PERCENTILE):
    return np.quantile(scores, q)

# ── POT threshold 
def pot_threshold(scores, q=POT_Q):
    t0      = np.quantile(scores, q)
    excesses = scores[scores > t0] - t0
    if len(excesses) < 10:
        return t0
    c, loc, scale = genpareto.fit(excesses, floc=0)
    prob    = 0.01
    thresh  = t0 + scale / c * ((len(scores) * prob / len(excesses)) ** (-c) - 1)
    return thresh

# ── Main 
def main():
    test_windows = np.load(os.path.join(PROC_DIR, "test.npy"))
    model        = load_model()

    print("Running inference...")
    raw_errors = get_errors(model, test_windows)
    smoothed   = ema_smooth(raw_errors)

    # Save anomaly scores
    scores_df = pd.DataFrame({
        "timestamp":     range(len(raw_errors)),
        "raw_error":     raw_errors,
        "smoothed_error": smoothed
    })
    scores_df.to_csv(os.path.join(RES_DIR, "anomaly_scores.csv"), index=False)

    # Percentile anomalies
    pct_thresh = percentile_threshold(smoothed)
    pct_mask   = smoothed > pct_thresh
    pct_df     = pd.DataFrame({
        "timestamp":    np.where(pct_mask)[0],
        "anomaly_score": smoothed[pct_mask]
    })
    pct_df.to_csv(os.path.join(RES_DIR, "anomalies_percentile.csv"), index=False)

    # POT anomalies
    pot_thresh = pot_threshold(smoothed)
    pot_mask   = smoothed > pot_thresh
    pot_df     = pd.DataFrame({
        "timestamp":    np.where(pot_mask)[0],
        "anomaly_score": smoothed[pot_mask]
    })
    pot_df.to_csv(os.path.join(RES_DIR, "anomalies_pot.csv"), index=False)

    print(f"Percentile threshold : {pct_thresh:.6f} | Anomalies: {pct_mask.sum()}")
    print(f"POT threshold        : {pot_thresh:.6f} | Anomalies: {pot_mask.sum()}")
    print("Results saved to results/")

if __name__ == "__main__":
    main()