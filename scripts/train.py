import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ── Config 
WINDOW_SIZE  = int(os.getenv("WINDOW_SIZE",   100))
LATENT_DIM   = int(os.getenv("LATENT_DIM",    32))
TCN_LAYERS   = int(os.getenv("TCN_LAYERS",    4))
KERNEL_SIZE  = int(os.getenv("TCN_KERNEL_SIZE", 3))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE",    64))
EPOCHS       = int(os.getenv("EPOCHS",        50))
LR           = float(os.getenv("LEARNING_RATE", 0.001))
PROC_DIR     = "data/processed"
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Causal Conv 
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        return self.conv(x)[:, :, :x.size(2)]

# ── TCN Residual Block
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = self.res(x) if self.res else x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return self.relu(out + res)

# ── TCN Autoencoder
class TCNAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, tcn_layers, kernel_size):
        super().__init__()
        channels = [32 * (2 ** i) for i in range(tcn_layers)]

        # Encoder
        enc_layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            enc_layers.append(TCNBlock(ch_in, ch_out, kernel_size, dilation=2**i))
            ch_in = ch_out
        enc_layers.append(nn.Conv1d(ch_in, latent_dim, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        ch_in = latent_dim
        for i, ch_out in enumerate(reversed(channels)):
            dec_layers.append(TCNBlock(ch_in, ch_out, kernel_size, dilation=2**i))
            ch_in = ch_out
        dec_layers.append(nn.Conv1d(ch_in, in_channels, 1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ── Train 
def main():
    train_data = np.load(os.path.join(PROC_DIR, "train.npy")).astype(np.float32)
    # shape: (N, window_size, features) → (N, features, window_size)
    train_data = np.transpose(train_data, (0, 2, 1))
    in_channels = train_data.shape[1]

    dataset    = TensorDataset(torch.tensor(train_data))
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = TCNAutoencoder(in_channels, LATENT_DIM, TCN_LAYERS, KERNEL_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training on {len(train_data)} windows | {in_channels} channels")
    import time
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {avg:.6f}")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    model_path = os.path.join(MODEL_DIR, "tcn_autoencoder.pth")
    torch.save({
        "model_state": model.state_dict(),
        "in_channels": in_channels,
        "latent_dim":  LATENT_DIM,
        "tcn_layers":  TCN_LAYERS,
        "kernel_size": KERNEL_SIZE,
        "training_time_seconds": elapsed
    }, model_path)
    print(f"Model saved → {model_path}")

if __name__ == "__main__":
    main()