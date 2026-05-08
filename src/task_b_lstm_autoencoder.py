"""
Task B Extension: TF-IDF + LSTM Autoencoder for Anomaly Detection
====================================================================
Third unsupervised method, based on:
  Coote & Lachine, "Platform Management System Host-Based Anomaly
  Detection using TF-IDF and an LSTM Autoencoder" (2024)

Architecture:
  TF-IDF (n-gram=5, 500 features) → LSTM Encoder → Latent → LSTM Decoder
  Anomaly = reconstruction error (MAE) > threshold

Key parameters from the paper:
  - n-gram size: 5
  - Corpus size: 500 features
  - Activation: SeLU
  - Optimizer: Adam
  - Loss: MAE
  - Batch size: 1024
  - Epochs: 20
  - Dropout: 0.2
"""

import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("output/task_b")


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for sequence anomaly detection.
    Input shape: (batch, seq_len=1, features) — we treat each skill as 1 timestep
    with feature vector as input dimension."""

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, dropout=0.2):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_to_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim * 2)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=2,
                               batch_first=True, dropout=dropout)
        self.dec_to_out = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, 1, features)
        # Encoder
        enc_out, (h_n, c_n) = self.encoder(x)
        # Use last hidden state from both directions
        h_forward = h_n[-2, :, :]  # forward last layer
        h_backward = h_n[-1, :, :]  # backward last layer
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        latent = self.activation(self.enc_to_latent(h_combined))
        latent = self.dropout(latent)

        # Decoder — reconstruct the input
        dec_input = self.activation(self.latent_to_dec(latent))
        dec_input = dec_input.unsqueeze(1)  # (batch, 1, hidden*2)
        dec_out, _ = self.decoder(dec_input)
        reconstructed = self.dec_to_out(dec_out)
        return reconstructed, latent


def build_tfidf(texts, max_features=500, ngram_range=(5, 5)):
    """Build TF-IDF with paper's parameters: n-gram=5, 500 features."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        max_df=0.8,
        min_df=1,
        analyzer="char_wb",  # character n-grams within word boundaries
    )
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return X, vec


def train_autoencoder(model, train_loader, epochs=20, lr=1e-3):
    """Train the LSTM autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch:2d}/{epochs}  loss={avg_loss:.6f}")

    return losses


def compute_threshold(model, loader, percentile=90):
    """Compute anomaly threshold from reconstruction errors on benign data."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_x, in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed, _ = model(batch_x)
            # Per-sample MAE
            mae = torch.abs(reconstructed - batch_x).mean(dim=(1, 2))
            errors.extend(mae.cpu().numpy().tolist())
    threshold = np.percentile(errors, percentile)
    return threshold, np.array(errors)


def detect_anomalies(model, X_tensor, threshold):
    """Detect anomalies using reconstruction error threshold."""
    model.eval()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=1024, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed, _ = model(batch_x)
            mae = torch.abs(reconstructed - batch_x).mean(dim=(1, 2))
            errors.extend(mae.cpu().numpy().tolist())
    errors = np.array(errors)
    preds = np.where(errors > threshold, -1, 1)
    scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
    return preds, scores, errors


def main():
    print("=" * 70)
    print("  LSTM Autoencoder for Skill Anomaly Detection")
    print("  Based on: Coote & Lachine (2024)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv("data/skills_raw.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))
    labels = pd.read_csv(OUTPUT_DIR / "anomaly_results.csv")["weak_label"].values

    # TF-IDF with paper parameters: n-gram=5, 500 features
    print("\n[2] Building TF-IDF (n-gram=5, 500d)...")
    X_tfidf, tfidf_vec = build_tfidf(df["text"].tolist())
    X_scaled = StandardScaler().fit_transform(X_tfidf).astype(np.float32)
    print(f"  Shape: {X_scaled.shape}")

    # Reshape for LSTM: (samples, seq_len=1, features)
    X_seq = X_scaled[:, np.newaxis, :]
    X_tensor = torch.tensor(X_seq)

    # Split: use normal-labeled data for training (autoencoder learns "normal" patterns)
    normal_mask = labels == "normal"
    X_normal = X_seq[normal_mask]
    X_all_tensor = X_tensor

    print(f"  Train (normal): {X_normal.shape[0]}, Total: {X_seq.shape[0]}")

    # Build model
    print("\n[3] Building LSTM Autoencoder...")
    input_dim = X_seq.shape[2]
    model = LSTMAutoencoder(input_dim, hidden_dim=128, latent_dim=64, dropout=0.2).to(DEVICE)
    print(f"  Input dim: {input_dim}, Hidden: 128, Latent: 64")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Train on normal data
    print("\n[4] Training on normal skills...")
    normal_loader = DataLoader(
        TensorDataset(torch.tensor(X_normal)),
        batch_size=1024, shuffle=True
    )
    train_losses = train_autoencoder(model, normal_loader, epochs=20, lr=1e-3)

    # Compute threshold
    print("\n[5] Computing anomaly threshold (p90)...")
    threshold, train_errors = compute_threshold(model, normal_loader, percentile=90)
    print(f"  Threshold (p90): {threshold:.6f}")

    # Detect anomalies on ALL data
    print("\n[6] Detecting anomalies...")
    preds, scores, all_errors = detect_anomalies(model, X_all_tensor, threshold)

    # Evaluate
    binary_pred = (preds == -1).astype(int)
    binary_true = (labels != "normal").astype(int)
    results = {
        "method": "LSTM Autoencoder (TF-IDF 500d, n-gram=5)",
        "f1": round(f1_score(binary_true, binary_pred, zero_division=0), 4),
        "precision": round(precision_score(binary_true, binary_pred, zero_division=0), 4),
        "recall": round(recall_score(binary_true, binary_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(binary_true, binary_pred), 4),
        "threshold": round(float(threshold), 6),
        "n_anomalies": int((preds == -1).sum()),
        "anomaly_pct": round((preds == -1).mean() * 100, 1),
        "params": {
            "input_dim": input_dim,
            "hidden_dim": 128,
            "latent_dim": 64,
            "epochs": 20,
            "batch_size": 1024,
            "dropout": 0.2,
            "activation": "SeLU",
            "optimizer": "Adam",
            "loss": "MAE",
        },
    }
    print(f"\n  Results: F1={results['f1']:.4f} P={results['precision']:.4f} "
          f"R={results['recall']:.4f} Acc={results['accuracy']:.4f}")
    print(f"  Anomalies flagged: {results['n_anomalies']} ({results['anomaly_pct']}%)")

    # Compare with IF and LOF from Task B
    print(f"\n  Method comparison (3 unsupervised methods vs weak labels):")
    report = json.loads(Path(OUTPUT_DIR / "report.json").read_text())
    for r in report["unsupervised_results"]:
        print(f"    {r['method']:<40} F1={r['f1']:.4f}")
    print(f"    {'LSTM Autoencoder (TF-IDF, ngram=5)':<40} F1={results['f1']:.4f}")

    # Save
    Path(OUTPUT_DIR / "lstm_autoencoder_results.json").write_text(
        json.dumps({**results, "train_losses": [float(l) for l in train_losses]}, indent=2))
    print(f"\n  Results saved → output/task_b/lstm_autoencoder_results.json")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
