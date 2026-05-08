"""
Method Ensemble: IF + LOF + LSTM Autoencoder Voting
=====================================================
Evaluate whether consensus voting among 3 unsupervised methods
improves anomaly detection over individual methods.
"""
import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = Path("output/task_b")


def load_data():
    df = pd.read_csv("data/skills_raw_merged.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))
    labels = pd.read_csv(OUT / "merged_rule_labels.csv")["rule_label"].values
    return df, labels


def run_if(X):
    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42, n_jobs=-1)
    return iso.fit_predict(X)


def run_lof(X):
    n_comp = min(80, X.shape[1] // 2, X.shape[0] // 10)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1, novelty=False, n_jobs=-1)
    return lof.fit_predict(X_red)


def run_lstm_ae():
    return json.loads(Path(OUT / "lstm_autoencoder_results.json").read_text())


def evaluate(preds, labels, name):
    bp = (preds == -1).astype(int)
    bt = (labels != "normal").astype(int)
    return {
        "method": name,
        "f1": round(f1_score(bt, bp, zero_division=0), 4),
        "precision": round(precision_score(bt, bp, zero_division=0), 4),
        "recall": round(recall_score(bt, bp, zero_division=0), 4),
        "accuracy": round(accuracy_score(bt, bp), 4),
        "n_flagged": int((preds == -1).sum()),
    }


def main():
    print("=" * 60)
    print("  Method Ensemble: IF + LOF + LSTM AE Voting")
    print("=" * 60)

    df, labels = load_data()
    bt = (labels != "normal").astype(int)

    # Build features
    print("\n[1] Building features...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                            stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    X = StandardScaler().fit_transform(
        tfidf.fit_transform(df["text"]).toarray().astype(np.float32))

    # Run each method
    print("[2] Running detectors...")
    if_preds = run_if(X)
    lof_preds = run_lof(X)

    # LSTM AE results from saved file (trained on normal, detects on all)
    lstm_res = run_lstm_ae()
    # Reconstruct LSTM predictions: use threshold from saved results
    lstm_threshold = lstm_res["threshold"]
    print(f"  LSTM AE threshold: {lstm_threshold:.6f}")

    # LSTM preds: we need to re-run LSTM detection or load from saved
    # For now, use the known anomaly count to estimate
    lstm_anomaly_pct = lstm_res["anomaly_pct"] / 100
    print(f"  LSTM AE flagged: {lstm_res['n_anomalies']} ({lstm_res['anomaly_pct']}%)")

    # Individual evaluations
    results = []
    for name, preds in [("Isolation Forest", if_preds), ("LOF", lof_preds)]:
        r = evaluate(preds, labels, name)
        results.append(r)
        print(f"  {name}: F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # Add LSTM AE from saved
    # We need actual LSTM predictions — let me compute them
    import torch
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.task_b_lstm_autoencoder import LSTMAutoencoder, build_tfidf as lstm_tfidf

    print("\n[3] Re-running LSTM AE for predictions...")
    X_lstm, _ = lstm_tfidf(df["text"].tolist(), max_features=500, ngram_range=(5, 5))
    X_lstm = StandardScaler().fit_transform(X_lstm).astype(np.float32)
    X_seq = torch.tensor(X_lstm[:, np.newaxis, :])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMAutoencoder(500, hidden_dim=128, latent_dim=64, dropout=0.2).to(device)

    # Train on normal
    normal_mask = labels == "normal"
    X_normal = X_seq[normal_mask]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_normal), batch_size=1024, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    model.train()
    for epoch in range(20):
        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            recon, _ = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            opt.step()

    # Detect
    model.eval()
    errors = []
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_seq), batch_size=1024)
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            recon, _ = model(bx)
            mae = torch.abs(recon - bx).mean(dim=(1, 2))
            errors.extend(mae.cpu().numpy())

    errors = np.array(errors)
    threshold = np.percentile(errors[normal_mask], 90)
    lstm_preds = np.where(errors > threshold, -1, 1)

    r = evaluate(lstm_preds, labels, "LSTM Autoencoder")
    results.append(r)
    print(f"  LSTM Autoencoder: F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # ── Ensemble strategies ──
    print("\n[4] Ensemble voting strategies...")

    # Strategy 1: Majority (≥2/3 vote anomaly)
    votes = np.stack([if_preds == -1, lof_preds == -1, lstm_preds == -1], axis=1).sum(axis=1)
    maj_preds = np.where(votes >= 2, -1, 1)
    r = evaluate(maj_preds, labels, "Majority (≥2/3)")
    results.append(r)
    print(f"  Majority (≥2/3):  F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # Strategy 2: Unanimous (3/3)
    una_preds = np.where(votes >= 3, -1, 1)
    r = evaluate(una_preds, labels, "Unanimous (3/3)")
    results.append(r)
    print(f"  Unanimous (3/3):  F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # Strategy 3: Weighted (XGBoost-like weights based on individual F1)
    weights = {"IF": 0.068, "LOF": 0.103, "LSTM": 0.072}
    w_sum = sum(weights.values())
    w_norm = {k: v / w_sum for k, v in weights.items()}
    w_if = np.where(if_preds == -1, w_norm["IF"], 0)
    w_lof = np.where(lof_preds == -1, w_norm["LOF"], 0)
    w_lstm = np.where(lstm_preds == -1, w_norm["LSTM"], 0)
    w_score = w_if + w_lof + w_lstm
    # Threshold at 0.4 (balanced between 1/3 and 2/3)
    w_preds = np.where(w_score >= 0.4, -1, 1)
    r = evaluate(w_preds, labels, "Weighted (F1-based)")
    results.append(r)
    print(f"  Weighted (F1):    F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # Strategy 4: Any (≥1/3, highest recall)
    any_preds = np.where(votes >= 1, -1, 1)
    r = evaluate(any_preds, labels, "Any (≥1/3)")
    results.append(r)
    print(f"  Any (≥1/3):       F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} flagged={r['n_flagged']}")

    # Agreement matrix
    both_if_lof = ((if_preds == -1) & (lof_preds == -1)).sum()
    both_if_lstm = ((if_preds == -1) & (lstm_preds == -1)).sum()
    both_lof_lstm = ((lof_preds == -1) & (lstm_preds == -1)).sum()
    all_three = ((if_preds == -1) & (lof_preds == -1) & (lstm_preds == -1)).sum()

    agreement = {
        "IF ∩ LOF": int(both_if_lof), "IF ∩ LSTM": int(both_if_lstm),
        "LOF ∩ LSTM": int(both_lof_lstm), "IF ∩ LOF ∩ LSTM": int(all_three),
    }

    # Best strategy
    best = max([r for r in results if "Ensemble" not in r["method"] and "Voting" not in r["method"]],
               key=lambda x: x["f1"])
    best_ens = max([r for r in results if "≥" in r["method"] or "Weighted" in r["method"] or "Any" in r["method"]],
                   key=lambda x: x["f1"])

    print(f"\n  Best individual:  {best['method']} (F1={best['f1']:.4f})")
    print(f"  Best ensemble:    {best_ens['method']} (F1={best_ens['f1']:.4f})")
    print(f"  Agreement matrix: {agreement}")

    # Save
    report = {
        "individual_results": results[:3],
        "ensemble_results": results[3:],
        "agreement_matrix": agreement,
        "best_individual": best["method"],
        "best_ensemble": best_ens["method"],
    }
    Path(OUT / "ensemble_results.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Saved → output/task_b/ensemble_results.json")


if __name__ == "__main__":
    main()
