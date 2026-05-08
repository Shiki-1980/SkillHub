"""
Task B Extension: SBERT Embedding Benchmark
=============================================
Replaces TF-IDF with SBERT sentence embeddings for anomaly detection.
Compares: TF-IDF vs SBERT vs Hybrid (TF-IDF + SBERT) across IF / LOF / XGBoost.

Based on:
  - "Comparative analysis of anomaly detection algorithms in text data" (2024):
    22 TAD algorithms on 17 corpora, SBERT + semi-supervised > unsupervised
  - "Extract sentence embeddings from pretrained transformer models" (2024):
    Multi-layer aggregation + post-processing improves STS/clustering
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/task_b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================
# 1. SBERT EMBEDDING
# ============================================================

def build_sbert_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    """Generate SBERT sentence embeddings."""
    print(f"  Loading SBERT: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    print(f"  Embedding dim: {embeddings.shape[1]}")
    return embeddings, model


# ============================================================
# 2. EVALUATION UTILITIES
# ============================================================

def run_isolation_forest(X, contamination=0.1):
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)
    raw = iso.decision_function(X)
    scores = (raw.max() - raw) / (raw.max() - raw.min() + 1e-6)
    return iso, preds, scores


def run_lof(X, n_neighbors=30, contamination=0.1):
    n_comp = min(80, X.shape[1] // 2, X.shape[0] // 10)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,
                             novelty=False, n_jobs=-1)
    preds = lof.fit_predict(X_red)
    raw = -lof.negative_outlier_factor_
    scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    return lof, preds, scores


def evaluate(preds, labels):
    binary_pred = (preds == -1).astype(int)
    binary_true = (labels != "normal").astype(int)
    return {
        "f1": f1_score(binary_true, binary_pred, zero_division=0),
        "precision": precision_score(binary_true, binary_pred, zero_division=0),
        "recall": recall_score(binary_true, binary_pred, zero_division=0),
        "accuracy": accuracy_score(binary_true, binary_pred),
    }


def train_xgb(X, y):
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(xgb, X, y, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(xgb, X, y, cv=skf, scoring="accuracy")
    xgb.fit(X, y)
    return xgb, cv_f1, cv_acc


# ============================================================
# 3. MAIN BENCHMARK
# ============================================================

def main():
    print("=" * 70)
    print("  SBERT Embedding Benchmark for Skill Anomaly Detection")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_DIR / "skills_raw.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))

    # Load labels from Task B
    results_df = pd.read_csv(OUTPUT_DIR / "anomaly_results.csv")
    labels = results_df["weak_label"].values
    print(f"  {len(df)} skills, labels: {pd.Series(labels).value_counts().to_dict()}")

    # Feature sets
    print("\n[2] Building feature sets...")

    # TF-IDF (reduced to 500 for fair comparison with 384-dim SBERT)
    print("  [a] TF-IDF (500d, baseline)...")
    tfidf_vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    X_tfidf = tfidf_vec.fit_transform(df["text"]).toarray().astype(np.float32)
    print(f"      Shape: {X_tfidf.shape}")

    # SBERT
    print("  [b] SBERT (all-MiniLM-L6-v2, 384d)...")
    X_sbert, sbert_model = build_sbert_embeddings(df["text"].tolist())
    X_sbert = X_sbert.astype(np.float32)
    print(f"      Shape: {X_sbert.shape}")

    # Structural features only
    print("  [c] Loading structural features...")
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.task_b_anomaly_detection import extract_structural_features
    structural = extract_structural_features(df)
    structural_scaled = StandardScaler().fit_transform(structural.values.astype(float)).astype(np.float32)

    # Build comparison feature sets (keep dims manageable)
    X_tfidf_s = StandardScaler().fit_transform(X_tfidf).astype(np.float32)
    X_sbert_s = StandardScaler().fit_transform(X_sbert).astype(np.float32)
    X_tfidf_struct = np.hstack([X_tfidf_s, structural_scaled])
    X_sbert_struct = np.hstack([X_sbert_s, structural_scaled])
    print(f"      TF-IDF+Struct: {X_tfidf_struct.shape}, SBERT+Struct: {X_sbert_struct.shape}")

    # Labels for XGBoost
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Benchmark
    print("\n[3] Benchmarking...")
    all_results = []

    feature_sets = {
        "TF-IDF (500d)": X_tfidf_s,
        "SBERT (384d)": X_sbert_s,
        "TF-IDF + Structural": X_tfidf_struct,
        "SBERT + Structural": X_sbert_struct,
    }

    for name, X in feature_sets.items():
        print(f"\n  {'─'*50}")
        print(f"  {name} ({X.shape[1]}d)")

        # Isolation Forest
        _, if_preds, if_scores = run_isolation_forest(X)
        if_eval = evaluate(if_preds, labels)
        print(f"    IF:     F1={if_eval['f1']:.4f}  P={if_eval['precision']:.4f}  "
              f"R={if_eval['recall']:.4f}  Acc={if_eval['accuracy']:.4f}")
        all_results.append({"method": f"IF - {name}", **if_eval})

        # LOF
        _, lof_preds, _ = run_lof(X, n_neighbors=30, contamination=0.1)
        lof_eval = evaluate(lof_preds, labels)
        print(f"    LOF:    F1={lof_eval['f1']:.4f}  P={lof_eval['precision']:.4f}  "
              f"R={lof_eval['recall']:.4f}  Acc={lof_eval['accuracy']:.4f}")
        all_results.append({"method": f"LOF - {name}", **lof_eval})

        # XGBoost
        xgb, cv_f1, cv_acc = train_xgb(X, y)
        print(f"    XGBoost: CV F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}  "
              f"CV Acc={cv_acc.mean():.4f}±{cv_acc.std():.4f}")
        all_results.append({
            "method": f"XGB - {name}",
            "cv_f1_macro_mean": cv_f1.mean(),
            "cv_f1_macro_std": cv_f1.std(),
            "cv_accuracy_mean": cv_acc.mean(),
        })

    # Summary table
    print(f"\n{'='*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Best per method type
    print(f"\n  Best IF F1:")
    if_results = [r for r in all_results if r["method"].startswith("IF")]
    for r in sorted(if_results, key=lambda x: x["f1"], reverse=True):
        print(f"    {r['method']:<45} F1={r['f1']:.4f}")

    print(f"\n  Best LOF F1:")
    lof_results = [r for r in all_results if r["method"].startswith("LOF")]
    for r in sorted(lof_results, key=lambda x: x["f1"], reverse=True):
        print(f"    {r['method']:<45} F1={r['f1']:.4f}")

    print(f"\n  Best XGBoost CV F1:")
    xgb_results = [r for r in all_results if r["method"].startswith("XGB")]
    for r in sorted(xgb_results, key=lambda x: x["cv_f1_macro_mean"], reverse=True):
        print(f"    {r['method']:<45} F1={r['cv_f1_macro_mean']:.4f}")

    # Key comparison: TF-IDF only vs SBERT only
    print(f"\n  Key comparison (feature-only, no structural):")
    for prefix in ["IF", "LOF", "XGB"]:
        tfidf_only = [r for r in all_results
                      if r["method"] == f"{prefix} - TF-IDF (500d)"]
        sbert_only = [r for r in all_results
                      if r["method"] == f"{prefix} - SBERT (384d)"]
        if tfidf_only and sbert_only:
            metric = "f1" if prefix != "XGB" else "cv_f1_macro_mean"
            tfidf_v = tfidf_only[0][metric]
            sbert_v = sbert_only[0][metric]
            delta = sbert_v - tfidf_v
            winner = "SBERT" if delta > 0 else "TF-IDF"
            print(f"    {prefix}: TF-IDF={tfidf_v:.4f}  SBERT={sbert_v:.4f}  "
                  f"Δ={delta:+.4f} → {winner} wins")

    # Save
    output_path = OUTPUT_DIR / "sbert_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {output_path}")
    print(f"\n{'='*70}")
    print("  Benchmark complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
