"""
Pipeline: Weak Labeling → Feature Engineering → Anomaly Detection (Task B).
Usage: python3 -m src.pipeline.02_label_and_detect
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_skills
from src.features.structural_features import extract as extract_structural
from src.labeling.rule_labeling import label as rule_label
from src.detection.isolation_forest import detect as run_if
from src.detection.local_outlier_factor import detect as run_lof
from src.detection.xgboost_classifier import train as train_xgb
from src.evaluation.metrics import evaluate

OUT = Path("output/task_b")
OUT.mkdir(parents=True, exist_ok=True)


def build_tfidf(df, max_features=500):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    X = vec.fit_transform(df["text"]).toarray().astype(np.float32)
    return StandardScaler().fit_transform(X), vec


def main():
    print("Pipeline: Label → Detect (Task B)")
    print("=" * 50)

    # Load
    df = load_skills()
    print(f"[1] Loaded {len(df)} skills")

    # Label
    structural = extract_structural(df)
    labels, conf = rule_label(df, structural)
    print(f"[2] Labels: {pd.Series(labels).value_counts().to_dict()}")

    # Features
    X_tfidf, tfidf_vec = build_tfidf(df)
    X_struct = StandardScaler().fit_transform(structural.values.astype(float))
    X = np.hstack([X_tfidf, X_struct])
    print(f"[3] Features: {X.shape[1]} dims (TF-IDF {X_tfidf.shape[1]} + Struct {X_struct.shape[1]})")

    # Unsupervised
    _, if_preds, _ = run_if(X_tfidf)
    _, lof_preds, _, _ = run_lof(X_tfidf)
    if_e = evaluate(if_preds, labels); lof_e = evaluate(lof_preds, labels)
    print(f"[4] IF: F1={if_e['f1']:.4f}, LOF: F1={lof_e['f1']:.4f}")

    # Supervised
    xgb, cv_f1, cv_acc, le = train_xgb(X, labels)
    xgb_preds = le.inverse_transform(xgb.predict(X))
    xgb_binary = np.where(xgb_preds != "normal", -1, 1)
    xgb_e = evaluate(xgb_binary, labels)
    print(f"[5] XGBoost: CV-F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}, test-F1={xgb_e['f1']:.4f}")

    # Save
    out = df[["name", "category", "description"]].copy()
    out["weak_label"] = labels
    out["if_pred"] = if_preds
    out["lof_pred"] = lof_preds
    out["xgb_pred"] = xgb_preds
    out.to_csv(OUT / "detection_results.csv", index=False)

    report = {
        "unsupervised": {"IF": if_e, "LOF": lof_e},
        "supervised": {"XGBoost": {"cv_f1": round(cv_f1.mean(), 4), "cv_acc": round(cv_acc.mean(), 4),
                                    "test_f1": xgb_e["f1"]}},
        "label_distribution": pd.Series(labels).value_counts().to_dict(),
    }
    json.dump(report, open(OUT / "detection_report.json", "w"), indent=2, default=str)
    print(f"[6] Saved → {OUT}/detection_results.csv + detection_report.json")


if __name__ == "__main__":
    main()
