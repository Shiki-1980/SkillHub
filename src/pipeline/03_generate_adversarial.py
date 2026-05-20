"""
Pipeline: Adversarial Skill Generation (Task C).
Usage: python3 -m src.pipeline.03_generate_adversarial
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack
from xgboost import XGBClassifier

from src.data.loader import load_skills
from src.features.structural_features import extract as extract_structural
from src.adversarial.perturbation_operators import PerturbationOperators
from src.adversarial.genetic_algorithm import EASG

OUT = Path("output/task_c")
OUT.mkdir(parents=True, exist_ok=True)


def build_models(df, labels):
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words="english",
                            sublinear_tf=True, max_df=0.8, min_df=3)
    X_tfidf = tfidf.fit_transform(df["text"])
    structural = extract_structural(df)
    X_struct = StandardScaler().fit_transform(structural.values.astype(float))
    X_all = hstack([X_tfidf, X_struct]).tocsr()

    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42, n_jobs=-1).fit(X_tfidf)
    svd = TruncatedSVD(n_components=80, random_state=42).fit(X_tfidf.toarray())
    tfidf_red = svd.transform(X_tfidf.toarray())
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1, novelty=True, n_jobs=-1).fit(tfidf_red)
    le = LabelEncoder(); y = le.fit_transform(labels)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_all, y)

    return {"tfidf_vec": tfidf, "iso": iso, "lof": lof, "svd": svd, "xgb": xgb,
            "structural_scaled": X_struct, "label_encoder": le}


def main():
    print("Pipeline: Adversarial Generation (Task C)")
    print("=" * 50)

    df = load_skills()
    labels = pd.read_csv("output/task_b/merged_rule_labels.csv")["rule_label"].values
    anomalous = df[labels.isin(["malicious", "unsafe"])]
    n_seeds = min(100, len(anomalous))
    seeds = anomalous.sample(n_seeds, random_state=42)
    print(f"[1] {n_seeds} seeds from {len(anomalous)} malicious/unsafe (80% train split)")

    models = build_models(df, labels)
    df_normal = df[labels == "normal"]
    ops = PerturbationOperators(df_normal, models["tfidf_vec"])
    print("[2] Models + operators ready")

    ga = EASG(models, ops, pop_size=50, generations=20)
    results = []
    for i, (idx, row) in enumerate(seeds.iterrows()):
        r = ga.run(row["text"], int(idx), verbose=(i < 3))
        od, vd = r["original_detection"], r["variant_detection"]
        init_ev = sum([not od["iso_anomaly"], not od["lof_anomaly"], not od["xgb_anomaly"]])
        fin_ev = sum([not vd["iso_anomaly"], not vd["lof_anomaly"], not vd["xgb_anomaly"]])
        print(f"  [{i+1}/{n_seeds}] {row['name'][:45]}... {init_ev}/3→{fin_ev}/3  sim={r['fitness']['semantic_sim']:.3f}")
        results.append(r)

    improved = sum(1 for r in results
                   if sum([not r["variant_detection"][k] for k in ["iso_anomaly","lof_anomaly","xgb_anomaly"]])
                   > sum([not r["original_detection"][k] for k in ["iso_anomaly","lof_anomaly","xgb_anomaly"]]))
    print(f"\n[3] Improved: {improved}/{n_seeds} ({improved/n_seeds*100:.1f}%)")

    json.dump({"n_seeds": n_seeds, "n_improved": improved, "results": [
        {"idx": r["idx"], "fitness": r["fitness"],
         "orig_det": {k: bool(v) if isinstance(v, (np.bool_, bool)) else float(v) for k, v in r["original_detection"].items()},
         "var_det": {k: bool(v) if isinstance(v, (np.bool_, bool)) else float(v) for k, v in r["variant_detection"].items()}}
        for r in results
    ]}, open(OUT / "adversarial_results.json", "w"), indent=2, default=str)
    print(f"[4] Saved → {OUT}/adversarial_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    main(args.seeds)
