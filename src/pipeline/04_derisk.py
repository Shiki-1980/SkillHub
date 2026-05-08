"""
Pipeline: De-risking (Task D).
Usage: python3 -m src.pipeline.04_derisk
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.loader import load_skills
from src.derisking.kri_scorer import KRIRiskScorer
from src.derisking.derisking_operators import DeRiskingOperators

OUT = Path("output/task_d")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Pipeline: De-risking (Task D)")
    print("=" * 50)

    df = load_skills()
    labels = pd.read_csv("output/task_b/merged_rule_labels.csv")["rule_label"].values
    anomalous = df[labels.isin(["malicious", "unsafe"])]
    n_seeds = min(100, len(anomalous))
    seeds = anomalous.sample(n_seeds, random_state=42)

    print(f"[1] {n_seeds} seeds")

    scorer = KRIRiskScorer(w_threat=0.55, w_impact=0.25, w_exposure=0.20)
    derisker = DeRiskingOperators()
    tfidf = TfidfVectorizer(max_features=500, stop_words="english").fit(df["text"])

    results = []
    for i, (idx, row) in enumerate(seeds.iterrows()):
        ot, ors, opm = row["text"], str(row.get("risks","")), str(row.get("permissions",""))
        orig_kri = scorer.compute(ot, ors, opm)
        dt = derisker.apply_all(ot)
        derisked_kri = scorer.compute(dt, ors, opm)
        sim = float(cosine_similarity(tfidf.transform([ot]), tfidf.transform([dt]))[0, 0])
        results.append({
            "idx": int(idx), "name": str(row["name"]),
            "orig_kri": orig_kri["kri"], "derisked_kri": derisked_kri["kri"],
            "kri_delta": round(orig_kri["kri"] - derisked_kri["kri"], 4),
            "semantic_similarity": round(sim, 4),
        })
        if i < 3:
            print(f"  [{i+1}] {row['name'][:45]}... KRI: {orig_kri['kri']:.3f}→{derisked_kri['kri']:.3f} (Δ={orig_kri['kri']-derisked_kri['kri']:.3f}) sim={sim:.3f}")

    reduced = sum(1 for r in results if r["kri_delta"] > 0)
    print(f"\n[2] KRI reduced: {reduced}/{n_seeds} ({reduced/n_seeds*100:.1f}%)")
    print(f"    Avg KRI delta: {np.mean([r['kri_delta'] for r in results]):.4f}")
    print(f"    Avg semantic sim: {np.mean([r['semantic_similarity'] for r in results]):.4f}")

    json.dump({"n_seeds": n_seeds, "kri_reduced_rate": f"{reduced}/{n_seeds}", "results": results},
              open(OUT / "derisking_results.json", "w"), indent=2, default=str)
    print(f"[3] Saved → {OUT}/derisking_results.json")


if __name__ == "__main__":
    main()
