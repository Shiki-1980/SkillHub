"""
Pipeline: Layer 2 LLM Post-Hoc Interpretation.
Takes XGBoost predictions → explains WHY each was flagged → feeds Task D.
Usage: python3 -m src.pipeline.05_layer2_explain [--max 50]
"""
import sys, json, argparse
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
from src.interpretability.layer2_llm_analysis import run_layer2, compute_layer2_stats

OUT = Path("output/layer2")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("  Layer 2: LLM Post-Hoc Interpretation")
    print("=" * 60)

    # Load XGBoost predictions from 80/20 test set
    df = pd.read_csv("output/task_b/test_set_20pct.csv")
    anomalous = df[df["label"].isin(["malicious", "unsafe"])]
    print(f"\n[1] Test set anomalous: {len(anomalous)}")

    # Run Layer 2
    print(f"\n[2] Running LLM analysis on {args.max} samples...")
    results = run_layer2(anomalous, sample_size=args.max)

    # Stats
    stats = compute_layer2_stats(results)
    print(f"\n[3] Agreement with XGBoost:")
    for k, v in stats["agreement"].items():
        print(f"    {k}: {v} ({v/len(results)*100:.0f}%)")

    print(f"\n  Primary risk dimension:")
    for k, v in sorted(stats["primary_risk_dimension"].items(), key=lambda x: -x[1]):
        dim_names = {"A": "Intent Alignment", "B": "Permission Justification",
                     "C": "Covert Behavior", "unknown": "Unknown"}
        print(f"    {dim_names.get(k, k)}: {v} ({v/len(results)*100:.0f}%)")

    print(f"\n  Dimension risk levels:")
    for d, name in [("A", "Intent"), ("B", "Permission"), ("C", "Covert")]:
        dist = stats["dim_risk_distribution"].get(d, {})
        print(f"    {name}: {dist}")

    # Save
    output = {
        "n_analyzed": len(results),
        "stats": stats,
        "results": results,
    }
    with open(OUT / "layer2_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[4] Saved → {OUT}/layer2_results.json")

    # Derisking insights for Task D
    dim_derisking = {
        "A": "d4_action_justification",
        "B": "d2_permission_minimization",
        "C": "d3_risk_transparency",
    }
    print(f"\n[5] Task D guidance:")
    for r in results[:5]:
        dim = r.get("primary_risk_dimension", "")
        suggestion = r.get("derisking_suggestion", "")[:100]
        print(f"  {r['name'][:45]} | dim_{dim} | {suggestion}")

    print(f"\n{'='*60}")
    print("  Layer 2 complete. XGBoost classifies, LLM explains.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
