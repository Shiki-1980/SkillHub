"""
Merge LLM labels back into the pipeline output.
Usage: python3 merge_labels.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

LLM_LABELS_PATH = Path("output/task_b/llm_labels_checkpoint.json")
ANOMALY_RESULTS_PATH = Path("output/task_b/anomaly_results.csv")
OUTPUT_DIR = Path("output/task_b")


def main():
    # Load LLM labels
    with open(LLM_LABELS_PATH) as f:
        llm_results = json.load(f)

    llm_labels = {r["idx"]: r.get("label") for r in llm_results if r.get("label")}
    print(f"Loaded {len(llm_labels)} LLM labels")
    from collections import Counter
    print(f"Distribution: {Counter(llm_labels.values())}")

    # Load full data
    df = pd.read_csv(Path("data/skills_raw.csv"))

    # Create LLM label column (default to rule-based label for unlabeled)
    df["llm_label"] = "unlabeled"
    for idx, label in llm_labels.items():
        if idx < len(df):
            df.loc[idx, "llm_label"] = label

    # Save
    label_path = OUTPUT_DIR / "llm_labels_merged.csv"
    df[["name", "category", "llm_label"]].to_csv(label_path, index=False)
    print(f"Saved → {label_path}")

    # Summary
    labeled = df[df["llm_label"] != "unlabeled"]
    print(f"\nLabeled skills: {len(labeled)}")
    print(labeled["llm_label"].value_counts())


if __name__ == "__main__":
    main()
