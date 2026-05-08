"""Data loading and text preprocessing."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_skills(path=None):
    """Load skills CSV, build combined text field. Returns DataFrame."""
    path = path or DATA_DIR / "skills_raw_merged.csv"
    if not Path(path).exists():
        path = DATA_DIR / "skills_raw.csv"

    df = pd.read_csv(path)
    df["text"] = (
        df["name"].fillna("")
        + " "
        + df["description"].fillna("")
        + " "
        + df["actions"].fillna("")
        + " "
        + df["permissions"].fillna("")
    )
    return df


def load_labels(path=None):
    """Load labels from Task B output."""
    path = path or Path("output/task_b/merged_rule_labels.csv")
    if not Path(path).exists():
        path = Path("output/task_b/anomaly_results.csv")
    return pd.read_csv(path)
