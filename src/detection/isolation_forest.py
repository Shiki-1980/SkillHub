"""Isolation Forest wrapper for anomaly detection."""
from sklearn.ensemble import IsolationForest
import numpy as np


def detect(X, contamination=0.1, random_state=42):
    """Run Isolation Forest. Returns (model, predictions, normalized_scores)."""
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    preds = iso.fit_predict(X)
    raw = iso.decision_function(X)
    scores = (raw.max() - raw) / (raw.max() - raw.min() + 1e-6)
    return iso, preds, scores
