"""Evaluation metrics for anomaly detection."""
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def evaluate(predictions, true_labels):
    """Compare predictions against true labels. Returns dict of metrics."""
    bp = (np.array(predictions) == -1).astype(int)
    bt = (np.array(true_labels) != "normal").astype(int)
    return {
        "f1": round(f1_score(bt, bp, zero_division=0), 4),
        "precision": round(precision_score(bt, bp, zero_division=0), 4),
        "recall": round(recall_score(bt, bp, zero_division=0), 4),
        "accuracy": round(accuracy_score(bt, bp), 4),
    }


def detection_summary(if_preds, lof_preds, lstm_preds):
    """Compute agreement matrix between methods."""
    a = np.array
    return {
        "IF ∩ LOF": int(((a(if_preds) == -1) & (a(lof_preds) == -1)).sum()),
        "IF ∩ LSTM": int(((a(if_preds) == -1) & (a(lstm_preds) == -1)).sum()),
        "LOF ∩ LSTM": int(((a(lof_preds) == -1) & (a(lstm_preds) == -1)).sum()),
        "All three": int(((a(if_preds) == -1) & (a(lof_preds) == -1) & (a(lstm_preds) == -1)).sum()),
    }
