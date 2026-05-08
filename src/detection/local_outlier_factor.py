"""Local Outlier Factor wrapper for anomaly detection."""
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


def detect(X, n_neighbors=30, contamination=0.1):
    """Run LOF with PCA pre-reduction. Returns (model, predictions, scores, svd_reducer)."""
    n_comp = min(80, X.shape[1] // 2, X.shape[0] // 10)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,
                             novelty=False, n_jobs=-1)
    preds = lof.fit_predict(X_red)
    raw = -lof.negative_outlier_factor_
    scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    return lof, preds, scores, svd
