"""XGBoost classifier for anomaly detection."""
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


def train(X, y, n_estimators=300, max_depth=6, learning_rate=0.05):
    """Train XGBoost with 5-fold CV. Returns (model, cv_f1_scores, cv_acc_scores, label_encoder)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=learning_rate, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, eval_metric="mlogloss")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(xgb, X, y_enc, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(xgb, X, y_enc, cv=skf, scoring="accuracy")
    xgb.fit(X, y_enc)
    return xgb, cv_f1, cv_acc, le
