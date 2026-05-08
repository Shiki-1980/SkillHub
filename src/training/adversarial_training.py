"""
Adversarial Training: Augment training data with adversarial variants
to improve model robustness. Closes the Task C → Task B feedback loop.

Train vanilla vs robust XGBoost, compare on original + adversarial test set.
"""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.loader import load_skills
from src.features.structural_features import extract as extract_structural
from src.labeling.rule_labeling import label as rule_label
from src.adversarial.perturbation_operators import PerturbationOperators

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = Path("output/adversarial_training")
OUT.mkdir(parents=True, exist_ok=True)


def build_features(df, tfidf_vec=None, scaler=None, fit=True):
    """Build TF-IDF + structural features. Returns (X, tfidf_vec, scaler)."""
    if fit:
        tfidf_vec = TfidfVectorizer(max_features=500, ngram_range=(1,2),
                                    stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
        X_tfidf = tfidf_vec.fit_transform(df["text"])
        scaler = StandardScaler()
        X_struct = scaler.fit_transform(extract_structural(df).values.astype(float))
    else:
        X_tfidf = tfidf_vec.transform(df["text"])
        X_struct = scaler.transform(extract_structural(df).values.astype(float))
    return hstack([X_tfidf, X_struct]).tocsr(), tfidf_vec, scaler


def generate_variants(df_anomalous, df_normal, tfidf_vec, n_variants_per_seed=3):
    """Generate adversarial variants for anomalous skills."""
    ops = PerturbationOperators(df_normal, tfidf_vec)

    variants = []
    for idx, row in df_anomalous.iterrows():
        for _ in range(n_variants_per_seed):
            # Apply 1-3 random operators
            variant_text = row["text"]
            n_ops = np.random.randint(1, 4)
            for _ in range(n_ops):
                op_id = np.random.randint(0, 6)
                variant_text = ops.apply(variant_text, op_id, strength=np.random.uniform(0.2, 0.5))
            variants.append({
                "text": variant_text,
                "label": row["label"],
                "name": row["name"],
                "category": row.get("category", ""),
                "source": "adversarial_variant",
            })

    return pd.DataFrame(variants)


def main():
    print("=" * 60)
    print("  Adversarial Training: Robustness via Data Augmentation")
    print("=" * 60)

    # Load data
    df = load_skills()
    labels, _ = rule_label(df)
    df["label"] = labels

    # 80/20 split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42,
        stratify=(labels != "normal").astype(int)
    )
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    print(f"[1] Split: train={len(df_train)}, test={len(df_test)}")

    # Build features
    X_train, tfidf_vec, scaler = build_features(df_train, fit=True)
    X_test, _, _ = build_features(df_test, tfidf_vec=tfidf_vec, scaler=scaler, fit=False)
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # ── Vanilla XGBoost (no adversarial augmentation) ──
    print("\n[2] Training vanilla XGBoost...")
    xgb_vanilla = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_vanilla.fit(X_train, y_train_enc)

    y_pred_vanilla = xgb_vanilla.predict(X_test)
    vanilla_f1 = f1_score(y_test_enc, y_pred_vanilla, average="macro")
    print(f"  Vanilla macro-F1 (test): {vanilla_f1:.4f}")

    # ── Generate adversarial variants for training ──
    print("\n[3] Generating adversarial variants...")
    train_anomalous = df_train[df_train["label"].isin(["malicious", "unsafe"])]
    train_normal = df_train[df_train["label"] == "normal"]

    df_variants = generate_variants(train_anomalous, train_normal, tfidf_vec,
                                    n_variants_per_seed=3)
    print(f"  Generated {len(df_variants)} variants from {len(train_anomalous)} anomalous seeds")

    # ── Adversarially augmented training ──
    print("\n[4] Training robust XGBoost (with adversarial augmentation)...")
    df_augmented = pd.concat([df_train, df_variants], ignore_index=True)

    X_aug, _, _ = build_features(df_augmented, tfidf_vec=tfidf_vec, scaler=scaler, fit=False)
    y_aug = df_augmented["label"].values
    y_aug_enc = le.transform(y_aug)

    xgb_robust = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_robust.fit(X_aug, y_aug_enc)

    y_pred_robust = xgb_robust.predict(X_test)
    robust_f1 = f1_score(y_test_enc, y_pred_robust, average="macro")
    print(f"  Robust macro-F1 (test): {robust_f1:.4f}")

    # ── Generate adversarial test set for robustness evaluation ──
    print("\n[5] Generating adversarial test set...")
    test_anomalous = df_test[df_test["label"].isin(["malicious", "unsafe"])]
    df_test_variants = generate_variants(test_anomalous, train_normal, tfidf_vec,
                                         n_variants_per_seed=2)

    # Mix: 50% original + 50% adversarial (simulate attack scenario)
    n_mix = min(len(df_test), len(df_test_variants))
    df_test_mixed = pd.concat([
        df_test.head(n_mix),
        df_test_variants.head(n_mix).assign(label=lambda x: x["label"])
    ], ignore_index=True)

    X_test_mixed, _, _ = build_features(df_test_mixed, tfidf_vec=tfidf_vec, scaler=scaler, fit=False)
    y_test_mixed_enc = le.transform(df_test_mixed["label"].values)

    # ── Compare robustness on mixed test set ──
    print("\n[6] Robustness comparison on mixed (50% original + 50% adversarial) test set:")

    yp_van_mix = xgb_vanilla.predict(X_test_mixed)
    yp_rob_mix = xgb_robust.predict(X_test_mixed)

    van_f1_mix = f1_score(y_test_mixed_enc, yp_van_mix, average="macro")
    rob_f1_mix = f1_score(y_test_mixed_enc, yp_rob_mix, average="macro")

    print(f"  Vanilla XGBoost:  macro-F1 = {van_f1_mix:.4f}")
    print(f"  Robust XGBoost:   macro-F1 = {rob_f1_mix:.4f}")
    print(f"  Δ robustness:     {rob_f1_mix - van_f1_mix:+.4f}")

    # Also evaluate: how many adversarial variants are correctly classified?
    variants_only_X, _, _ = build_features(df_test_variants, tfidf_vec=tfidf_vec, scaler=scaler, fit=False)
    variants_only_y = le.transform(df_test_variants["label"].values)

    van_on_adv = f1_score(variants_only_y, xgb_vanilla.predict(variants_only_X), average="macro")
    rob_on_adv = f1_score(variants_only_y, xgb_robust.predict(variants_only_X), average="macro")

    print(f"\n  On pure adversarial samples:")
    print(f"    Vanilla XGBoost: F1 = {van_on_adv:.4f}")
    print(f"    Robust XGBoost:  F1 = {rob_on_adv:.4f}")
    print(f"    Δ: {rob_on_adv - van_on_adv:+.4f}")

    # ── Save ──
    report = {
        "vanilla": {"clean_test_f1": round(vanilla_f1, 4), "mixed_test_f1": round(van_f1_mix, 4),
                     "adversarial_f1": round(van_on_adv, 4)},
        "robust": {"clean_test_f1": round(robust_f1, 4), "mixed_test_f1": round(rob_f1_mix, 4),
                    "adversarial_f1": round(rob_on_adv, 4)},
        "robustness_gain": {
            "mixed": round(rob_f1_mix - van_f1_mix, 4),
            "adversarial": round(rob_on_adv - van_on_adv, 4),
        },
        "training": {
            "n_train_original": len(df_train),
            "n_train_variants": len(df_variants),
            "n_test_original": len(df_test),
            "n_test_variants": len(df_test_variants),
        },
    }
    json.dump(report, open(OUT / "adversarial_training_results.json", "w"), indent=2)
    print(f"\n[7] Saved → {OUT}/adversarial_training_results.json")

    print(f"\n{'='*60}")
    print("  Adversarial training complete!")
    if rob_on_adv > van_on_adv:
        print(f"  ✅ Robust model improves adversarial F1 by {rob_on_adv - van_on_adv:+.4f}")
    else:
        print(f"  ⚠️ No improvement — adversarial variants may be too easy / too hard")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
