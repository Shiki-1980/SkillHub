"""
SHAP Explainability Analysis for XGBoost Anomaly Detection.
Explains which features drive malicious / unsafe classification.

Global: summary plot, bar plot, feature clustering
Local: waterfall plots for example skills, dependence plots
"""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, ".")
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = Path("output/explainability")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight"})


def load_and_train():
    """Load data and train the model used for SHAP analysis."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.loader import load_skills
    from src.features.structural_features import extract as extract_structural
    from src.labeling.rule_labeling import label as rule_label
    from sklearn.preprocessing import LabelEncoder

    df = load_skills()
    labels, _ = rule_label(df)

    # Use smaller subset for SHAP (5000 samples — full 10501 is too slow for SHAP)
    idx = np.random.choice(len(df), 5000, replace=False)
    df_sub = df.iloc[idx].copy()
    y_sub = labels[idx]

    # Features
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2),
                            stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    X_tfidf = tfidf.fit_transform(df_sub["text"]).toarray().astype(np.float32)

    struct = extract_structural(df_sub)
    scaler = StandardScaler()
    X_struct = scaler.fit_transform(struct.values.astype(float))

    X = np.hstack([X_tfidf, X_struct])

    # Feature names
    tfidf_names = tfidf.get_feature_names_out().tolist()
    struct_names = struct.columns.tolist()
    feature_names = tfidf_names + struct_names

    # Train XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y_sub)

    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X, y_enc)

    return X, y_enc, feature_names, xgb, le, df_sub, y_sub, struct_names


def main():
    print("=" * 60)
    print("  SHAP Explainability Analysis")
    print("=" * 60)

    # Load & train
    print("\n[1] Training model for SHAP analysis...")
    X, y, feature_names, model, le, df, labels, struct_names = load_and_train()
    n_class = len(le.classes_)
    print(f"  Samples: {len(X)}, Features: {len(feature_names)}")
    print(f"  Classes: {dict(zip(range(n_class), le.classes_))}")

    # Create SHAP explainer (TreeExplainer is fast for XGBoost)
    print("\n[2] Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X[:2000])  # 2000 samples for speed

    # shap_values shape: (n_samples, n_features, n_classes)
    malicious_idx = list(le.classes_).index("malicious")
    unsafe_idx = list(le.classes_).index("unsafe")
    normal_idx = list(le.classes_).index("normal")

    # ── Structural features only (more interpretable) ──
    struct_start = 500  # first 500 are TF-IDF
    X_struct_only = X[:2000, struct_start:]
    struct_shap = shap_values[:, struct_start:, malicious_idx]  # (2000, 45)
    assert struct_shap.shape == X_struct_only.shape, f"{struct_shap.shape} vs {X_struct_only.shape}"

    # ================================================================
    # FIG 1: Structural feature importance (malicious class)
    # ================================================================
    print("\n[3] Generating figures...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        struct_shap, X_struct_only,
        feature_names=struct_names,
        max_display=20, show=False
    )
    ax.set_title("SHAP Feature Importance for 'malicious' Classification\n(Top 20 Structural Features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig1_shap_malicious_importance.png")
    plt.close()
    print("  fig1: malicious feature importance")

    # ────────────────────────────────────────────────────────────────
    # FIG 2: Feature importance bar chart (all classes side by side)
    # ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, (class_idx, class_name) in enumerate([
        (malicious_idx, "malicious"), (unsafe_idx, "unsafe"), (normal_idx, "normal")
    ]):
        vals = np.abs(shap_values[:, struct_start:, class_idx]).mean(0)
        top = np.argsort(vals)[-15:]
        ax = axes[ax_i]
        ax.barh(range(len(top)), vals[top][::-1], color=sns.color_palette("rocket", 15))
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([struct_names[i] for i in top[::-1]], fontsize=7)
        ax.set_title(f"'{class_name}'", fontsize=11, fontweight="bold")
        ax.set_xlabel("mean(|SHAP|)")
    fig.suptitle("Top 15 Structural Features by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_shap_all_classes.png")
    plt.close()
    print("  fig2: all classes feature importance")

    # ────────────────────────────────────────────────────────────────
    # FIG 3: Dependence plots for top 3 features
    # ────────────────────────────────────────────────────────────────
    vals = np.abs(struct_shap).mean(0)
    top3_idx = np.argsort(vals)[-3:]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, feat_idx in enumerate(top3_idx[::-1]):
        ax = axes[i]
        shap.dependence_plot(
            feat_idx + struct_start, shap_values[:, :, malicious_idx], X[:2000],
            feature_names=feature_names, ax=ax, show=False,
            dot_size=12, alpha=0.5
        )
        ax.set_title(struct_names[feat_idx], fontsize=10, fontweight="bold")
    fig.suptitle("SHAP Dependence: Top 3 Features (malicious class)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_dependence_top3.png")
    plt.close()
    print("  fig3: dependence plots")

    # ────────────────────────────────────────────────────────────────
    # FIG 4: Waterfall — explain one malicious skill
    # ────────────────────────────────────────────────────────────────
    mal_samples = np.where(labels == "malicious")[0]
    if len(mal_samples) > 0:
        sample_idx = mal_samples[0]
        if sample_idx >= 2000:
            sample_idx = mal_samples[0] if mal_samples[0] < 2000 else 0

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[:, :, malicious_idx][sample_idx, struct_start:],
                base_values=explainer.expected_value[malicious_idx],
                data=X[sample_idx, struct_start:],
                feature_names=struct_names,
            ),
            max_display=12, show=False
        )
        ax.set_title(
            f"SHAP Waterfall: Why is skill classified as 'malicious'?\n"
            f"Skill: {df.iloc[sample_idx]['name'][:80]}",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(OUT / "fig4_waterfall_malicious.png")
        plt.close()
        print(f"  fig4: waterfall for '{df.iloc[sample_idx]['name'][:60]}'")

    # ────────────────────────────────────────────────────────────────
    # FIG 5: Feature correlation clustering (structural)
    # ────────────────────────────────────────────────────────────────
    struct_importance = np.abs(struct_shap).mean(0)
    top15 = np.argsort(struct_importance)[-15:]
    X_top = pd.DataFrame(X[:2000, struct_start:][:, top15], columns=[struct_names[i] for i in top15])
    corr = X_top.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True,
                fmt=".2f", linewidths=0.5, ax=ax, annot_kws={"fontsize": 7})
    ax.set_title("Correlation of Top 15 Structural Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig5_feature_correlation.png")
    plt.close()
    print("  fig5: feature correlation heatmap")

    # ────────────────────────────────────────────────────────────────
    # FIG 6: DASHBOARD — comprehensive overview
    # ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # (0,0): malicious feature importance
    ax = fig.add_subplot(gs[0, 0])
    vals_m = np.abs(shap_values[:, struct_start:, malicious_idx]).mean(0)
    top10_m = np.argsort(vals_m)[-10:]
    ax.barh(range(10), vals_m[top10_m][::-1], color="#D55E00")
    ax.set_yticks(range(10))
    ax.set_yticklabels([struct_names[i][:25] for i in top10_m[::-1]], fontsize=7)
    ax.set_title("malicious", fontweight="bold", color="#D55E00")
    ax.set_xlabel("mean(|SHAP|)")

    # (0,1): unsafe feature importance
    ax = fig.add_subplot(gs[0, 1])
    vals_u = np.abs(shap_values[:, struct_start:, unsafe_idx]).mean(0)
    top10_u = np.argsort(vals_u)[-10:]
    ax.barh(range(10), vals_u[top10_u][::-1], color="#F0E442")
    ax.set_yticks(range(10))
    ax.set_yticklabels([struct_names[i][:25] for i in top10_u[::-1]], fontsize=7)
    ax.set_title("unsafe", fontweight="bold", color="#CC9900")
    ax.set_xlabel("mean(|SHAP|)")

    # (0,2): normal feature importance
    ax = fig.add_subplot(gs[0, 2])
    vals_n = np.abs(shap_values[:, struct_start:, normal_idx]).mean(0)
    top10_n = np.argsort(vals_n)[-10:]
    ax.barh(range(10), vals_n[top10_n][::-1], color="#009E73")
    ax.set_yticks(range(10))
    ax.set_yticklabels([struct_names[i][:25] for i in top10_n[::-1]], fontsize=7)
    ax.set_title("normal", fontweight="bold", color="#009E73")
    ax.set_xlabel("mean(|SHAP|)")

    # (1,0): Feature overlap across classes (Venn-like bar)
    ax = fig.add_subplot(gs[1, 0])
    all_top = set(top10_m) | set(top10_u) | set(top10_n)
    shared = {(i, "malicious") for i in top10_m} | \
             {(i, "unsafe") for i in top10_u} | \
             {(i, "normal") for i in top10_n}
    feat_shared = {}
    for i in all_top:
        feat_shared[struct_names[i]] = sum([
            i in top10_m, i in top10_u, i in top10_n
        ])
    sorted_feats = sorted(feat_shared.items(), key=lambda x: -x[1])
    colors = ["#2ecc71" if v >= 3 else "#f39c12" if v >= 2 else "#e74c3c"
              for _, v in sorted_feats[:12]]
    ax.barh(range(len(sorted_feats[:12])), [v for _, v in sorted_feats[:12]][::-1],
            color=colors[::-1])
    ax.set_yticks(range(len(sorted_feats[:12])))
    ax.set_yticklabels([f[:25] for f, _ in sorted_feats[:12]][::-1], fontsize=7)
    ax.set_xlabel("Number of classes (out of 3)")
    ax.set_title("Feature Overlap Across Classes", fontweight="bold")

    # (1,1): Feature group contribution
    ax = fig.add_subplot(gs[1, 1])
    groups = {
        "Risk": ["n_danger", "n_warn", "n_safe", "danger_ratio", "total_risks"],
        "Perms": ["perm_os_count", "perm_shell_count", "perm_network_count", "total_perm_items"],
        "Security\nSignals": ["sig_dangerous_exec", "sig_dangerous_net", "sig_sensitive_paths",
                              "sig_disable_safety", "sig_hidden_behavior"],
        "Text\nStats": ["actions_len", "desc_len", "name_len", "desc_ratio"],
        "Meta": ["stars_log", "score", "non_english"],
    }
    group_contrib = {}
    for gname, gfeats in groups.items():
        idxs = [struct_names.index(f) for f in gfeats if f in struct_names]
        if idxs:
            group_contrib[gname] = vals_m[idxs].sum()
    ax.pie(group_contrib.values(), labels=group_contrib.keys(), autopct="%1.1f%%",
           colors=sns.color_palette("Set2", len(group_contrib)), textprops={"fontsize": 9})
    ax.set_title("Feature Group Contribution (malicious)", fontweight="bold")

    # (1,2): TF-IDF vs Structural contribution
    ax = fig.add_subplot(gs[1, 2])
    tfidf_contrib = np.abs(shap_values[:, :, malicious_idx][:, :500]).sum()
    struct_contrib = np.abs(shap_values[:, :, malicious_idx][:, 500:]).sum()
    ax.bar(["TF-IDF (500d)", "Structural (45d)"], [tfidf_contrib, struct_contrib],
           color=["#E69F00", "#0072B2"], edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Total |SHAP| contribution")
    ax.set_title("TF-IDF vs Structural Contribution\n(malicious class)", fontweight="bold")
    for bar, val in zip(ax.patches, [tfidf_contrib, struct_contrib]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                f"{val/(tfidf_contrib+struct_contrib)*100:.0f}%", ha="center", fontweight="bold")

    fig.suptitle("SHAP Explainability Dashboard — Skill Anomaly Detection",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(OUT / "fig6_shap_dashboard.png")
    plt.close()
    print("  fig6: dashboard")

    # ── Summary stats ──
    print(f"\n[4] Top findings:")
    print(f"  TF-IDF contribution: {tfidf_contrib/(tfidf_contrib+struct_contrib)*100:.0f}%")
    print(f"  Structural contribution: {struct_contrib/(tfidf_contrib+struct_contrib)*100:.0f}%")

    # Most important features per class
    for class_name, class_idx in [("malicious", malicious_idx), ("unsafe", unsafe_idx)]:
        vals = np.abs(shap_values[:, struct_start:, class_idx]).mean(0)
        top5 = np.argsort(vals)[-5:][::-1]
        print(f"\n  {class_name} — top 5 structural features:")
        for i in top5:
            print(f"    {struct_names[i]:40} SHAP={vals[i]:.4f}")

    # Feature overlap insight
    shared_3 = sum(1 for v in feat_shared.values() if v >= 3)
    print(f"\n  Features in top 10 of ALL 3 classes: {shared_3}")
    print(f"  Features in top 10 of only 1 class: {sum(1 for v in feat_shared.values() if v == 1)}")

    print(f"\n  All figures saved → {OUT}/")
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".png":
            print(f"    {f.name}")


if __name__ == "__main__":
    main()
