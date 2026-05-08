"""
Publication-quality embedding visualizations: TF-IDF vs SBERT.
Multi-panel comparison, colorblind-friendly, statistical annotations.
"""
import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

OUT = Path("output/visualization")
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Wong 2011)
C = {
    "normal": "#009E73",     # bluish green
    "unsafe": "#F0E442",     # yellow
    "malicious": "#D55E00", # vermillion
    "useless": "#CC79A7",   # reddish purple
    "sbert": "#0072B2",     # blue
    "tfidf": "#E69F00",     # orange
}

# ── Helpers ──

def scatter_panel(ax, X, labels, title, legend=True):
    """Single clean scatter panel."""
    order = ["normal", "unsafe", "malicious", "useless"]
    for lbl in order:
        mask = np.array(labels) == lbl
        if mask.sum() == 0: continue
        ax.scatter(X[mask, 0], X[mask, 1], c=C[lbl], label=f"{lbl} ({mask.sum()})",
                   alpha=0.55, s=3, edgecolors="none", rasterized=True)
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(loc="upper right", framealpha=0.9, markerscale=4,
                  handletextpad=0.3, borderpad=0.3, fontsize=8)


def save(name):
    plt.savefig(OUT / name, dpi=300, facecolor="white", edgecolor="none")


# ── Main ──

def main():
    print("Loading data...")
    df = pd.read_csv("data/skills_raw_merged.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))
    labels = pd.read_csv("output/task_b/merged_rule_labels.csv")["rule_label"].values
    cats = df["category"].values

    # Feature matrices
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words="english",
                            sublinear_tf=True, max_df=0.8, min_df=3)
    X_tf = StandardScaler().fit_transform(tfidf.fit_transform(df["text"]).toarray().astype(np.float32))
    X_sb = np.load("output/task_b/sbert_embeddings_full.npy")

    # Subsample
    n = min(5000, len(df))
    ix = np.random.RandomState(42).choice(len(df), n, replace=False)
    print(f"Subsampled {n} points")

    # Compute projections once
    print("t-SNE...")
    tsne = TSNE(2, perplexity=30, random_state=42, n_jobs=-1)
    T_tf = tsne.fit_transform(X_tf[ix])
    T_sb = tsne.fit_transform(X_sb[ix])

    try:
        import umap
        print("UMAP...")
        u = umap.UMAP(2, random_state=42, verbose=False)
        U_tf = u.fit_transform(X_tf[ix])
        U_sb = u.fit_transform(X_sb[ix])
        has_umap = True
    except ImportError:
        has_umap = False

    # ═══════════════════════════════════════════
    # FIGURE 1: Side-by-side t-SNE (labels)
    # ═══════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    scatter_panel(ax1, T_tf, labels[ix], "TF-IDF (500d)")
    scatter_panel(ax2, T_sb, labels[ix], "SBERT (384d)", legend=False)
    # Shared legend
    handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=C[l], markersize=8, label=l)
               for l in ["normal", "unsafe", "malicious", "useless"]]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("t-SNE Projection of Skill Embeddings by Anomaly Label",
                 fontsize=16, fontweight="bold", y=1.01)
    save("fig1_tsne_label_comparison.png")
    plt.close()
    print("  fig1 done")

    # ═══════════════════════════════════════════
    # FIGURE 2: Side-by-side t-SNE (categories)
    # ═══════════════════════════════════════════
    top8 = pd.Series(cats[ix]).value_counts().head(8).index
    cat_colors = dict(zip(top8, sns.color_palette("husl", 8)))
    cat_disp = np.array([c if c in top8 else "other" for c in cats[ix]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for lbl in list(top8) + ["other"]:
        m = cat_disp == lbl
        c = cat_colors.get(lbl, "#cccccc")
        a = 0.3 if lbl == "other" else 0.6
        ax1.scatter(T_tf[m,0], T_tf[m,1], c=[c], label=lbl, alpha=a, s=3, edgecolors="none", rasterized=True)
        ax2.scatter(T_sb[m,0], T_sb[m,1], c=[c], label=lbl, alpha=a, s=3, edgecolors="none", rasterized=True)

    ax1.set_title("TF-IDF (500d)", fontweight="bold")
    ax2.set_title("SBERT (384d)", fontweight="bold")
    for ax in [ax1, ax2]: ax.set_xticks([]); ax.set_yticks([])
    fig.legend(loc='lower center', ncol=5, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("t-SNE Projection by Skill Category", fontsize=16, fontweight="bold", y=1.01)
    save("fig2_tsne_category_comparison.png")
    plt.close()
    print("  fig2 done")

    # ═══════════════════════════════════════════
    # FIGURE 3: UMAP comparison (labels)
    # ═══════════════════════════════════════════
    if has_umap:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        scatter_panel(ax1, U_tf, labels[ix], "TF-IDF (500d)")
        scatter_panel(ax2, U_sb, labels[ix], "SBERT (384d)", legend=False)
        handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=C[l], markersize=8, label=l)
                   for l in ["normal", "unsafe", "malicious", "useless"]]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=True,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("UMAP Projection of Skill Embeddings by Anomaly Label",
                     fontsize=16, fontweight="bold", y=1.01)
        save("fig3_umap_label_comparison.png")
        plt.close()
        print("  fig3 done")

    # ═══════════════════════════════════════════
    # FIGURE 4: Density + Anomaly concentration
    # ═══════════════════════════════════════════
    from sklearn.neighbors import NearestNeighbors
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    anomaly = labels[ix] != "normal"

    for row, (X, name) in enumerate([(T_tf, "TF-IDF"), (T_sb, "SBERT")]):
        nn = NearestNeighbors(n_neighbors=50).fit(X)
        d = nn.kneighbors(X)[0].mean(1)

        # Density
        ax = axes[row, 0]
        sc = ax.scatter(X[:,0], X[:,1], c=d, cmap="viridis", alpha=0.6, s=4,
                        edgecolors="none", rasterized=True)
        ax.set_title(f"{name}: Density Map", fontweight="bold")
        plt.colorbar(sc, ax=ax, shrink=0.75, label="Mean distance to 50-NN")
        ax.set_xticks([]); ax.set_yticks([])

        # Anomaly overlay
        ax = axes[row, 1]
        ax.scatter(X[~anomaly,0], X[~anomaly,1], c=C["normal"], alpha=0.25, s=3,
                   edgecolors="none", label="normal", rasterized=True)
        ax.scatter(X[anomaly,0], X[anomaly,1], c=C["malicious"], alpha=0.75, s=8,
                   edgecolors="none", label="malicious/unsafe", rasterized=True)

        # Stats
        sp = d > np.percentile(d, 90)
        de = d < np.percentile(d, 10)
        ax.text(0.02, 0.98, f"Sparse region anom.: {anomaly[sp].mean():.1%}\n"
                             f"Dense region anom.:  {anomaly[de].mean():.1%}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_title(f"{name}: Anomaly Overlay", fontweight="bold")
        ax.legend(markerscale=4, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Embedding Density vs Anomaly Distribution", fontsize=15, fontweight="bold")
    save("fig4_density_analysis.png")
    plt.close()
    print("  fig4 done")

    # ═══════════════════════════════════════════
    # FIGURE 5: Dashboard summary
    # ═══════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)

    # (0,0): TF-IDF t-SNE
    ax = fig.add_subplot(gs[0, 0])
    scatter_panel(ax, T_tf, labels[ix], "TF-IDF · t-SNE")

    # (0,1): SBERT t-SNE
    ax = fig.add_subplot(gs[0, 1])
    scatter_panel(ax, T_sb, labels[ix], "SBERT · t-SNE")

    # (0,2): Label distribution bar chart
    ax = fig.add_subplot(gs[0, 2])
    lbl_counts = pd.Series(labels).value_counts()
    bars = ax.bar(range(len(lbl_counts)), lbl_counts.values,
                  color=[C.get(l, "#999") for l in lbl_counts.index])
    ax.set_xticks(range(len(lbl_counts)))
    ax.set_xticklabels(lbl_counts.index, rotation=30, ha="right")
    ax.set_ylabel("Count"); ax.set_title("Label Distribution", fontweight="bold")
    for bar, v in zip(bars, lbl_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, str(v),
                ha="center", fontsize=9)
    ax.set_yscale("log")

    # (1,0): Category distribution
    ax = fig.add_subplot(gs[1, 0])
    top_cats = pd.Series(cats).value_counts().head(10)
    ax.barh(range(len(top_cats)), top_cats.values[::-1],
            color=sns.color_palette("viridis", 10))
    ax.set_yticks(range(len(top_cats)))
    ax.set_yticklabels(top_cats.index[::-1])
    ax.set_xlabel("Count"); ax.set_title("Top 10 Categories", fontweight="bold")

    # (1,1): Detection method comparison
    ax = fig.add_subplot(gs[1, 1])
    methods = ["IF (TF-IDF)", "LOF (TF-IDF)", "LSTM AE", "XGBoost"]
    f1_scores = [0.0676, 0.1032, 0.0720, 0.8293]
    colors = [C["tfidf"], C["tfidf"], C["sbert"], C["normal"]]
    bars = ax.bar(methods, f1_scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("F1 Score"); ax.set_title("Detection Method F1 Scores", fontweight="bold")
    for bar, v in zip(bars, f1_scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.95)

    # (1,2): SBERT vs TF-IDF head-to-head
    ax = fig.add_subplot(gs[1, 2])
    comparisons = ["IF (+11%)", "LOF (+24%)", "XGBoost (+8%)"]
    tfidf_vals = [0.0641, 0.1192, 0.4429]
    sbert_vals = [0.0712, 0.1477, 0.4779]
    x = np.arange(len(comparisons))
    w = 0.3
    ax.bar(x-w/2, tfidf_vals, w, label="TF-IDF", color=C["tfidf"], edgecolor="white")
    ax.bar(x+w/2, sbert_vals, w, label="SBERT", color=C["sbert"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(comparisons)
    ax.set_ylabel("F1 Score"); ax.set_title("SBERT vs TF-IDF (feature-only)", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("SkillHub Anomaly Detection — Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)
    save("fig5_dashboard.png")
    plt.close()
    print("  fig5 done")

    # ═══════════════════════════════════════════
    # FIGURE 6: Anomaly heatmap by category × method
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_anom = {}
    for cat in top8:
        mask = cats == cat
        if mask.sum() > 0:
            cat_anom[cat] = (labels[mask] != "normal").mean()
    sorted_cats = sorted(cat_anom.items(), key=lambda x: -x[1])
    ax.barh([c for c, _ in sorted_cats], [v*100 for _, v in sorted_cats],
            color=sns.color_palette("Reds", len(sorted_cats)))
    ax.set_xlabel("Anomaly Rate (%)"); ax.set_title("Anomaly Rate by Category", fontweight="bold")
    for i, (cat, v) in enumerate(sorted_cats):
        ax.text(v*100+0.3, i, f"{v*100:.1f}%", va="center", fontsize=9)
    save("fig6_anomaly_by_category.png")
    plt.close()
    print("  fig6 done")

    print(f"\nAll 6 figures saved to {OUT}/")
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
