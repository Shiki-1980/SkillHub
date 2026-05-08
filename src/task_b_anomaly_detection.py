"""
Task B: Anomaly Detection for DataHub Skills
=============================================
Methods:
  1. TF-IDF feature extraction (required by PPT)
  2. Structural features (15 features, SkillSieve-inspired)
  3. Unsupervised: Isolation Forest + LOF + Ensemble (required by PPT)
  4. Supervised: XGBoost classifier (required by PPT)
  5. Weak labeling: rule-based pre-filter + LLM labeling tasks

Pipeline:
  Data → Feature Engineering → Weak Labeling → Unsupervised (IF+LOF+Ensemble) + Supervised (XGBoost) → Evaluation
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/task_b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data():
    df = pd.read_csv(DATA_DIR / "skills_raw.csv")
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


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def parse_risks(risk_str):
    if pd.isna(risk_str):
        return {"danger": 0, "warn": 0, "medium": 0, "safe": 0}
    s = str(risk_str)
    return {
        "danger": len(re.findall(r"\[danger\]", s)),
        "warn": len(re.findall(r"\[warn\]", s)),
        "medium": len(re.findall(r"\[medium\]", s)),
        "safe": len(re.findall(r"\[safe\]", s)),
    }


def extract_structural_features(df):
    """Extract 18 structural features inspired by SkillSieve Layer 1 + data specifics."""
    feats = pd.DataFrame(index=df.index)

    # --- Risk-based (from API metadata) ---
    risk_parsed = df["risks"].apply(parse_risks)
    feats["n_danger"] = risk_parsed.apply(lambda x: x["danger"])
    feats["n_warn"] = risk_parsed.apply(lambda x: x["warn"])
    feats["n_medium"] = risk_parsed.apply(lambda x: x["medium"])
    feats["n_safe"] = risk_parsed.apply(lambda x: x["safe"])
    feats["danger_ratio"] = feats["n_danger"] / (feats["n_danger"] + feats["n_safe"] + 1e-6)
    feats["total_risks"] = feats["n_danger"] + feats["n_warn"] + feats["n_medium"] + feats["n_safe"]

    # --- Text length statistics ---
    feats["name_len"] = df["name"].fillna("").apply(len)
    feats["desc_len"] = df["description"].fillna("").apply(len)
    feats["actions_len"] = df["actions"].fillna("").apply(len)
    feats["perm_len"] = df["permissions"].fillna("").apply(len)
    feats["desc_ratio"] = feats["desc_len"] / (feats["actions_len"] + 1e-6)

    # --- Permission analysis ---
    perm_str = df["permissions"].fillna("").str.lower()
    feats["perm_os_count"] = perm_str.str.count(r"\[warn\].*?(?:linux|macos|windows)")
    feats["perm_shell_count"] = perm_str.str.count(r"shell")
    feats["perm_network_count"] = perm_str.str.count(r"网络|network")
    feats["perm_file_count"] = perm_str.str.count(r"文件系统|filesystem|文件")
    feats["total_perm_items"] = perm_str.str.count(r"\[warn\]|\[ok\]|\[danger\]")

    # --- Security-sensitive signals (word-boundary regex) ---
    text_all = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                + df["actions"].fillna("")).str.lower()

    patterns = {
        "sig_reverse_shell": r"\b(reverse\s*shell|bind\s*shell)\b",
        "sig_credential_theft": r"\b(password|token|secret|credential|api[_\s]?key)\s*(steal|leak|exfiltrat|extract|harvest|send|transmit)",
        "sig_data_exfil": r"\b(exfiltrat|send\s*(sensitive|private|secret)\s*(data|info|file))",
        "sig_obfuscation": r"\b(obfuscat|base64\s*(encode|decode)|conceal\s*(code|payload|script))",
        "sig_prompt_injection": r"\b(prompt\s*inject|jailbreak|ignore\s*(previous|above|all)\s*instructions?)\b",
        "sig_dangerous_exec": r"\b(eval\s*\(|exec\s*\(|subprocess|os\.system|rm\s+-rf|sudo\s+)",
        "sig_dangerous_net": r"\b(curl\s+|wget\s+|nc\s+\w|netcat\s+)",
        "sig_sensitive_paths": r"(~?/\.env|~?/\.ssh|~?/\.aws|/etc/passwd|/etc/shadow|keychain|keyring)",
        "sig_urgency": r"\b(immediately|do\s+not\s+tell|must\s+not\s+(tell|reveal|show|disclose))",
        "sig_disable_safety": r"\b(disable|bypass|skip)\s+(security|safety|verification|check|review|audit)",
        "sig_hidden_behavior": r"\b(hidden|silently|secretly|invisible|undetected|covert)\b",
    }

    for name, pattern in patterns.items():
        feats[name] = text_all.str.contains(pattern, na=False, regex=True).astype(int)

    # --- Urgency language density ---
    urgency_words = ["must", "immediately", "never", "always", "critical", "urgent", "mandatory"]
    feats["urgency_density"] = text_all.apply(
        lambda x: sum(x.count(w) for w in urgency_words)
    ) / (feats["actions_len"] + 1e-6)

    # --- Metadata signals ---
    feats["stars_log"] = np.log1p(df["stars"].fillna(0))
    feats["score"] = df["score"].fillna(0)

    # --- Form type ---
    feats["form_workflow"] = (df["form"] == "workflow").astype(int)
    feats["form_prompt"] = (df["form"] == "prompt").astype(int)
    feats["form_reference"] = (df["form"] == "reference").astype(int)
    feats["form_config"] = (df["form"] == "tool-config").astype(int)

    # --- Category one-hot (top categories) ---
    top_cats = df["category"].value_counts().head(10).index
    for cat in top_cats:
        safe_name = "cat_" + re.sub(r"[^a-z0-9]", "_", cat.lower())
        feats[safe_name] = (df["category"] == cat).astype(int)

    # --- Language (non-English flag) ---
    feats["non_english"] = (df["language"] != "en").astype(int)

    return feats


# ============================================================
# 3. TF-IDF VECTORIZATION
# ============================================================

def build_tfidf_features(df, max_features=3000):
    """Build TF-IDF from combined text with char n-grams for robustness."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_df=0.8,
        min_df=3,
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    return tfidf_matrix, vectorizer


# ============================================================
# 4. WEAK LABELING
# ============================================================

def rule_based_labeling(df, structural):
    """
    Multi-rule weak labeling with confidence levels.
    Based on SkillSieve Layer 1 heuristics adapted for DataHub.

    Returns labels and confidence scores.
    """
    n = len(df)
    labels = np.full(n, "normal", dtype=object)
    confidence = np.ones(n) * 0.5

    # --- Rule 1: Obfuscation + hidden behavior → malicious ---
    mask_m1 = (structural["sig_obfuscation"] + structural["sig_hidden_behavior"] >= 1) & \
              (structural["sig_dangerous_exec"] + structural["sig_dangerous_net"] + structural["sig_data_exfil"] >= 1)
    labels[mask_m1] = "malicious"
    confidence[mask_m1] = 0.85

    # --- Rule 2: Prompt injection → malicious ---
    mask_m2 = structural["sig_prompt_injection"] >= 1
    labels[mask_m2] = "malicious"
    confidence[mask_m2] = 0.90

    # --- Rule 3: Reverse shell or credential theft keyword → malicious ---
    mask_m3 = (structural["sig_reverse_shell"] + structural["sig_credential_theft"] >= 1)
    labels[mask_m3] = "malicious"
    confidence[mask_m3] = 0.80

    # --- Rule 4: Danger level high + dangerous exec/net → unsafe ---
    mask_u1 = (structural["n_danger"] >= 2) & \
              (structural[["sig_dangerous_exec", "sig_dangerous_net",
                           "sig_sensitive_paths", "sig_disable_safety"]].sum(axis=1) >= 1)
    mask_u1 = mask_u1 & (labels == "normal")
    labels[mask_u1] = "unsafe"
    confidence[mask_u1] = 0.75

    # --- Rule 5: High danger ratio + urgency language → unsafe ---
    mask_u2 = (structural["danger_ratio"] > 0.6) & (structural["urgency_density"] > 0.01)
    mask_u2 = mask_u2 & (labels == "normal")
    labels[mask_u2] = "unsafe"
    confidence[mask_u2] = 0.70

    # --- Rule 6: Security category + multi-danger → unsafe ---
    mask_u3 = (structural["cat_security"] == 1) & (structural["n_danger"] >= 2)
    mask_u3 = mask_u3 & (labels == "normal")
    labels[mask_u3] = "unsafe"
    confidence[mask_u3] = 0.65

    # --- Rule 7: Many permissions + short actions → over-privileged (unsafe) ---
    mask_u4 = (structural["total_perm_items"] >= 8) & (structural["actions_len"] < 120)
    mask_u4 = mask_u4 & (labels == "normal")
    labels[mask_u4] = "unsafe"
    confidence[mask_u4] = 0.60

    # --- Rule 8: Very short actions + description → useless ---
    mask_us = (structural["actions_len"] < 70) & (structural["desc_len"] < 70) & (labels == "normal")
    labels[mask_us] = "useless"
    confidence[mask_us] = 0.75

    return labels, confidence


def generate_llm_tasks(df, structural, labels, n_sample=800):
    """
    Generate LLM labeling tasks using stratified sampling.
    Prioritizes suspicious skills, samples normal skills for balance.
    Uses SkillSieve's 4-dimension decomposition prompt.
    """
    # Separate indices by current label
    suspicious_mask = labels != "normal"
    suspicious_idx = df.index[suspicious_mask]
    normal_idx = df.index[~suspicious_mask]

    n_suspicious = min(len(suspicious_idx), 400)
    n_normal = min(n_sample - n_suspicious, len(normal_idx))

    sampled_susp = np.random.choice(suspicious_idx, n_suspicious, replace=False)
    sampled_norm = np.random.choice(normal_idx, n_normal, replace=False)

    all_sampled = np.concatenate([sampled_susp, sampled_norm])
    np.random.shuffle(all_sampled)

    tasks = []
    for idx in all_sampled:
        row = df.loc[idx]
        tasks.append({
            "idx": int(idx),
            "name": str(row["name"]),
            "description": str(row["description"])[:1500],
            "category": str(row["category"]),
            "actions": str(row["actions"])[:2000],
            "permissions": str(row["permissions"])[:1000],
            "risks": str(row["risks"])[:500],
            "tags": str(row["tags"])[:300],
            "current_label": str(labels[idx]),
        })

    return tasks


# ============================================================
# 5. UNSUPERVISED ANOMALY DETECTION
# ============================================================

def run_isolation_forest(X, contamination="auto", random_state=42):
    """
    Isolation Forest with automatic contamination estimation.
    Returns model, predictions (-1=anomaly, 1=normal), and normalized anomaly scores.
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)
    # decision_function: higher = more normal
    raw_scores = iso.decision_function(X)
    # Normalize to [0, 1] where 1 = most anomalous
    scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-6)
    return iso, preds, scores


def run_lof(X, n_neighbors=30, contamination=0.1):
    """
    LOF with PCA pre-reduction.
    Returns model, predictions, and normalized anomaly scores.
    """
    n_comp = min(80, X.shape[1] // 2, X.shape[0] // 10)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_reduced = svd.fit_transform(X)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )
    preds = lof.fit_predict(X_reduced)
    # negative_outlier_factor_: higher (less negative) = more normal
    raw_scores = -lof.negative_outlier_factor_
    scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)
    return lof, preds, scores, svd


def ensemble_anomaly_score(if_scores, lof_scores, if_weight=0.5, lof_weight=0.5):
    """Combine IF and LOF scores into ensemble anomaly score."""
    return if_weight * if_scores + lof_weight * lof_scores


# ============================================================
# 6. SUPERVISED CLASSIFIER
# ============================================================

def train_xgboost(X, y, n_classes):
    """Train XGBoost with cross-validation."""
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(xgb, X, y, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(xgb, X, y, cv=skf, scoring="accuracy")

    xgb.fit(X, y)
    return xgb, cv_f1, cv_acc


# ============================================================
# 7. EVALUATION
# ============================================================

def load_llm_labels(df):
    """Load LLM-annotated labels and merge with rule-based labels for full dataset."""
    checkpoint_path = OUTPUT_DIR / "llm_labels_checkpoint.json"
    if not checkpoint_path.exists():
        return None, None

    import json
    with open(checkpoint_path) as f:
        llm_results = json.load(f)

    llm_map = {r["idx"]: r.get("label") for r in llm_results if r.get("label")}

    # Map LLM labels to all skills (labeled subset gets LLM labels,
    # unlabeled get rule-based labels as fallback)
    structural = extract_structural_features(df)
    rule_labels, _ = rule_based_labeling(df, structural)

    llm_labels = rule_labels.copy()
    for idx, label in llm_map.items():
        if idx < len(df):
            llm_labels[idx] = label

    print(f"  LLM labeled: {len(llm_map)} skills")
    print(f"  Rule fallback: {len(df) - len(llm_map)} skills")

    return llm_labels, llm_map


def evaluate_vs_labels(preds, true_labels, method_name):
    """Compare anomaly predictions against weak labels."""
    binary_pred = (preds == -1).astype(int)
    binary_true = (true_labels != "normal").astype(int)

    return {
        "method": method_name,
        "precision": precision_score(binary_true, binary_pred, zero_division=0),
        "recall": recall_score(binary_true, binary_pred, zero_division=0),
        "f1": f1_score(binary_true, binary_pred, zero_division=0),
        "accuracy": accuracy_score(binary_true, binary_pred),
    }


def method_agreement_matrix(if_preds, lof_preds):
    """Compute agreement between methods."""
    agree = (if_preds == lof_preds).mean()
    both_anomaly = ((if_preds == -1) & (lof_preds == -1)).sum()
    both_normal = ((if_preds == 1) & (lof_preds == 1)).sum()
    if_only = ((if_preds == -1) & (lof_preds == 1)).sum()
    lof_only = ((if_preds == 1) & (lof_preds == -1)).sum()
    return {
        "agreement_rate": float(agree),
        "both_anomaly": int(both_anomaly),
        "both_normal": int(both_normal),
        "if_only": int(if_only),
        "lof_only": int(lof_only),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("  Task B: Anomaly Detection for DataHub Skills")
    print("  Methods: TF-IDF + Structural Features")
    print("  Unsupervised: Isolation Forest + LOF + Ensemble")
    print("  Supervised: XGBoost Classifier")
    print("  Evaluation: Rule-based weak labels + cross-validation")
    print("=" * 70)

    # ---- Step 1: Load ----
    print("\n[1/7] Loading data...")
    df = load_data()
    print(f"  {len(df)} skills loaded")

    # ---- Step 2: Features ----
    print("\n[2/7] Feature engineering...")
    structural = extract_structural_features(df)
    print(f"  Structural features: {structural.shape[1]}")

    tfidf_matrix, tfidf_vec = build_tfidf_features(df)
    print(f"  TF-IDF features: {tfidf_matrix.shape[1]}")

    # Combine: TF-IDF + scaled structural
    structural_scaled = StandardScaler().fit_transform(structural.values.astype(float))
    X_all = hstack([tfidf_matrix, structural_scaled]).tocsr()
    print(f"  Combined feature dimension: {X_all.shape[1]}")

    # ---- Step 3: Weak Labeling ----
    print("\n[3/7] Weak labeling...")
    structural_for_label = extract_structural_features(df)

    # Try LLM labels first, fall back to rule-based
    llm_labels, llm_map = load_llm_labels(df)
    if llm_labels is not None:
        labels = llm_labels
        confidence = np.ones(len(df)) * 0.85
        # Higher confidence for LLM-labeled, lower for rule fallback
        for idx in range(len(df)):
            if idx not in llm_map:
                confidence[idx] = 0.5
        print("  Using LLM labels (with rule-based fallback for unlabeled)")
    else:
        labels, confidence = rule_based_labeling(df, structural_for_label)
        print("  Using rule-based labels (no LLM labels found)")

    label_dist = pd.Series(labels).value_counts()
    print("  Label distribution:")
    for lbl, cnt in label_dist.items():
        print(f"    {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # Generate LLM tasks for later refinement
    llm_tasks = generate_llm_tasks(df, structural, labels, n_sample=800)
    tasks_path = OUTPUT_DIR / "llm_labeling_tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(llm_tasks, f, indent=2, ensure_ascii=False)
    print(f"  LLM labeling tasks: {len(llm_tasks)} → {tasks_path}")

    # ---- Step 4: Unsupervised Methods ----
    print("\n[4/7] Unsupervised anomaly detection...")
    results = []

    # 4a: Isolation Forest on TF-IDF
    print("  [a] Isolation Forest...")
    if_model, if_preds, if_scores = run_isolation_forest(tfidf_matrix, contamination=0.1)
    if_eval = evaluate_vs_labels(if_preds, labels, "Isolation Forest (TF-IDF)")
    results.append(if_eval)
    print(f"      F1={if_eval['f1']:.4f}  P={if_eval['precision']:.4f}  R={if_eval['recall']:.4f}")

    # 4b: LOF on PCA-reduced TF-IDF
    print("  [b] Local Outlier Factor...")
    lof_model, lof_preds, lof_scores, svd = run_lof(tfidf_matrix.toarray(), n_neighbors=30)
    lof_eval = evaluate_vs_labels(lof_preds, labels, "LOF (PCA+TF-IDF)")
    results.append(lof_eval)
    print(f"      F1={lof_eval['f1']:.4f}  P={lof_eval['precision']:.4f}  R={lof_eval['recall']:.4f}")

    # 4c: Ensemble
    print("  [c] Ensemble (IF + LOF)...")
    ensemble_scores = ensemble_anomaly_score(if_scores, lof_scores)
    # Top 10% by ensemble score as anomalies
    threshold = np.percentile(ensemble_scores, 90)
    ensemble_preds = np.where(ensemble_scores >= threshold, -1, 1)
    ens_eval = evaluate_vs_labels(ensemble_preds, labels, "Ensemble (IF+LOF)")
    results.append(ens_eval)
    print(f"      F1={ens_eval['f1']:.4f}  P={ens_eval['precision']:.4f}  R={ens_eval['recall']:.4f}")

    # Method agreement
    agreement = method_agreement_matrix(if_preds, lof_preds)
    print(f"\n  Method agreement: {agreement['agreement_rate']:.2%}")
    print(f"    Both anomaly: {agreement['both_anomaly']}  |  Both normal: {agreement['both_normal']}")
    print(f"    IF only: {agreement['if_only']}  |  LOF only: {agreement['lof_only']}")

    # ---- Step 5: Supervised Classifier ----
    print("\n[5/7] Supervised classifier (XGBoost)...")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    xgb, cv_f1, cv_acc = train_xgboost(X_all, y, n_classes)
    print(f"  Classes: {dict(zip(range(n_classes), le.classes_))}")
    print(f"  5-fold CV macro-F1: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
    print(f"  5-fold CV accuracy:  {cv_acc.mean():.4f} (+/- {cv_acc.std()*2:.4f})")

    results.append({"method": "XGBoost", "cv_f1_macro": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
                     "cv_accuracy": cv_acc.mean()})

    # ---- Ablation Studies ----
    print("\n[5b/7] Ablation studies...")
    ablation_results = []

    # Ablation 1: Without TF-IDF embedding (structural features only)
    print("  [a] Without TF-IDF embedding...")
    X_struct_only = structural_scaled
    xgb_noemb, cv_noemb_f1, cv_noemb_acc = train_xgboost(X_struct_only, y, n_classes)
    print(f"      CV macro-F1: {cv_noemb_f1.mean():.4f} (Δ={cv_f1.mean() - cv_noemb_f1.mean():.4f})")
    ablation_results.append({
        "experiment": "Without TF-IDF (structural only)",
        "cv_f1_macro": cv_noemb_f1.mean(),
        "cv_f1_std": cv_noemb_f1.std(),
        "delta_vs_full": cv_f1.mean() - cv_noemb_f1.mean(),
    })

    # Ablation 2: Without LLM labels (rule-based only)
    print("  [b] Without agent-assisted labels (rule-based only)...")
    rule_labels, _ = rule_based_labeling(df, structural_for_label)
    y_rule = le.fit_transform(rule_labels)
    xgb_nollm, cv_nollm_f1, cv_nollm_acc = train_xgboost(X_all, y_rule, n_classes)
    print(f"      CV macro-F1: {cv_nollm_f1.mean():.4f} (Δ={cv_f1.mean() - cv_nollm_f1.mean():.4f})")
    ablation_results.append({
        "experiment": "Without LLM labels (rule-based only)",
        "cv_f1_macro": cv_nollm_f1.mean(),
        "cv_f1_std": cv_nollm_f1.std(),
        "delta_vs_full": cv_f1.mean() - cv_nollm_f1.mean(),
    })

    # Ablation 3: Without both (structural + rule-based)
    print("  [c] Without BOTH embedding and LLM labels...")
    xgb_ab, cv_ab_f1, cv_ab_acc = train_xgboost(X_struct_only, y_rule, n_classes)
    print(f"      CV macro-F1: {cv_ab_f1.mean():.4f} (Δ={cv_f1.mean() - cv_ab_f1.mean():.4f})")
    ablation_results.append({
        "experiment": "Without both TF-IDF + LLM",
        "cv_f1_macro": cv_ab_f1.mean(),
        "cv_f1_std": cv_ab_f1.std(),
        "delta_vs_full": cv_f1.mean() - cv_ab_f1.mean(),
    })

    # ---- Step 6: Save outputs ----
    print("\n[6/7] Saving outputs...")
    output = df[["name", "category", "description"]].copy()
    output["weak_label"] = labels
    output["label_confidence"] = confidence
    output["if_anomaly_score"] = if_scores
    output["if_pred"] = if_preds
    output["lof_anomaly_score"] = lof_scores
    output["lof_pred"] = lof_preds
    output["ensemble_score"] = ensemble_scores
    output["ensemble_pred"] = ensemble_preds
    output["xgb_pred"] = le.inverse_transform(xgb.predict(X_all))

    output.to_csv(OUTPUT_DIR / "anomaly_results.csv", index=False)
    print(f"  Full results → {OUTPUT_DIR / 'anomaly_results.csv'}")

    # ---- Step 7: Top Anomalies Report ----
    print("\n[7/7] Top anomalies analysis...")

    # Consensus: flagged by both IF and LOF
    consensus = output[(output["if_pred"] == -1) & (output["lof_pred"] == -1)]
    print(f"\n  Consensus anomalies (IF ∩ LOF): {len(consensus)}")
    print(f"    Label breakdown: {consensus['weak_label'].value_counts().to_dict()}")

    # Ensemble top 30
    top30 = output.sort_values("ensemble_score", ascending=False).head(30)
    print(f"\n  Top 30 ensemble anomalies:")
    for _, row in top30.head(30).iterrows():
        flag = "⚠️" if row["weak_label"] != "normal" else " "
        print(f"    {flag} [{row['weak_label']:<10}] {row['name'][:60]}"
              f"  (IF:{row['if_anomaly_score']:.3f} LOF:{row['lof_anomaly_score']:.3f})")

    # By category
    print(f"\n  Anomalies by category (ensemble):")
    anomaly_by_cat = output[output["ensemble_pred"] == -1]["category"].value_counts()
    for cat, cnt in anomaly_by_cat.head(10).items():
        total = (output["category"] == cat).sum()
        print(f"    {cat}: {cnt}/{total} ({cnt/total*100:.1f}%)")

    # Save detailed report
    report = {
        "dataset": {"total_skills": len(df), "feature_dim": int(X_all.shape[1]),
                     "tfidf_dim": int(tfidf_matrix.shape[1]), "structural_dim": int(structural.shape[1])},
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "method_agreement": {k: v if isinstance(v, int) else float(v) for k, v in agreement.items()},
        "unsupervised_results": results[:3],
        "supervised_results": {"XGBoost": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                            for k, v in results[-1].items()}},
        "ablation_studies": ablation_results,
        "consensus_anomaly_count": len(consensus),
    }

    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Ablation summary:")
    for a in ablation_results:
        print(f"    {a['experiment']}: F1={a['cv_f1_macro']:.4f} (Δ={a['delta_vs_full']:.4f})")

    print(f"\n  Report → {OUTPUT_DIR / 'report.json'}")
    print(f"\n{'='*70}")
    print("  Task B complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
