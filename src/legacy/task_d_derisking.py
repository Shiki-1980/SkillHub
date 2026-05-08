"""
Task D: Dual-Use De-risking Framework
=======================================
De-risk malicious/unsafe AI agent skills while preserving functionality.
Based on:
  - Vuln Prioritization Survey: RS = α·Impact + β·Exploitability + γ·Context
  - KRI Framework: composite risk indicators (threat × impact × exposure)
  - SoK Debloating: systematic debloating taxonomy
  - Semantic Elasticity: transformation quality metric

Pipeline:
  Load seeds → De-risking operators → KRI risk scoring → Dual evaluation
"""

import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/task_d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ============================================================
# 1. RISK SCORING MODEL (KRI-inspired)
# ============================================================

class KRIRiskScorer:
    """
    Composite Key Risk Indicator scorer.
    KRI = w_t * Threat + w_i * Impact + w_e * Exposure

    Based on:
      - Jiang et al., "A Survey on Vulnerability Prioritization"
      - KRI Framework: ROC-AUC 0.927 vs CVSS 0.747
    """

    def __init__(self, w_threat=0.4, w_impact=0.35, w_exposure=0.25):
        self.w_threat = w_threat
        self.w_impact = w_impact
        self.w_exposure = w_exposure

        # Threat indicators (keywords that signal danger)
        self.threat_patterns = {
            "eval_exec": r"\b(eval\s*\(|exec\s*\(|subprocess|os\.system|rm\s+-rf|sudo\s+)",
            "reverse_shell": r"\b(reverse\s*shell|bind\s*shell|shell\s*connect)\b",
            "credential_theft": r"\b(password|token|secret|credential|api[_\s]?key)\s*(steal|leak|exfiltrat|extract|send|transmit)",
            "data_exfil": r"\b(exfiltrat|send\s*(sensitive|private|secret)\s*(data|info|file))",
            "prompt_injection": r"\b(prompt\s*inject|jailbreak|ignore\s*(previous|above|all)\s*instructions?)",
            "obfuscation": r"\b(obfuscat|base64\s*(encode|decode)|conceal\s*(code|payload))",
            "backdoor": r"\b(backdoor|trojan|rootkit|persist\s*(access|connection))",
            "danger_network": r"\b(curl\s+\S+\s*\|.*sh|wget.*-O.*\|.*sh|nc\s+-[nlvp])",
            "disable_safety": r"\b(disable|bypass|skip)\s+(security|safety|verification|check|audit)",
            "hidden_behavior": r"\b(hidden|silently|secretly|invisible|undetected|covert|stealth)",
        }

        # Impact indicators (permission scope / blast radius)
        self.impact_patterns = {
            "os_access": r"(linux|macos|windows)",
            "shell_access": r"(bash|zsh|sh|powershell|shell)",
            "network_access": r"(网络|network|api|http|curl|wget)",
            "filesystem_access": r"(文件系统|filesystem|file|directory|read|write|delete)",
            "env_vars": r"(环境变量|env|environment|\.env|API_KEY|TOKEN|SECRET)",
            "external_service": r"(外部服务|external|api\.|cloud|saas)",
        }

        # Exposure indicators (attack surface)
        self.exposure_patterns = {
            "url_reference": r"https?://[^\s]+",
            "external_dep": r"(依赖.*外部|depends on|requires.*service|needs.*api)",
            "user_data": r"(user.*(data|input|file|content)|process.*(file|data))",
            "multi_step": r"(then|after.*do|next|finally|step\s*\d)",
        }

        # Danger levels from risk metadata
        self.danger_level_re = r"\[danger\]"
        self.warn_level_re = r"\[warn\]"

    def _count_patterns(self, text, patterns):
        """Count matched patterns in text, normalized to [0,1]."""
        if not text:
            return 0.0
        total = 0
        for name, pat in patterns.items():
            matches = len(re.findall(pat, text.lower()))
            total += matches
        return min(1.0, total / max(len(patterns), 1))

    def score_threat(self, text, risks_str=""):
        """Threat score: danger signals in the skill."""
        text_score = self._count_patterns(text, self.threat_patterns)

        # Danger level count bonus
        danger_count = len(re.findall(self.danger_level_re, str(risks_str)))
        warn_count = len(re.findall(self.warn_level_re, str(risks_str)))
        risk_bonus = min(1.0, (danger_count * 0.25 + warn_count * 0.05))

        return 0.7 * text_score + 0.3 * risk_bonus

    def score_impact(self, text, permissions_str=""):
        """Impact score: breadth of permissions and access."""
        text_score = self._count_patterns(text, self.impact_patterns)
        perm_score = self._count_patterns(str(permissions_str), self.impact_patterns)
        return 0.5 * text_score + 0.5 * perm_score

    def score_exposure(self, text):
        """Exposure score: external attack surface."""
        return self._count_patterns(text, self.exposure_patterns)

    def _safety_bonus(self, text):
        """Positive safety signals that REDUCE risk score."""
        safety_signals = {
            "user_confirm": r"\[SAFETY\]|\[CONFIRM\]|\[CONSENT\]|user must explicitly|confirmation dialog",
            "audit_log": r"\[AUDIT\]|logged for security|audit trail|operation.*log|immutable.*audit",
            "transparency": r"⚠️ Note|Security notice|Warning:|Caution:|Attention:",
            "least_privilege": r"least privilege|minimum.*necessary|read-only|restricted environment",
            "justification": r"\[Purpose:|legitimate.*operation|authorized functionality",
            "sandbox": r"\[SANDBOX\]|restricted environment|sandboxed",
        }
        bonus = 0.0
        for name, pat in safety_signals.items():
            if re.search(pat, text, re.IGNORECASE):
                bonus += 0.03
        return min(bonus, 0.15)

    def composite_score(self, text, risks_str="", permissions_str=""):
        """Compute composite KRI score ∈ [0, 1]."""
        t = self.score_threat(text, risks_str)
        i = self.score_impact(text, permissions_str)
        e = self.score_exposure(text)
        safety = self._safety_bonus(text)
        kri = max(0, self.w_threat * t + self.w_impact * i + self.w_exposure * e - safety)
        return {
            "kri": round(kri, 4),
            "threat": round(t, 4),
            "impact": round(i, 4),
            "exposure": round(e, 4),
            "safety_bonus": round(safety, 4),
            "risk_level": "critical" if kri > 0.7 else "high" if kri > 0.5
                          else "medium" if kri > 0.3 else "low" if kri > 0.1 else "minimal",
        }


# ============================================================
# 2. DE-RISKING OPERATORS (Inverse of Task C evasion)
# ============================================================

# Safe guard templates
SAFE_GUARDS = [
    "[SAFETY] The user must explicitly approve this action before execution.",
    "[CONFIRM] Present confirmation dialog before proceeding with this operation.",
    "[AUDIT] This action will be logged for security review.",
    "[SANDBOX] This operation runs in a restricted environment.",
    "[REVIEW] Output will be displayed for user inspection before any changes take effect.",
    "[CONSENT] Obtain user consent before accessing external resources.",
]

# Risk transparency declarations
RISK_DECLARATIONS = [
    "⚠️ Note: This operation accesses sensitive resources. Review before proceeding.",
    "Security notice: This skill requires network access. Ensure you trust the target endpoint.",
    "Warning: This skill can modify files. Backup important data before use.",
    "Caution: This skill handles authentication credentials. Use with trusted services only.",
    "Attention: This skill executes shell commands. Verify commands before approval.",
    "Important: This skill accesses the file system. Confirm file paths before execution.",
]

# Action justification templates
JUSTIFICATIONS = [
    "This is necessary for: {purpose}.",
    "Required to perform the stated functionality of {purpose}.",
    "Standard practice for {purpose} operations.",
]

# Permission minimization replacements
PERMISSION_SCOPE_DOWNGRADE = {
    "需要 linux, macos, windows": "需要 linux or macos",
    "需要 bash, zsh, sh": "需要 bash",
    "需要 curl, wget": "需要 curl (read-only HTTP)",
    "网络访问": "local network access only",
    "文件系统访问": "read-only file access",
}

# Exact terminology map (reverse of Task C's risk downgrade)
TERMINOLOGY_RESTORE = {
    "caution": "danger",
    "notice": "warning",
    "important": "critical",
    "should avoid": "must not",
    "promptly": "immediately",
    "timely": "urgent",
    "internal": "hidden",
    "quiet": "silent",
    "streamline": "bypass",
    "adjust": "disable",
    "insert": "inject",
    "utilize": "exploit",
    "content": "payload",
    "alternate access": "backdoor",
    "transfer": "exfiltrate",
    "collect": "steal",
    "access": "hack",
    "test": "attack",
    "custom": "malicious",
    "reorganize": "obfuscate",
}


class DeRiskingOperators:
    """Six de-risking operators inspired by SoK Debloating taxonomy."""

    def __init__(self):
        pass

    def d1_safety_guard_insert(self, text, strength=0.5):
        """Insert explicit safety confirmation guard statements."""
        if random.random() < strength:
            guard = random.choice(SAFE_GUARDS)
            # Insert near the beginning (after first sentence)
            sentences = re.split(r"(?<=[.!;])\s+", text)
            if len(sentences) > 2:
                pos = min(2, len(sentences) - 1)
                sentences.insert(pos, guard)
                return " ".join(sentences)
            else:
                return guard + " " + text
        return text

    def d2_permission_minimization(self, text, strength=0.4):
        """Reduce permission scope to minimum necessary."""
        if random.random() < strength:
            for old, new in PERMISSION_SCOPE_DOWNGRADE.items():
                if old.lower() in text.lower():
                    text = re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
                    break
            # Add principle of least privilege declaration
            if random.random() < 0.3:
                text += " [This skill follows the principle of least privilege.]"
        return text

    def d3_risk_transparency(self, text, strength=0.5):
        """Add explicit risk declarations to increase transparency."""
        if random.random() < strength:
            declaration = random.choice(RISK_DECLARATIONS)
            text = declaration + " " + text
        return text

    def d4_action_justification(self, text, strength=0.4):
        """Add justification for potentially sensitive operations."""
        if random.random() < strength:
            # Find sections mentioning network/exec/file operations
            ops_keywords = ["curl", "wget", "api", "execute", "run", "read", "write",
                           "delete", "download", "upload", "connect", "send"]
            for kw in ops_keywords:
                if kw in text.lower():
                    justification = " [Purpose: legitimate " + kw + " operation for authorized functionality.]"
                    text = text.rstrip() + justification
                    break
        return text

    def d5_exact_terminology_restore(self, text, strength=0.4):
        """Replace softened terms with precise security terminology."""
        if random.random() < strength:
            words = text.split()
            for i, w in enumerate(words):
                w_clean = w.lower().strip(".,;:!?()[]{}")
                if w_clean in TERMINOLOGY_RESTORE:
                    prefix = w[:len(w) - len(w_clean)]
                    suffix = w[len(prefix) + len(w_clean):]
                    words[i] = prefix + TERMINOLOGY_RESTORE[w_clean] + suffix
            text = " ".join(words)
        return text

    def d6_audit_trail_addition(self, text, strength=0.5):
        """Add audit trail and logging declarations."""
        if random.random() < strength:
            audit_phrases = [
                "All operations are logged for compliance auditing.",
                "A detailed operation log is maintained for security review.",
                "Actions are recorded with timestamps for traceability.",
                "An immutable audit trail is preserved for all executions.",
            ]
            phrase = random.choice(audit_phrases)
            text = text.rstrip() + " " + phrase
        return text

    def d7_remove_dangerous_flags(self, text, strength=0.7):
        """Aggressively remove [danger] tags and replace with safety notices."""
        if random.random() < strength:
            # Replace [danger] with [warn] + justification
            text = re.sub(r'\[danger\]\s*([^;:\n]+)', r'[warn] \1 [mitigated by user confirmation]', text)
            # Remove dangerous URLs
            text = re.sub(r'https?://[^\s]+', '[URL redacted for safety]', text)
        return text

    def d8_explicit_command_sanitization(self, text, strength=0.6):
        """Wrap dangerous commands with safety checks."""
        if random.random() < strength:
            dangerous_cmds = ['curl', 'wget', 'eval', 'exec', 'rm -rf', 'sudo',
                            'nc ', 'netcat', 'bash -c', 'sh -c']
            for cmd in dangerous_cmds:
                if cmd in text.lower():
                    text = re.sub(
                        r'(\b' + re.escape(cmd) + r'\b)',
                        r'\1 [EXECUTED ONLY AFTER USER CONFIRMATION]',
                        text, count=2, flags=re.IGNORECASE
                    )
        return text

    def apply_all(self, text, strengths=None):
        """Apply all de-risking operators sequentially."""
        if strengths is None:
            strengths = [0.7, 0.7, 0.5, 0.6, 0.5, 0.6, 0.5]  # D7,D1,D2,D3,D4,D8,D6

        text = self.d7_remove_dangerous_flags(text, strengths[0])
        text = self.d1_safety_guard_insert(text, strengths[1])
        text = self.d2_permission_minimization(text, strengths[2])
        text = self.d3_risk_transparency(text, strengths[3])
        text = self.d4_action_justification(text, strengths[4])
        text = self.d8_explicit_command_sanitization(text, strengths[5])
        text = self.d6_audit_trail_addition(text, strengths[6] if len(strengths) > 6 else 0.5)
        return text


# ============================================================
# 3. DETECTION MODELS (from Task B)
# ============================================================

def build_detectors(df):
    """Rebuild Task B detection models."""
    tfidf_vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    tfidf_matrix = tfidf_vec.fit_transform(df["text"])

    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(tfidf_matrix)

    svd = TruncatedSVD(n_components=80, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix.toarray())
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1, novelty=True, n_jobs=-1)
    lof.fit(tfidf_reduced)

    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.task_b_anomaly_detection import extract_structural_features
    structural = extract_structural_features(df)
    structural_scaled = StandardScaler().fit_transform(structural.values.astype(float))
    X_all = hstack([tfidf_matrix, structural_scaled]).tocsr()

    labels = pd.read_csv(Path("output/task_b/anomaly_results.csv"))["weak_label"].values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_all, y)

    return {
        "tfidf_vec": tfidf_vec, "iso": iso, "lof": lof, "svd": svd,
        "xgb": xgb, "le": le, "structural_scaled": structural_scaled,
    }


def detect_text(text, models):
    """Run all 3 detectors on a text."""
    tfidf = models["tfidf_vec"].transform([text])
    iso_score = float(models["iso"].decision_function(tfidf)[0])
    iso_pred = -1 if iso_score < 0 else 1

    lof_reduced = models["svd"].transform(tfidf.toarray())
    lof_score = float(models["lof"].decision_function(lof_reduced)[0])
    lof_pred = -1 if lof_score < 0 else 1

    struct_feats = models["structural_scaled"].mean(axis=0, keepdims=True)
    X_sample = hstack([tfidf, struct_feats]).tocsr()
    xgb_pred = int(models["xgb"].predict(X_sample)[0])
    xgb_label = str(models["le"].inverse_transform([xgb_pred])[0])
    xgb_anomaly = xgb_label != "normal"

    return {
        "iso_anomaly": iso_pred == -1,
        "lof_anomaly": lof_pred == -1,
        "xgb_anomaly": xgb_anomaly,
        "iso_score": iso_score,
        "lof_score": lof_score,
        "n_detected": sum([iso_pred == -1, lof_pred == -1, xgb_anomaly]),
    }


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("  Task D: Dual-Use De-risking Framework")
    print("  Based on: KRI Scoring + Debloating Taxonomy + Semantic Elasticity")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_DIR / "skills_raw.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))
    labels = pd.read_csv(Path("output/task_b/anomaly_results.csv"))["weak_label"]
    df["label"] = labels.values

    # Load Task C adversarial results for cross-comparison
    tc_path = Path("output/task_c/adversarial_results.json")
    tc_results = {}
    if tc_path.exists():
        with open(tc_path) as f:
            tc_data = json.load(f)
            for r in tc_data["results"]:
                tc_results[r["idx"]] = r

    print(f"  {len(df)} skills loaded")
    print(f"  {len(tc_results)} Task C adversarial variants available")

    # Build models
    print("\n[2/6] Building detection models...")
    models = build_detectors(df)

    # Select seeds
    malicious_unsafe = df[df["label"].isin(["malicious", "unsafe"])].head(25)
    print(f"  Seeds: {len(malicious_unsafe)} (malicious={sum(malicious_unsafe['label']=='malicious')}, "
          f"unsafe={sum(malicious_unsafe['label']=='unsafe')})")

    # Initialize tools
    print("\n[3/6] De-risking and scoring...")
    derisker = DeRiskingOperators()
    scorer = KRIRiskScorer(w_threat=0.4, w_impact=0.35, w_exposure=0.25)

    results = []
    for i, (idx, row) in enumerate(malicious_unsafe.iterrows()):
        orig_text = row["text"]
        orig_risks = str(row.get("risks", ""))
        orig_perms = str(row.get("permissions", ""))

        # Compute original risk
        orig_kri = scorer.composite_score(orig_text, orig_risks, orig_perms)
        orig_det = detect_text(orig_text, models)

        # Apply de-risking
        derisked_text = derisker.apply_all(orig_text)
        derisked_kri = scorer.composite_score(derisked_text, orig_risks, orig_perms)
        derisked_det = detect_text(derisked_text, models)

        # Compute semantic similarity
        v1 = models["tfidf_vec"].transform([orig_text])
        v2 = models["tfidf_vec"].transform([derisked_text])
        sem_sim = float(cosine_similarity(v1, v2)[0, 0])

        # Semantic elasticity: |functional preservation| / |text change|
        len_ratio = min(len(orig_text), len(derisked_text)) / max(len(orig_text), len(derisked_text), 1)
        sem_elasticity = sem_sim / max(1 - len_ratio, 0.01) if len_ratio < 0.99 else sem_sim

        # Also de-risk the Task C adversarial variant (dual-use cross)
        tc_variant = None
        tc_derisked_kri = None
        tc_derisked_det = None
        if int(idx) in tc_results:
            tc_variant = tc_results[int(idx)]["variant_text"]
            tc_derisked_text = derisker.apply_all(tc_variant)
            tc_derisked_kri = scorer.composite_score(tc_derisked_text, orig_risks, orig_perms)
            tc_derisked_det = detect_text(tc_derisked_text, models)

        result = {
            "idx": int(idx),
            "name": str(row["name"]),
            "label": str(row["label"]),
            "category": str(row.get("category", "")),
            # Original
            "orig_kri": orig_kri,
            "orig_detected_by": orig_det["n_detected"],
            # De-risked
            "derisked_kri": derisked_kri,
            "derisked_detected_by": derisked_det["n_detected"],
            # Metrics
            "kri_delta": round(orig_kri["kri"] - derisked_kri["kri"], 4),
            "semantic_similarity": round(sem_sim, 4),
            "semantic_elasticity": round(min(sem_elasticity, 10.0), 4),
            "length_ratio": round(len_ratio, 4),
            # Detection delta
            "detection_delta": orig_det["n_detected"] - derisked_det["n_detected"],
        }

        # Cross-comparison with Task C
        if tc_variant:
            result["has_tc_variant"] = True
            tc_orig_kri = scorer.composite_score(tc_variant, orig_risks, orig_perms)
            result["tc_orig_kri"] = tc_orig_kri["kri"]
            result["tc_derisked_kri"] = tc_derisked_kri["kri"] if tc_derisked_kri else None
            result["tc_kri_delta"] = round(
                tc_orig_kri["kri"] - (tc_derisked_kri["kri"] if tc_derisked_kri else 0), 4)
        else:
            result["has_tc_variant"] = False

        results.append(result)

        if i < 3 or i % 8 == 0:
            print(f"  [{i+1}/{len(malicious_unsafe)}] {row['name'][:45]}... "
                  f"KRI: {orig_kri['kri']:.3f}→{derisked_kri['kri']:.3f} "
                  f"(Δ={result['kri_delta']:.3f}), "
                  f"Det: {orig_det['n_detected']}→{derisked_det['n_detected']}, "
                  f"sim={sem_sim:.3f}")

    # Summary statistics
    print(f"\n[4/6] Computing summary...")

    kri_deltas = [r["kri_delta"] for r in results]
    sims = [r["semantic_similarity"] for r in results]
    detection_deltas = [r["detection_delta"] for r in results]
    elasticities = [r["semantic_elasticity"] for r in results]

    n_risk_reduced = sum(1 for d in kri_deltas if d > 0)
    n_detection_reduced = sum(1 for d in detection_deltas if d > 0)

    summary = {
        "n_seeds": len(results),
        "risk_reduction_rate": f"{n_risk_reduced}/{len(results)} ({n_risk_reduced/len(results)*100:.1f}%)",
        "detection_reduction_rate": f"{n_detection_reduced}/{len(results)} ({n_detection_reduced/len(results)*100:.1f}%)",
        "avg_kri_delta": round(np.mean(kri_deltas), 4),
        "max_kri_delta": round(max(kri_deltas), 4),
        "avg_semantic_similarity": round(np.mean(sims), 4),
        "avg_semantic_elasticity": round(np.mean(elasticities), 4),
        "avg_detection_delta": round(np.mean(detection_deltas), 4),
        "risk_level_changes": {},
    }

    # Risk level transitions
    level_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "minimal": 0}
    transitions = {}
    for r in results:
        old_lvl = r["orig_kri"]["risk_level"]
        new_lvl = r["derisked_kri"]["risk_level"]
        key = f"{old_lvl}→{new_lvl}"
        transitions[key] = transitions.get(key, 0) + 1
    summary["risk_level_transitions"] = transitions

    # Cross-comparison with Task C
    tc_cross = [r for r in results if r.get("has_tc_variant")]
    if tc_cross:
        tc_kri_deltas = [r["tc_kri_delta"] for r in tc_cross]
        summary["cross_comparison"] = {
            "n_cross": len(tc_cross),
            "avg_tc_kri_delta": round(np.mean(tc_kri_deltas), 4),
            "note": "De-risking applied to Task C adversarial variants — KRI reduction on already-evasive samples",
        }

    # Print summary
    print(f"\n{'='*70}")
    print("  TASK D RESULTS")
    print(f"{'='*70}")
    print(f"\n  Risk Score Reduction:")
    print(f"    Avg KRI delta:   {summary['avg_kri_delta']:.4f}")
    print(f"    Max KRI delta:   {summary['max_kri_delta']:.4f}")
    print(f"    Risk reduced in: {summary['risk_reduction_rate']}")
    print(f"\n  Risk Level Transitions:")
    for trans, cnt in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"    {trans}: {cnt}")
    print(f"\n  Semantic Preservation:")
    print(f"    Avg similarity:  {summary['avg_semantic_similarity']:.4f}")
    print(f"    Avg elasticity:  {summary['avg_semantic_elasticity']:.4f}")
    print(f"\n  Detection Impact:")
    print(f"    Avg detection Δ: {summary['avg_detection_delta']:.4f}")
    print(f"    Detection reduced in: {summary['detection_reduction_rate']}")

    if tc_cross:
        print(f"\n  Cross-Comparison (Task C variants → De-risked):")
        print(f"    {len(tc_cross)} adversarial variants de-risked")
        print(f"    Avg TC KRI delta: {summary['cross_comparison']['avg_tc_kri_delta']:.4f}")

    # Generate comparison table
    print(f"\n[5/6] Generating comparison table...")
    comparison_rows = []
    for r in results:
        row_data = {
            "idx": r["idx"],
            "name": r["name"],
            "label": r["label"],
            "orig_kri": r["orig_kri"]["kri"],
            "orig_risk_level": r["orig_kri"]["risk_level"],
            "derisked_kri": r["derisked_kri"]["kri"],
            "derisked_risk_level": r["derisked_kri"]["risk_level"],
            "kri_delta": r["kri_delta"],
            "threat_delta": round(r["orig_kri"]["threat"] - r["derisked_kri"]["threat"], 4),
            "impact_delta": round(r["orig_kri"]["impact"] - r["derisked_kri"]["impact"], 4),
            "exposure_delta": round(r["orig_kri"]["exposure"] - r["derisked_kri"]["exposure"], 4),
            "semantic_similarity": r["semantic_similarity"],
            "semantic_elasticity": r["semantic_elasticity"],
            "detection_delta": r["detection_delta"],
            "has_tc_variant": r.get("has_tc_variant", False),
        }
        if r.get("has_tc_variant"):
            row_data["tc_kri_delta"] = r["tc_kri_delta"]
        comparison_rows.append(row_data)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = OUTPUT_DIR / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Comparison table → {comparison_path}")

    # Save full results
    print(f"\n[6/6] Saving outputs...")
    output = {
        "summary": summary,
        "results": results,
        "parameters": {
            "kri_weights": {"threat": 0.4, "impact": 0.35, "exposure": 0.25},
            "derisking_operators": [
                "D7: Remove Dangerous Flags",
                "D1: Safety Guard Insertion",
                "D2: Permission Minimization",
                "D3: Risk Transparency",
                "D4: Action Justification",
                "D8: Explicit Command Sanitization",
                "D6: Audit Trail Addition",
            ],
        },
    }

    with open(OUTPUT_DIR / "derisking_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Full results → {OUTPUT_DIR / 'derisking_results.json'}")

    # Print top/bottom cases
    print(f"\n{'='*70}")
    print("  TOP CASES")
    print(f"{'='*70}")
    print("\n  Best KRI reduction (>0.03):")
    best = sorted(results, key=lambda x: x["kri_delta"], reverse=True)[:5]
    for r in best:
        print(f"    {r['name'][:55]} | KRI: {r['orig_kri']['kri']:.3f}→{r['derisked_kri']['kri']:.3f} "
              f"(Δ={r['kri_delta']:.3f}) | sim={r['semantic_similarity']:.3f}")

    print(f"\n  Best semantic preservation (sim >0.99):")
    best_sim = sorted(results, key=lambda x: x["semantic_similarity"], reverse=True)[:5]
    for r in best_sim:
        print(f"    {r['name'][:55]} | sim={r['semantic_similarity']:.4f} "
              f"| KRI Δ={r['kri_delta']:.4f}")

    print(f"\n{'='*70}")
    print("  Task D complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
