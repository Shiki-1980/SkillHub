"""
KRI Weight Calibration via Grid Search
========================================
Optimize w_threat, w_impact, w_exposure weights by maximizing
correlation with anomaly labels and detection agreement.
"""
import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

OUT = Path("output/task_b")


def load():
    df = pd.read_csv("data/skills_raw_merged.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))
    labels = pd.read_csv(OUT / "merged_rule_labels.csv")["rule_label"].values
    return df, labels


def compute_kri(text, risks, perms, w_t, w_i, w_e):
    """Simplified KRI scorer matching task_d_derisking logic."""
    import re

    patterns_t = {
        "eval_exec": r"\b(eval\s*\(|exec\s*\(|subprocess|os\.system|rm\s+-rf|sudo\s+)",
        "reverse_shell": r"\b(reverse\s*shell|bind\s*shell)",
        "credential_theft": r"\b(password|token|secret|credential).*(steal|leak|exfiltrat|extract)",
        "prompt_injection": r"\b(prompt\s*inject|jailbreak|ignore.*instructions?)",
        "danger_network": r"\b(curl\s+\S+\s*\|.*sh|wget.*-O.*\|.*sh|nc\s+-[nlvp])",
        "disable_safety": r"\b(disable|bypass|skip)\s+(security|safety|verification)",
        "hidden_behavior": r"\b(hidden|silently|secretly|invisible|undetected|covert)",
    }
    patterns_i = {
        "os_access": r"(linux|macos|windows)",
        "shell_access": r"(bash|zsh|sh|powershell|shell)",
        "network_access": r"(网络|network|api|http|curl|wget)",
        "filesystem_access": r"(文件系统|filesystem|file|directory|read|write|delete)",
        "env_vars": r"(环境变量|env|environment|\.env|API_KEY|TOKEN|SECRET)",
    }
    patterns_e = {
        "url_reference": r"https?://[^\s]+",
        "external_dep": r"(依赖.*外部|depends on|requires.*service|needs.*api)",
        "user_data": r"(user.*(data|input|file|content)|process.*(file|data))",
    }

    def score(pat_dict, txt):
        if not txt: return 0
        total = sum(len(re.findall(p, txt.lower())) for p in pat_dict.values())
        return min(1.0, total / max(len(pat_dict), 1))

    t = score(patterns_t, text)
    i = score(patterns_i, text) * 0.5 + score(patterns_i, str(perms)) * 0.5
    e = score(patterns_e, text)

    # Danger label bonus
    danger_n = len(re.findall(r"\[danger\]", str(risks)))
    t = 0.7 * t + 0.3 * min(1.0, danger_n * 0.25)

    return max(0, w_t * t + w_i * i + w_e * e)


def main():
    print("=" * 60)
    print("  KRI Weight Calibration via Grid Search")
    print("=" * 60)

    df, labels = load()
    bt = (labels != "normal").astype(int)

    # Compute KRI scores for all weight combinations
    print("\n[1] Grid search over weights...")
    weight_range = np.arange(0.15, 0.60, 0.05)
    best = {"f1": 0, "w_t": 0, "w_i": 0, "w_e": 0}

    all_results = []
    total = len(list(product(weight_range, repeat=3)))
    for wi, (w_t, w_i, w_e) in enumerate(product(weight_range, repeat=3)):
        if abs(w_t + w_i + w_e - 1.0) > 0.01:
            continue

        # Compute KRI for all skills
        kri_scores = np.array([
            compute_kri(df.loc[i, "text"], str(df.loc[i, "risks"]),
                        str(df.loc[i, "permissions"]), w_t, w_i, w_e)
            for i in range(len(df))
        ])

        # Threshold sweep to find best F1
        best_thresh_f1 = 0
        best_thresh = 0.3
        for thresh in np.arange(0.1, 0.8, 0.02):
            preds = (kri_scores > thresh).astype(int)
            f1 = f1_score(bt, preds, zero_division=0)
            if f1 > best_thresh_f1:
                best_thresh_f1 = f1
                best_thresh = thresh

        all_results.append({
            "w_t": round(w_t, 2), "w_i": round(w_i, 2), "w_e": round(w_e, 2),
            "f1": round(best_thresh_f1, 4), "threshold": round(best_thresh, 2),
        })

        if best_thresh_f1 > best["f1"]:
            best = {"f1": best_thresh_f1, "w_t": w_t, "w_i": w_i, "w_e": w_e,
                    "threshold": best_thresh}

        if wi % 50 == 0:
            print(f"  {wi}/{total}... best F1 so far: {best['f1']:.4f}")

    # Top 10
    top10 = sorted(all_results, key=lambda x: x["f1"], reverse=True)[:10]

    print(f"\n[2] Top 10 weight configurations:")
    print(f"    {'w_t':>6}  {'w_i':>6}  {'w_e':>6}  {'thresh':>7}  {'F1':>7}")
    print(f"    {'─'*40}")
    for r in top10:
        star = " ★" if r["f1"] == best["f1"] else ""
        print(f"    {r['w_t']:>6.2f}  {r['w_i']:>6.2f}  {r['w_e']:>6.2f}  "
              f"{r['threshold']:>7.2f}  {r['f1']:>7.4f}{star}")

    print(f"\n  Current weights: w_t=0.40, w_i=0.35, w_e=0.25")
    # Evaluate current weights
    cur_kri = np.array([compute_kri(df.loc[i,"text"], str(df.loc[i,"risks"]),
                                     str(df.loc[i,"permissions"]), 0.40, 0.35, 0.25)
                        for i in range(len(df))])
    cur_best = 0
    for thresh in np.arange(0.1, 0.8, 0.02):
        f1 = f1_score(bt, (cur_kri > thresh).astype(int), zero_division=0)
        if f1 > cur_best: cur_best = f1
    print(f"  Current F1: {cur_best:.4f}")
    print(f"  Best F1:    {best['f1']:.4f}")
    print(f"  Gain:       {best['f1'] - cur_best:+.4f}")

    # Stability analysis
    scores = [r["f1"] for r in all_results]
    print(f"\n[3] Stability analysis:")
    print(f"  F1 range:  {min(scores):.4f} – {max(scores):.4f}")
    print(f"  F1 mean:   {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Weight importance
    wt_effect = {}
    for w_val in weight_range:
        subset = [r["f1"] for r in all_results if abs(r["w_t"] - w_val) < 0.005]
        if subset: wt_effect[round(w_val, 2)] = np.mean(subset)
    print(f"\n  Threat weight sensitivity:")
    for w, f1 in sorted(wt_effect.items()):
        bar = "█" * int(f1 * 50)
        print(f"    w_t={w:.2f}: {f1:.4f} {bar}")

    # Save
    report = {
        "current_weights": {"w_t": 0.40, "w_i": 0.35, "w_e": 0.25},
        "current_best_f1": round(cur_best, 4),
        "best_weights": {k: round(v, 2) for k, v in best.items() if k != "f1"},
        "best_f1": round(best["f1"], 4),
        "f1_gain": round(best["f1"] - cur_best, 4),
        "top10": top10,
        "n_combinations_tested": len(all_results),
    }
    Path("output/task_d/kri_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Saved → output/task_d/kri_calibration.json")


if __name__ == "__main__":
    main()
