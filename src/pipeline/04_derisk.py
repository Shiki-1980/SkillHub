"""
Pipeline: De-risking + LLM Functionality Audit (Task D).
1. Apply 7 derisking operators to anomalous skills
2. Compute KRI delta (risk score change)
3. LLM audit: verify functionality preserved
Usage: python3 -m src.pipeline.04_derisk [--audit N]
"""
import sys, json, os, warnings, numpy as np, pandas as pd
sys.path.insert(0, ".")
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

from src.data.loader import load_skills
from src.derisking.kri_scorer import KRIRiskScorer
from src.derisking.derisking_operators import DeRiskingOperators

warnings.filterwarnings("ignore")
OUT = Path("output/task_d")
OUT.mkdir(parents=True, exist_ok=True)

FUNC_AUDIT_PROMPT = """You are auditing whether a de-risking transformation preserved the core functionality of an AI agent skill.

Given the ORIGINAL skill and the DE-RISKED version, answer:

1. Core functionality preserved? (yes / partial / no)
2. What was the original skill designed to do? (1 sentence)
3. Does the de-risked version still accomplish that? (explain)
4. What safety improvements were made? (list)
5. Was any functionality REMOVED? (yes / no)

Output JSON only:
{
  "functionality_preserved": "yes|partial|no",
  "original_purpose": "<1 sentence>",
  "still_accomplishes": "<explanation>",
  "safety_improvements": ["<item1>", "<item2>", ...],
  "functionality_removed": "yes|no"
}"""


def llm_audit(client, row, derisked_text):
    prompt = f"""ORIGINAL SKILL:
Name: {row.get('name','')}
Description: {str(row.get('description',''))[:1200]}
Actions: {str(row.get('actions',''))[:800]}
Permissions: {str(row.get('permissions',''))[:500]}

DE-RISKED VERSION:
{derisked_text[:1500]}

Audit the de-risking transformation."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": FUNC_AUDIT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400, temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            for fence in ["```json", "```"]:
                if fence in text:
                    text = text.split(fence)[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)[:200]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=int, default=3, help="Number of LLM functionality audits (0=skip)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Task D: De-risking + Functionality Audit")
    print("=" * 60)

    # Load
    df = load_skills()
    labels = pd.read_csv("output/task_b/full_labels_10501.csv")["label"].values
    anomalous = df[labels.isin(["malicious", "unsafe"])]
    n_seeds = min(100, len(anomalous))
    seeds = anomalous.sample(n_seeds, random_state=42)
    print(f"[1] {n_seeds} seeds")

    # De-risk
    scorer = KRIRiskScorer(w_threat=0.55, w_impact=0.25, w_exposure=0.20)
    derisker = DeRiskingOperators()
    tfidf = TfidfVectorizer(max_features=500, stop_words="english").fit(df["text"])

    results = []
    for i, (idx, row) in enumerate(seeds.iterrows()):
        ot = row["text"]
        orig_kri = scorer.compute(ot, str(row.get("risks","")), str(row.get("permissions","")))
        dt = derisker.apply_all(ot)
        derisked_kri = scorer.compute(dt, str(row.get("risks","")), str(row.get("permissions","")))
        sim = float(cosine_similarity(tfidf.transform([ot]), tfidf.transform([dt]))[0, 0])
        results.append({
            "idx": int(idx), "name": str(row["name"]),
            "orig_kri": orig_kri["kri"], "derisked_kri": derisked_kri["kri"],
            "kri_delta": round(orig_kri["kri"] - derisked_kri["kri"], 4),
            "semantic_similarity": round(sim, 4),
            "derisked_text": dt[:1500],
        })

    reduced = sum(1 for r in results if r["kri_delta"] > 0)
    print(f"[2] KRI reduced: {reduced}/{n_seeds} ({reduced/n_seeds*100:.0f}%)")
    print(f"    Avg KRI delta: {np.mean([r['kri_delta'] for r in results]):.4f}")
    print(f"    Avg cosine sim: {np.mean([r['semantic_similarity'] for r in results]):.4f}")

    # LLM Functionality Audit
    audit_results = []
    if args.audit > 0:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        env_path = Path(".env")
        if env_path.exists() and not api_key:
            for line in env_path.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
        if not api_key:
            print("[3] LLM audit: SKIPPED (no API key)")
        else:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            n_audit = min(args.audit, len(seeds))
            print(f"\n[3] LLM functionality audit on {n_audit} samples...")

            for i in range(n_audit):
                row = seeds.iloc[i]
                audit = llm_audit(client, row, results[i]["derisked_text"])
                audit["name"] = results[i]["name"]
                audit["idx"] = results[i]["idx"]
                audit_results.append(audit)
                status = audit.get("functionality_preserved", "error")
                removed = audit.get("functionality_removed", "?")
                print(f"  [{i+1}/{n_audit}] {row['name'][:45]}... → func={status}, removed={removed}")

            # Audit summary
            preserved = sum(1 for a in audit_results if a.get("functionality_preserved") == "yes")
            removed = sum(1 for a in audit_results if a.get("functionality_removed") == "yes")
            print(f"\n  Audit summary:")
            print(f"    Functionality preserved: {preserved}/{n_audit} ({preserved/n_audit*100:.0f}%)")
            print(f"    Functionality removed:   {removed}/{n_audit} ({removed/n_audit*100:.0f}%)")
            if audit_results:
                print(f"    Safety improvements (sample):")
                for imp in audit_results[0].get("safety_improvements", [])[:3]:
                    print(f"      - {imp}")

    # Save
    output = {
        "n_seeds": n_seeds,
        "kri_reduced": f"{reduced}/{n_seeds}",
        "avg_kri_delta": round(np.mean([r['kri_delta'] for r in results]), 4),
        "avg_semantic_sim": round(np.mean([r['semantic_similarity'] for r in results]), 4),
        "results": [{k: v for k, v in r.items() if k != "derisked_text"} for r in results],
    }
    if audit_results:
        output["functionality_audit"] = {
            "n_audited": len(audit_results),
            "preserved": preserved,
            "removed": removed,
            "results": audit_results,
        }

    json.dump(output, open(OUT / "derisking_results.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n[4] Saved → {OUT}/derisking_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
