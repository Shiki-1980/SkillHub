"""
Layer 2: LLM Four-Dimension Post-Hoc Interpretation (SkillSieve-inspired).
Does NOT change classification — only explains WHY a skill was flagged.
Feeds into Task D: pinpoint which dimension needs de-risking.
"""
import os, json, time, numpy as np, pandas as pd
from pathlib import Path
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are a security analyst reviewing AI agent skills flagged by an anomaly detector.

Analyze this skill across 4 dimensions and produce a structured assessment. Your analysis does NOT change the classification — it only explains WHY the detector flagged it, and what could be done to reduce its risk.

FOUR DIMENSIONS (SkillSieve-inspired):
A. Intent Alignment: Does what the skill CLAIMS (name/description) match what it INSTRUCTS (actions)?
B. Permission Justification: Are requested permissions reasonable for the stated purpose?
C. Covert Behavior: Instructions to hide actions, suppress errors, bypass safety?
D. Overall Risk: Synthesize. Is the detector's flag justified?

OUTPUT strict JSON only:
{
  "label": "<current classification>",
  "agreement": "agree|disagree|partial",
  "confidence": 0.0-1.0,
  "dim_A": {"finding": "<brief>", "risk": "none|low|medium|high"},
  "dim_B": {"finding": "<brief>", "risk": "none|low|medium|high"},
  "dim_C": {"finding": "<brief>", "risk": "none|low|medium|high"},
  "primary_risk_dimension": "A|B|C",
  "derisking_suggestion": "<concrete step to reduce risk while preserving functionality>",
  "explanation": "<2-3 sentence summary for an auditor>"
}"""

USER_PROMPT = """ANOMALOUS SKILL (classified as {label} by XGBoost):
---
Name: {name}
Description: {description}
Category: {category}
Actions: {actions}
Permissions: {permissions}
Risks: {risks}
---

Return JSON analysis."""


def analyze_skill(client, row, label, max_retries=3):
    prompt = USER_PROMPT.format(
        label=label,
        name=str(row.get("name", "")),
        description=str(row.get("description", ""))[:1500],
        category=str(row.get("category", "")),
        actions=str(row.get("actions", ""))[:2000],
        permissions=str(row.get("permissions", ""))[:1000],
        risks=str(row.get("risks", ""))[:500],
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500, temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences
            for fence in ["```json", "```"]:
                if fence in text:
                    text = text.split(fence)[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e)[:200]}


def run_layer2(df_anomalous, sample_size=100):
    """Run Layer 2 analysis on anomalous skills. Returns list of explanations."""
    api_key = API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    sample = df_anomalous.head(sample_size)
    results = []

    for i, (idx, row) in enumerate(sample.iterrows()):
        label = str(row.get("label", "unsafe"))
        print(f"  [{i+1}/{len(sample)}] {str(row['name'])[:50]}...", end=" ", flush=True)

        analysis = analyze_skill(client, row, label)
        analysis["idx"] = int(idx)
        analysis["name"] = str(row["name"])
        analysis["xgb_label"] = label
        results.append(analysis)

        agree = analysis.get("agreement", "?")
        dim = analysis.get("primary_risk_dimension", "?")
        print(f"→ {agree} | primary: dim_{dim}")

        if (i + 1) % 20 == 0:
            print(f"  [checkpoint {i+1}/{len(sample)}]")

    return results


def compute_layer2_stats(results):
    """Aggregate Layer 2 results."""
    stats = {
        "n_analyzed": len(results),
        "agreement": {},
        "primary_risk_dimension": {},
        "dim_risk_distribution": {"A": {}, "B": {}, "C": {}},
    }
    for r in results:
        stats["agreement"][r.get("agreement", "error")] = \
            stats["agreement"].get(r.get("agreement", "error"), 0) + 1
        dim = r.get("primary_risk_dimension", "unknown")
        stats["primary_risk_dimension"][dim] = \
            stats["primary_risk_dimension"].get(dim, 0) + 1
        for d in ["A", "B", "C"]:
            lvl = r.get(f"dim_{d}", {}).get("risk", "unknown") if isinstance(r.get(f"dim_{d}"), dict) else "unknown"
            stats["dim_risk_distribution"][d][lvl] = \
                stats["dim_risk_distribution"][d].get(lvl, 0) + 1
    return stats
