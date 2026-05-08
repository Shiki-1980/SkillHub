"""
LLM Weak Labeling for Task B — DeepSeek Edition
================================================
Uses DeepSeek API with SkillSieve's 4-dimension decomposition.
Usage:
  python3 run_llm_labeling.py --max 50           # Label first 50
  python3 run_llm_labeling.py --resume            # Resume from checkpoint
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

TASKS_PATH = Path("output/task_b/llm_labeling_tasks.json")
RESULTS_PATH = Path("output/task_b/llm_labels.json")
CHECKPOINT_PATH = Path("output/task_b/llm_labels_checkpoint.json")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are a senior security analyst auditing AI agent skills. Analyze each skill across 4 dimensions and classify it.

ANALYSIS DIMENSIONS:
A. Intent Alignment: Does what the skill CLAIMS to do match what it actually INSTRUCTS? Disguised attacks?
B. Permission Justification: Are requested permissions (env vars, files, network, binaries) reasonable?
C. Covert Behavior: Instructions to hide actions, suppress errors, or bypass safety?
D. Overall Risk: Synthesize all dimensions.

LABELS:
- "malicious": Intentionally harmful (steals data, malicious code, prompt injection, backdoor, exfiltration)
- "unsafe": Not intentionally harmful but dangerous (over-privileged, risky ops without safeguards)
- "useless": Placeholder, broken, empty, non-functional
- "normal": Legitimate, well-described, appropriate permissions

Respond ONLY with a valid JSON object (no markdown, no code fences):
{"label": "<malicious|unsafe|useless|normal>", "confidence": 0.0-1.0, "dim_A": "intent finding", "dim_B": "permission finding", "dim_C": "covert finding", "reasoning": "brief summary in English"}
"""

USER_PROMPT_TEMPLATE = """SKILL TO ANALYZE:
Name: {name}
Description: {description}
Category: {category}
Actions: {actions}
Permissions: {permissions}
Risks: {risks}
Tags: {tags}
Rule-based label hint: {current_label}

Return JSON classification only."""


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def label_skill(client, task, max_retries=3):
    prompt = USER_PROMPT_TEMPLATE.format(**task)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )

            msg = response.choices[0].message
            text = (msg.content or "").strip()

            # Strip markdown code fences if present
            for fence in ["```json", "```"]:
                if fence in text:
                    text = text.split(fence)[1].split("```")[0].strip()

            result = json.loads(text)
            result["idx"] = task["idx"]
            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"idx": task["idx"], "label": None, "confidence": 0,
                        "error": str(e)[:200]}

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    if args.resume:
        checkpoint = load_checkpoint()
        done_ids = {r["idx"] for r in checkpoint if r.get("label")}
        pending = [t for t in tasks if t["idx"] not in done_ids]
        results = checkpoint
        print(f"Resuming: {len(done_ids)} done, {len(pending)} remaining")
    else:
        pending = tasks[:args.max] if args.max else tasks
        results = []
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()

    print(f"Model: {MODEL} @ {BASE_URL}")
    print(f"Tasks: {len(pending)}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    start = time.time()

    for i, task in enumerate(pending):
        name_preview = task["name"][:55]
        print(f"[{i+1}/{len(pending)}] {name_preview}...", end=" ", flush=True)

        result = label_skill(client, task)
        results.append(result)

        label = result.get("label", "ERR")
        conf = result.get("confidence", 0)
        print(f"→ {label} ({conf:.2f})")

        if (i + 1) % 30 == 0:
            save_checkpoint(results)
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 60
            eta = (len(pending) - i - 1) / rate
            print(f"  [checkpoint] {rate:.1f} skills/min, ETA {eta:.0f}s")

    save_checkpoint(results)

    labels = [r.get("label") for r in results if r.get("label")]
    if labels:
        dist = Counter(labels)
        print(f"\nDone: {len(labels)} labeled")
        for lbl, cnt in dist.most_common():
            print(f"  {lbl}: {cnt} ({cnt/len(labels)*100:.1f}%)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
