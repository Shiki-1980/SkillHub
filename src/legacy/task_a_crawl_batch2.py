"""
DataHub Skills 增量爬虫 — Task A Batch 2
爬取 ~5000 条新 skill，跳过已有，不重复。
写入 data/skills_raw_batch2.csv。
"""
import requests, csv, time, os, sys, pandas as pd
from pathlib import Path

API_BASE = "https://www.fudankw.cn/skills-api"
OUTPUT_FILE = "data/skills_raw_batch2.csv"
PAGE_SIZE = 50
TARGET_NEW = 5000
DETAIL_DELAY = 0.15

FIELDNAMES = [
    "source_id", "name", "description", "actions", "permissions",
    "category", "tags", "form", "limitations", "risks",
    "author", "stars", "score", "language", "quality",
]

def norm_actions(capabilities):
    return "; ".join(capabilities) if capabilities else ""

def norm_permissions(requirements):
    if not requirements: return ""
    return "; ".join(f"[{r.get('status','')}] {r.get('label','')}: {r.get('desc','')}" for r in requirements)

def norm_list(lst):
    return "; ".join(lst) if lst else ""

def norm_risks(risks):
    if not risks: return ""
    return "; ".join(f"[{r.get('level','')}] {r.get('label','')}: {r.get('desc','')}" for r in risks)

def api_get(url, **kw):
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=30, **kw)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 3: time.sleep(1.0 * attempt)
            else: raise e

def main():
    # Load existing IDs
    print("Loading existing skill IDs...")
    existing = set()
    for f in ["data/skills_raw.csv"]:
        if Path(f).exists():
            df = pd.read_csv(f)
            existing.update(df["source_id"].tolist())
    print(f"  Existing: {len(existing)}")

    # Phase 1: collect new IDs
    print("\nPhase 1: Collecting new skill IDs...")
    new_summaries = []
    # Start from ~page 111 (existing 5499 / 50 per page ≈ 110)
    page = 111
    stopped_pages = False

    while len(new_summaries) < TARGET_NEW:
        try:
            data = api_get(f"{API_BASE}/skills",
                          params={"page": page, "page_size": PAGE_SIZE, "collection": "skills"})
        except Exception as e:
            print(f"  page {page} FAIL: {e}")
            page += 1
            continue

        datasets = data.get("datasets", [])
        if not datasets:
            if not stopped_pages:
                print(f"  page {page}: empty, may have reached end")
                stopped_pages = True
            page += 1
            if page > 2500:  # safety limit
                break
            continue

        new_on_page = 0
        for sk in datasets:
            sid = sk["id"]
            if sid not in existing:
                existing.add(sid)  # track to avoid intra-run duplicates
                new_summaries.append({
                    "id": sid,
                    "title": sk.get("title", ""),
                    "description": sk.get("description", ""),
                    "category": sk.get("category", ""),
                    "tags": sk.get("tags", []),
                    "form": sk.get("form", ""),
                    "author": sk.get("author", ""),
                    "stars": sk.get("stars", 0),
                    "score": sk.get("score", 0),
                })
                new_on_page += 1

        total = data.get("total", 0)
        print(f"  page {page:4d} | new on page: {new_on_page} | total new: {len(new_summaries)}/{TARGET_NEW} (API total: {total})", flush=True)

        if new_on_page == 0 and stopped_pages:
            print("  Two empty pages in a row, stopping")
            break

        page += 1

    print(f"\nPhase 1 done: {len(new_summaries)} new IDs\n")

    # Phase 2: fetch details
    print(f"Phase 2: Fetching details (delay={DETAIL_DELAY}s)...")
    ok = fail = 0
    t0 = time.time()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    f = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    buffer = []

    try:
        for i, sk in enumerate(new_summaries):
            time.sleep(DETAIL_DELAY)
            try:
                detail = api_get(f"{API_BASE}/skills/{sk['id']}")["skill"]
                row = {
                    "source_id": sk["id"],
                    "name": detail.get("title", sk["title"]),
                    "description": detail.get("desc", sk["description"]),
                    "actions": norm_actions(detail.get("capabilities", [])),
                    "permissions": norm_permissions(detail.get("requirements", [])),
                    "category": detail.get("category", sk["category"]),
                    "tags": "; ".join(detail.get("tags", sk["tags"])),
                    "form": detail.get("implementation", {}).get("form", sk["form"]),
                    "limitations": norm_list(detail.get("limitations", [])),
                    "risks": norm_risks(detail.get("risks", [])),
                    "author": sk["author"],
                    "stars": sk["stars"],
                    "score": sk["score"],
                    "language": detail.get("language", ""),
                    "quality": detail.get("quality", ""),
                }
                buffer.append(row)
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 3: print(f"  FAIL [{i}] {sk['id'][:60]}: {e}")

            if len(buffer) >= 100:
                writer.writerows(buffer)
                f.flush()
                buffer.clear()

            if (i+1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(new_summaries) - i - 1)
                print(f"  [{i+1:5d}/{len(new_summaries)}] ok={ok} fail={fail}  eta={eta:.0f}s", flush=True)

        if buffer:
            writer.writerows(buffer)
            f.flush()
    finally:
        f.close()

    elapsed = time.time() - t0
    print(f"\nPhase 2 done: {ok} success, {fail} failures ({elapsed:.0f}s)")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
