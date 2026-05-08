"""
DataHub Skills 全量爬虫 — Task A 数据获取。
列表接口缺 actions / permissions，须逐条调详情接口。
顺序执行 + 增量写入，≥5000 条，带去重、重试。
"""
import requests
import csv
import time
import os
import sys

API_BASE = "https://www.fudankw.cn/skills-api"
OUTPUT_DIR = "data"
RAW_CSV = os.path.join(OUTPUT_DIR, "skills_raw.csv")
PAGE_SIZE = 50
TARGET_COUNT = 5500
DETAIL_DELAY = 0.2  # 可调小加快速度

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    "source_id", "name", "description", "actions", "permissions",
    "category", "tags", "form", "limitations", "risks",
    "author", "stars", "score", "language", "quality",
]

# ---------- normalizers ----------
def norm_actions(capabilities):
    return "; ".join(capabilities) if capabilities else ""

def norm_permissions(requirements):
    if not requirements:
        return ""
    return "; ".join(
        f"[{r.get('status','')}] {r.get('label','')}: {r.get('desc','')}" for r in requirements
    )

def norm_list(lst):
    return "; ".join(lst) if lst else ""

def norm_risks(risks):
    if not risks:
        return ""
    return "; ".join(
        f"[{r.get('level','')}] {r.get('label','')}: {r.get('desc','')}" for r in risks
    )

# ---------- API helpers (with retry) ----------
def api_get(url, **kwargs):
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < 3:
                time.sleep(1.0 * attempt)
    raise last_exc

def fetch_page(page):
    url = f"{API_BASE}/skills"
    return api_get(url, params={"page": page, "page_size": PAGE_SIZE, "collection": "skills"})

def fetch_detail(skill_id):
    url = f"{API_BASE}/skills/{skill_id}"
    return api_get(url)["skill"]

# ---------- main ----------
def main():
    # Phase 1: collect skill IDs from list API
    print("Phase 1: 采集 skill ID 列表...", flush=True)
    seen = set()
    summaries = []
    page = 1

    while len(summaries) < TARGET_COUNT:
        try:
            data = fetch_page(page)
        except Exception as e:
            print(f"  page {page} 失败: {e}", flush=True)
            page += 1
            continue

        datasets = data.get("datasets", [])
        if not datasets:
            print(f"  page {page} returned 0 datasets, stopping", flush=True)
            break

        for sk in datasets:
            sid = sk["id"]
            if sid not in seen:
                seen.add(sid)
                summaries.append({
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

        total = data.get("total", 0)
        print(f"  page {page:4d} │ {len(summaries):5d}/{TARGET_COUNT}  (total: {total})", flush=True)
        page += 1

    print(f"\nPhase 1 done: {len(summaries)} unique IDs\n", flush=True)

    # Phase 2: fetch details sequentially, write incrementally
    print(f"Phase 2: fetching details (sequential, delay={DETAIL_DELAY}s)...", flush=True)

    ok = 0
    fail = 0
    t0 = time.time()

    # Write header + start with empty file
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    f = open(RAW_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    buffer = []

    try:
        for i, sk in enumerate(summaries):
            time.sleep(DETAIL_DELAY)
            try:
                detail = fetch_detail(sk["id"])
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
                if fail <= 5:
                    print(f"  FAIL [{i}] {sk['id']}: {e}", flush=True)

            # flush buffer every 100 rows
            if len(buffer) >= 100:
                writer.writerows(buffer)
                f.flush()
                buffer.clear()

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(summaries) - i - 1) if i > 0 else 0
                print(f"  [{i+1:5d}/{len(summaries)}] ok={ok} fail={fail}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        # final flush
        if buffer:
            writer.writerows(buffer)
            f.flush()

    finally:
        f.close()

    elapsed = time.time() - t0
    print(f"\nPhase 2 done: {ok} success, {fail} failures  ({elapsed:.0f}s)", flush=True)
    print(f"Saved to {RAW_CSV}", flush=True)

if __name__ == "__main__":
    main()
