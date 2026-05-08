# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

期末项目：DataHub 异常技能挖掘。四个任务：A 数据获取 → B 异常检测 → C 对抗生成 → D 去风险化。
数据源：DataHub Skills API (`https://www.fudankw.cn/skills-api`)，平台共 105586 条 skill。

## Directory structure

```
skillHub/
├── CLAUDE.md
├── docs/                            # 课程材料
│   └── 期末项目_SkillHub异常挖掘项目.pptx
├── data/
│   ├── skills_raw.csv               # 5499 条完整数据
│   ├── skills_test.csv              # 测试数据
│   └── skills_sample.csv            # 样本数据 (11 条)
├── papers/
│   ├── TaskB/                       # Task B 参考文献 (3 篇)
│   └── TaskC/                       # Task C 参考文献 (9 篇)
├── src/
│   ├── task_a_crawl_skills.py       # Task A: 数据爬取
│   ├── task_b_anomaly_detection.py  # Task B: 异常检测
│   ├── task_c_adversarial_gen.py    # Task C: 对抗生成
│   └── utils/
│       ├── llm_labeling.py          # LLM 弱标注脚本
│       └── merge_labels.py          # 标签合并脚本
└── output/
    ├── task_b/
    │   ├── anomaly_results.csv      # 5499 条全量检测结果
    │   ├── report.json              # 评估指标 JSON
    │   ├── Task_B_Report.md         # Task B 详细报告
    │   ├── llm_labels_checkpoint.json  # 800 条 LLM 标注
    │   └── llm_labeling_tasks.json     # LLM 标注任务定义
    └── task_c/
        ├── adversarial_results.json # 对抗生成结果
        └── Task_C_Report.md         # Task C 详细报告
```

## Data

`data/skills_raw.csv` — 5499 条 skill，15 个字段。

必需字段（Task A 要求）：`name` / `description` / `actions` / `permissions`。
其他字段：`source_id` / `category` / `tags` / `form` / `limitations` / `risks` / `author` / `stars` / `score` / `language` / `quality`。

## Commands

```bash
# Task A: 数据爬取
python3 src/task_a_crawl_skills.py

# Task B: 异常检测
python3 src/task_b_anomaly_detection.py

# Task C: 对抗生成
python3 src/task_c_adversarial_gen.py

# LLM 标注 (DeepSeek API)
python3 src/utils/llm_labeling.py --max 800
python3 src/utils/llm_labeling.py --resume   # 断点续跑

# 标签合并
python3 src/utils/merge_labels.py
```

依赖：`requests` `scikit-learn` `xgboost` `pandas` `numpy` `scipy` `openai`。

## Key details

- API 列表接口返回 `title/description/tags/category/form` 等，但**不含** `capabilities` 和 `requirements`
- 详情接口的 `risks` 是 dict 列表（`{label, desc, level}`），不是字符串
- `requirements` 也是 dict 列表（`{label, desc, status}`）
- 不要用 `requests.Session` 跨线程共享（不线程安全）
- skill ID 含 `/`（如 `steipete/clawdis/notion`），直接拼 URL 即可

## Task status

| Task | Status | Output |
|---|---|---|
| A 数据获取 | ✅ 完成 | `data/skills_raw.csv` (5499 条) |
| B 异常检测 | ✅ 完成 | `output/task_b/` |
| C 对抗生成 | ✅ 完成 | `output/task_c/` |
| D 去风险化 | ⏳ 待做 | — |

## PPT 关键要求

- Task B：至少 2 种方法 + TF-IDF 必须有 + 至少一种无监督异常检测（LOF/DBSCAN/Isolation Forest）+ 一种分类器。禁止 LLM 直接出分类结果。
- Task C：生成策略必须可复现（迭代/规则/进化），禁止仅 prompt 改写。保留风险语义。
- Task D：改写前后逐条对照，风险评分 + 语义相似度双指标量化。
- 消融实验：去 embedding 效果 + 去 agent 辅助效果。
- 提交：PDF 报告 + 可运行代码 + 数据说明文档 + agent 使用记录。
