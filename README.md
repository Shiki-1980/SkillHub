# SkillHub — AI Agent Skill 安全分析

> 期末项目 · 数据挖掘 · 2026 年 5 月

对 DataHub 平台的 5,499 条 AI Agent Skill 进行系统性安全分析，涵盖异常检测、对抗生成与去风险化。

## 项目结构

```
SkillHub/
├── src/                        # 代码
│   ├── task_a_crawl_skills.py       # Task A: 数据爬取
│   ├── task_b_anomaly_detection.py  # Task B: 异常检测
│   ├── task_c_adversarial_gen.py    # Task C: 对抗生成
│   ├── task_d_derisking.py          # Task D: 去风险化
│   └── utils/llm_labeling.py        # LLM 弱标注工具
├── data/skills_raw.csv         # 5,499 条原始数据
├── output/                     # 实验结果与报告
│   ├── Final_Report.md                 # 汇总报告
│   ├── task_b/{报告,结果,LLM标注}       # Task B 产出
│   ├── task_c/{报告,对抗样本}            # Task C 产出
│   └── task_d/{报告,去风险结果}          # Task D 产出
└── CLAUDE.md                   # 项目指引
```

## 快速开始

```bash
pip install pandas scikit-learn xgboost numpy scipy openai

# Task A: 数据爬取
python3 src/task_a_crawl_skills.py

# Task B: 异常检测
python3 src/task_b_anomaly_detection.py

# Task C: 对抗生成
python3 src/task_c_adversarial_gen.py

# Task D: 去风险化
python3 src/task_d_derisking.py
```

## 四任务概览

| Task | 内容 | 方法 | 核心指标 |
|---|---|---|---|
| A | 数据获取 | API 爬虫 + 增量写入 | 5,499 条 skill |
| B | 异常检测 | TF-IDF + IF/LOF + XGBoost + LLM弱标注 | CV macro-F1 0.829 |
| C | 对抗生成 | 进化算法 + 6个扰动算子 + 多目标优化 | 100% 逃逸率, sim 0.749 |
| D | 去风险化 | KRI风险评分 + 7个去风险算子 + Safety Bonus | 84% 降风险率, sim 0.911 |

详见 `output/Final_Report.md`。

## 参考文献

主要参考论文：
- **SkillSieve** (Hou & Yang, 2026) — 恶意 skill 检测三层框架
- **Qiu et al.** (IJNM, 2024) — TF-IDF vs BERT 在异常检测中的对比
- **AOP-Mal** (Xu et al., 2025) — 进化计算对抗恶意软件生成
- **GA Text Attack** (2020) — 多目标遗传优化文本对抗
- **KRI Framework** (2025) — 复合风险指标 (ROC-AUC 0.927)
- **Dual-Use GenAI** (2025) — 生成式AI在攻防两端的双重用途

完整参考文献列表见 `output/Final_Report.md`。
