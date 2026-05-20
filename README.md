# SkillHub — AI Agent Skill 安全分析

> 期末项目 · 数据挖掘 · 2026 年 5 月

对 DataHub 平台 10,501 条 AI Agent Skill 进行系统性安全分析：异常检测 → 对抗生成 → 去风险化。

---

## 快速开始

```bash
pip install pandas scikit-learn xgboost numpy scipy openai sentence-transformers torch seaborn umap-learn

# 运行核心流水线
python3 -m src.pipeline.02_label_and_detect    # 标注 + 特征工程 + 异常检测
python3 src/task_c_adversarial_gen.py           # 对抗生成
python3 src/task_d_derisking.py                 # 去风险化

# 扩展实验
python3 src/task_b_lstm_autoencoder.py          # LSTM Autoencoder
python3 src/task_b_sbert_benchmark.py           # SBERT 嵌入对比
python3 src/visualization.py                    # 可视化
```

---

## 项目结构

```
skillHub/
│
├── data/
│   ├── skills_raw.csv              # 5,499 条 (batch 1)
│   ├── skills_raw_batch2.csv       # 5,002 条 (batch 2)
│   └── skills_raw_merged.csv       # 合并 10,501 条
│
├── src/                            # === 源码 ===
│   │
│   ├── data/                       # 数据加载
│   │   └── loader.py               #   CSV 读取 + 文本拼接
│   │
│   ├── features/                   # 特征工程
│   │   └── structural_features.py  #   45 维结构化特征 (SkillSieve启发)
│   │
│   ├── labeling/                   # 弱标注
│   │   └── rule_labeling.py        #   8 条规则标注 (确定性)
│   ├── utils/llm_labeling.py       #   DeepSeek-chat LLM标注
│   │
│   ├── detection/                  # 异常检测模型
│   │   ├── isolation_forest.py     #   Isolation Forest [Liu+ 2008]
│   │   ├── local_outlier_factor.py #   LOF [Breunig+ 2000]
│   │   └── xgboost_classifier.py   #   XGBoost [Chen+ 2016]
│   │
│   ├── evaluation/                 # 评估
│   │   └── metrics.py              #   F1 / Precision / Recall / Accuracy
│   │
│   ├── pipeline/                   # 流水线编排
│   │   └── 02_label_and_detect.py  #   标注 → 特征 → 检测 (一键Task B)
│   │
│   ├── adversarial/                # 对抗生成 (算子+GA)
│   ├── derisking/                  # 去风险化 (KRI+算子)
│   ├── interpretability/           # 可解释性 (SHAP + Layer2 LLM)
│   ├── training/                   # 对抗训练
│   │
│   ├── task_b_lstm_autoencoder.py  # LSTM Autoencoder (第3种无监督)
│   ├── task_b_sbert_benchmark.py   # SBERT vs TF-IDF 基准测试
│   ├── task_c_adversarial_gen.py   # EASG 进化对抗生成
│   ├── task_d_derisking.py         # KRI 去风险化
│   ├── task_d_kri_calibration.py   # KRI 权重校准
│   ├── task_b_ensemble.py          # 方法集成投票
│   ├── visualization.py            # t-SNE/UMAP 可视化
│   └── utils/                      # 工具
│       ├── llm_labeling.py         #   LLM 标注脚本
│       └── merge_labels.py         #   标签合并
│
├── output/                         # === 输出 ===
│   ├── Final_Report.md             # 汇总报告
│   ├── task_b/                     # Task B 结果
│   │   ├── Task_B_Report.md
│   │   ├── detection_results.csv   # 全量检测结果
│   │   ├── ensemble_results.md
│   │   ├── sbert_benchmark.json
│   │   └── llm_labels_checkpoint.json
│   ├── task_c/Task_C_Report.md     # Task C 结果
│   ├── task_d/Task_D_Report.md     # Task D 结果
│   └── visualization/              # 可视化图 (6张 + 说明)
│       ├── README.md
│       ├── fig1_tsne_label_comparison.png
│       ├── fig2_tsne_category_comparison.png
│       ├── fig3_umap_label_comparison.png
│       ├── fig4_density_analysis.png
│       ├── fig5_dashboard.png
│       └── fig6_anomaly_by_category.png
│
└── papers/                         # 参考文献 (35MB, 不push)
    ├── TaskB/ (3篇)
    ├── TaskC/ (9篇)
    ├── TaskD/ (11篇)
    └── SBERT/ (6篇)
```

---

## 四个任务

### Task A — 数据获取

| 文件 | 作用 |
|---|---|
| `src/task_a_crawl_skills.py` | Batch 1 爬虫 (5,499 条, 列表分页 + 详情逐条) |
| `src/task_a_crawl_batch2.py` | Batch 2 增量爬虫 (5,002 条, 跳过已有, 从 page 111 开始) |

收集 10,501 条 skill，15 个字段：`name / description / actions / permissions / category / tags / risks / ...`

---

### Task B — 异常检测

**特征工程**：

| 方法 | 维度 | 文件 | 参考文献 |
|---|---|---|---|
| TF-IDF | 500/3000 维 | `build_tfidf()` in pipeline | [Qiu+ 2024] |
| SBERT (all-MiniLM-L6-v2) | 384 维 | `task_b_sbert_benchmark.py` | [TAD Survey 2024] |
| 结构化特征 | 45 维 | `src/features/structural_features.py` | [SkillSieve 2026] |

**检测方法**：

| 方法 | 类型 | 文件 | 参考文献 |
|---|---|---|---|
| Isolation Forest | 无监督 | `src/detection/isolation_forest.py` | [Liu+ ICDM 2008] |
| LOF | 无监督 | `src/detection/local_outlier_factor.py` | [Breunig+ SIGMOD 2000] |
| LSTM Autoencoder | 无监督 | `src/task_b_lstm_autoencoder.py` | [Coote+ 2024] |
| XGBoost | 有监督 | `src/detection/xgboost_classifier.py` | [Chen+ KDD 2016] |
| Ensemble (投票) | 集成 | `src/task_b_ensemble.py` | — |

**弱标注**：

| 方法 | 文件 | 说明 |
|---|---|---|
| 规则标注 (8条) | `src/labeling/rule_labeling.py` | 确定性, 基于结构化特征 |
| LLM 标注 | `src/utils/llm_labeling.py` | DeepSeek-chat, SkillSieve 四维分解 |

**核心结果**：

| 方法 | F1 |
|---|---|
| XGBoost (TF-IDF + Struct, CV) | **0.841** |
| Ensemble "Any" (IF+LOF+LSTM) | 0.118 |
| LOF (TF-IDF) | 0.095 |
| IF (TF-IDF) | 0.082 |
| LSTM Autoencoder | 0.072 |

**SBERT vs TF-IDF**：SBERT 三项全胜 (IF +11%, LOF +24%, XGB +8%)

---

### Task C — 对抗生成

**EASG (Evolutionary Adversarial Skill Generation) 框架**：

| 组件 | 文件 | 方法 |
|---|---|---|
| 6 个扰动算子 | `src/task_c_adversarial_gen.py` | 同义词替换, 指令膨胀, 权限混淆, 上下文注入, 风险降级, 结构模仿 |
| 多目标遗传算法 | 同上 | 种群 N=50, 20 代, Pareto 选择, 两点交叉 |

**核心结果**：100% 逃逸率 (初始 1.95/3 → 3.00/3), 语义相似度 0.749

**参考文献**：[AOP-Mal 2025], [GA Text Attack 2020], [ICLT 2025], [GAOR 2025]

---

### Task D — 去风险化

**KRI 复合风险评分**：

| 组件 | 文件 | 方法 |
|---|---|---|
| KRI 评分 | `src/task_d_derisking.py` | Threat×0.55 + Impact×0.25 + Exposure×0.20 − SafetyBonus |
| 7 个去风险算子 | 同上 | 危险标记移除, 安全确认, 权限最小化, 风险透明, 操作理由, 命令标记, 审计日志 |
| 权重校准 | `src/task_d_kri_calibration.py` | 网格搜索最优 w_t=0.55, w_i=0.25, w_e=0.20 |

**核心结果**：84% 降风险率, 平均 KRI Δ=0.046, 语义相似度 0.911

**参考文献**：[KRI Framework 2025], [SoK Debloating 2024], [Dual-Use GenAI 2025], [Semantic Elasticity 2025]

---

## 分支说明

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | 基础版本 | ✅ |
| `refactor/clean-architecture` | 模块化重构 | ▶️ |
| `feature/sbert-full-and-viz` | SBERT + 可视化 | ✅ |
| `feature/ensemble-and-kri` | 集成 + KRI校准 | ✅ |
| `feature/full-llm-labeling` | 全量 LLM 标注 | ▶️ 33% |

---

## 参考文献

1. **SkillSieve** — Hou & Yang, "A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills," 2026.
2. **Qiu et al.** — "Assessing the impact of bag-of-words versus word-to-vector embedding methods," *IJNM*, 2024.
3. **Coote & Lachine** — "Platform Management System Host-Based Anomaly Detection using TF-IDF and an LSTM Autoencoder," 2024.
4. **Liu et al.** — "Isolation Forest," *IEEE ICDM*, 2008.
5. **Breunig et al.** — "LOF: Identifying Density-Based Local Outliers," *ACM SIGMOD*, 2000.
6. **Chen & Guestrin** — "XGBoost: A Scalable Tree Boosting System," *ACM KDD*, 2016.
7. **AOP-Mal** — Xu et al., "Automatic optimization for generating adversarial malware based on prioritized evolutionary computing," *Applied Soft Computing*, 2025.
8. **GA Text Attack** — "Adversarial Black-Box Attacks On Text Classifiers Using Multi-Objective Genetic Optimization," 2020.
9. **ICLT** — "Chinese legal adversarial text generation based on interpretable perturbation strategies," *WWW*, 2025.
10. **GAOR** — "Genetic Algorithm-Based Optimization for Machine Learning Robustness," *Network*, 2025.
11. **Vuln Prioritization Survey** — Jiang et al., *arXiv:2502.11070*, 2025.
12. **KRI Framework** — "Bridging the Gap Between Security Metrics and Key Risk Indicators," 2025.
13. **SoK Debloating** — Alhanahnah et al., *FEAST*, 2024.
14. **Dual-Use GenAI** — Korimilli et al., *JISEM*, 2025.
15. **Semantic Elasticity** — De Tomasi et al., *EASE*, 2025.
16. **Red/Blue Teaming** — Abuadbda et al., *arXiv:2506.13434*, 2025.
17. **TAD Survey** — "Comparative analysis of anomaly detection algorithms in text data," 2024.

---

*详见 `output/Final_Report.md`*
