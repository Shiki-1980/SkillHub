# SkillHub 流水线文档

## 完整流水线

```
                         DataHub API
                             │
              ┌──────────────┴──────────────┐
              │   src/legacy/task_a_crawl_*  │  ← Task A: 数据爬取
              └──────────────┬──────────────┘
                             │
                    data/skills_raw_merged.csv
                      (10,501 skills, 15 字段)
                             │
              ┌──────────────┴──────────────┐
              │     Weak Labeling 弱标注      │
              │  ├─ labeling/rule_labeling   │  8条确定性规则
              │  ├─ labeling/llm_labeling    │  DeepSeek-chat (SkillSieve四维分解)
              │  └─ utils/merge_labels       │  合并规则+LLM标签
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Feature Engineering      │
              │  ├─ TF-IDF (500/3000d)       │  (1,2)-gram, sublinear TF
              │  ├─ SBERT (384d)             │  all-MiniLM-L6-v2
              │  └─ Structural (45d)         │  SkillSieve Layer 1 启发
              └──────────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐        ┌──────────┐         ┌──────────┐
   │   IF    │        │   LOF    │         │LSTM AE   │  ← Task B: 无监督
   │ (密度)  │        │ (局部)   │         │ (重构)   │     异常检测
   └────┬────┘        └────┬─────┘         └────┬─────┘
        │                  │                    │
        └──────────────────┼────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Ensemble   │  ← "Any" 投票: F1=0.118
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  XGBoost    │  ← Task B: 有监督分类器
                    │  CV-F1 0.84 │
                    └──────┬──────┘
                           │
              ┌────────────┴──────────────┐
              │    Task C: 对抗生成        │
              │  ├─ 6 perturbation ops    │  同义词替换 / 指令膨胀 / 权限混淆
              │  ├─ EASG genetic algo     │  上下文注入 / 风险降级 / 结构模仿
              │  └─ Multi-obj fitness     │  逃逸率 + 语义相似度 + 结构相似度
              └────────────┬──────────────┘
                           │
              ┌────────────┴──────────────┐
              │    Task D: 去风险化        │
              │  ├─ KRI risk scoring      │  Threat×0.55 + Impact×0.25
              │  │                         │  + Exposure×0.20 - SafetyBonus
              │  └─ 7 derisking ops       │  危险标记移除 / 安全确认 / 权限最小化
              │                           │  风险透明 / 操作理由 / 命令标记 / 审计
              └───────────────────────────┘
```

---

## 模块地图

### `src/data/` — 数据层

| 文件 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `loader.py` | 加载 CSV, 拼接文本字段 | `data/skills_raw_merged.csv` | `DataFrame` with `text` 列 |

### `src/features/` — 特征工程

| 文件 | 方法 | 维度 | 原理 |
|---|---|---|---|
| `structural_features.py` | 结构化特征 | 45 | 风险标签计数 + 权限分析 + 安全信号正则 + 元数据 one-hot |

TF-IDF 和 SBERT 嵌入在各 pipeline 中按需生成，未单独模块化（调用 sklearn / sentence-transformers 各一行代码）。

### `src/labeling/` — 弱标注

| 文件 | 方法 | 参考文献 |
|---|---|---|
| `rule_labeling.py` | 8 条确定性规则 | 基于 SkillSieve Layer 1 启发式 |
| `utils/llm_labeling.py` | DeepSeek-chat 四维分解标注 | SkillSieve (Hou 2026) |

**四维分解**（SkillSieve 启发）：
- A: Intent Alignment — 声称 vs 实际指令是否一致
- B: Permission Justification — 权限是否合理
- C: Covert Behavior — 是否有隐蔽/绕过行为
- (D: Cross-File Consistency — 不适用于 DataHub)

### `src/detection/` — 异常检测模型

| 文件 | 方法 | 类型 | 参考文献 |
|---|---|---|---|
| `isolation_forest.py` | Isolation Forest | 无监督 | Liu+ ICDM 2008 |
| `local_outlier_factor.py` | LOF (PCA 降维) | 无监督 | Breunig+ SIGMOD 2000 |
| `xgboost_classifier.py` | XGBoost | 有监督 | Chen+ KDD 2016 |
| `legacy/task_b_lstm_autoencoder.py` | LSTM AE (TF-IDF→SeLU→MAE) | 无监督 | Coote+ 2024 |
| `legacy/task_b_ensemble.py` | 三种投票策略 | 集成 | — |

**检测方法对比**：

| 方法 | 原理 | F1 (vs 规则标签) | 10,501条标记数 |
|---|---|---|---|
| IF | 随机划分隔离树，路径短的为异常 | 0.082 | 1,050 |
| LOF | 局部密度 vs k-NN，密度低为异常 | 0.095 | 1,050 |
| LSTM AE | 重构误差 > p90 阈值为异常 | 0.072 | 983 |
| Ensemble "Any" | ≥1/3 投票 | **0.118** | 2,527 |
| XGBoost | 梯度提升树, 5-fold CV | **0.841** | — |

### `src/adversarial/` — 对抗生成 (Task C)

| 文件 | 内容 | 参考文献 |
|---|---|---|
| `perturbation_operators.py` | 6 个逃逸算子 + 词表 | AOP-Mal (Xu 2025), ICLT (2025) |
| `genetic_algorithm.py` | EASG 框架 (Individual + 多目标 GA) | GA Text Attack (2020) |

**6 个扰动算子** (全部基于确定性规则, 可复现)：

| 算子 | 操作 | 影响维度 |
|---|---|---|
| O₁ Synonym | TF-IDF 高权重词 → 风险降级词典 | 词汇层 |
| O₂ Expansion | 插入 benign 填充句, 稀释危险密度 | 结构层 |
| O₃ Permission Obfuscation | `[warn]→[ok]`, `[danger]→[warn]` | 权限层 |
| O₄ Context Injection | 注入正常 skill 描述或安全声明 | 语义层 |
| O₅ Risk Downgrade | 20 对风险词映射 (exploit→utilize 等) | 词汇层 |
| O₆ Structural Mimicry | 调整长度匹配 normal skill 分布 | 结构层 |

**遗传算法参数**：
- 种群 N=50, 代数 G=20, 精英率 10%
- 选择: Pareto 支配 + 锦标赛 (k=3)
- 交叉: 两点, 变异: 随机替换 (p=0.3) + 增删基因 (p=0.15)
- 适应度: f₁=逃逸率, f₂=语义相似度(TF-IDF cosine), f₃=结构相似度(长度比)

### `src/derisking/` — 去风险化 (Task D)

| 文件 | 内容 | 参考文献 |
|---|---|---|
| `kri_scorer.py` | KRI 三维风险评分 + Safety Bonus | Vuln Prioritization Survey (Jiang 2025), KRI Framework (2025) |
| `derisking_operators.py` | 7 个去风险算子 + 安全声明模板 | SoK Debloating (2024), Auto-SPT (2025) |

**KRI 公式**：
```
KRI = 0.55 × Threat + 0.25 × Impact + 0.20 × Exposure − SafetyBonus
```
权重通过网格搜索校准 (最优 F1=0.248, 原经验权重 F1=0.239)。

**7 个去风险算子**：

| 算子 | 操作 | 影响维度 |
|---|---|---|
| D₇ Remove Dangerous Flags | `[danger]→[warn]+mitigation`, URL 脱敏 | Threat ↓ |
| D₁ Safety Guard | 插入 `[SAFETY]`/`[CONFIRM]` 声明 | Safety ↑ |
| D₂ Permission Minimization | 权限范围缩小 + 最小权限声明 | Impact ↓ |
| D₃ Risk Transparency | 明确风险声明前缀 | Safety ↑ |
| D₄ Action Justification | 敏感操作附加 `[Purpose:]` 说明 | Safety ↑ |
| D₈ Command Sanitization | 危险命令附加确认标记 | Threat ↓ |
| D₆ Audit Trail | 操作日志声明 | Safety ↑ |

**Safety Bonus 机制**：6 类正向安全信号 (用户确认/审计/透明/最小权限/理由/沙箱), 每类 +0.03, 上限 0.15。

### `src/evaluation/` — 评估

| 文件 | 功能 |
|---|---|
| `metrics.py` | F1 / Precision / Recall / Accuracy + 一致性矩阵 |

### `src/pipeline/` — 流水线编排

| 文件 | 对应 Task | 一键执行 |
|---|---|---|
| `02_label_and_detect.py` | Task B | 标注→特征→检测 |
| `03_generate_adversarial.py` | Task C | 种子→GA→对抗样本 |
| `04_derisk.py` | Task D | 种子→去风险算子→KRI评分 |

### `src/legacy/` — 旧版脚本 (保留以兼容)

| 文件 | 对应新版模块 |
|---|---|
| `task_b_anomaly_detection.py` | `pipeline/02_label_and_detect.py` |
| `task_c_adversarial_gen.py` | `pipeline/03_generate_adversarial.py` |
| `task_d_derisking.py` | `pipeline/04_derisk.py` |
| `task_b_lstm_autoencoder.py` | `detection/` (待进一步模块化) |
| `task_b_sbert_benchmark.py` | 独立实验 |
| `task_b_ensemble.py` | 独立实验 |
| `task_d_kri_calibration.py` | 独立实验 |
| `visualization.py` | `evaluation/` (待进一步模块化) |
| `task_a_crawl_*.py` | 爬虫 (一次性) |

---

## 运行命令

```bash
# 核心流水线
python3 -m src.pipeline.02_label_and_detect     # Task B: 5 min
python3 -m src.pipeline.03_generate_adversarial  # Task C: ~30 min (100 seeds × 20 gen)
python3 -m src.pipeline.04_derisk                # Task D: 1 min

# 扩展实验
python3 src/legacy/task_b_lstm_autoencoder.py    # LSTM AE: 2 min
python3 src/legacy/task_b_sbert_benchmark.py     # SBERT vs TF-IDF: 3 min
python3 src/legacy/task_b_ensemble.py            # 方法集成: 3 min
python3 src/legacy/task_d_kri_calibration.py     # KRI权重校准: 1 min
python3 src/legacy/visualization.py              # 可视化: 3 min

# LLM 标注
python3 src/utils/llm_labeling.py --max 800      # DeepSeek-chat 标注
```

---

## 方法总览

| 类别 | 方法 | 数量 | 参考文献 |
|---|---|---|---|
| **特征** | TF-IDF, SBERT, 结构化特征 | 3 | [Qiu 2024], [TAD Survey 2024], [SkillSieve 2026] |
| **无监督检测** | IF, LOF, LSTM AE | 3 | [Liu 2008], [Breunig 2000], [Coote 2024] |
| **有监督检测** | XGBoost | 1 | [Chen 2016] |
| **集成** | Any/Majority/Unanimous/Weighted | 4 策略 | — |
| **弱标注** | 规则(8条) + LLM(DeepSeek-chat) | 2 | [SkillSieve 2026] |
| **对抗生成** | EASG (6 ops + GA) | 1 框架 | [AOP-Mal 2025], [GA Text 2020], [ICLT 2025] |
| **去风险化** | KRI评分 + 7 ops | 1 框架 | [KRI 2025], [SoK Debloating 2024], [Dual-Use 2025] |
| **评估** | F1/P/R/Acc + 一致性矩阵 | — | — |
| **可视化** | t-SNE, UMAP, 密度分析 | 3 | — |
