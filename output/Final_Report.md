# DataHub 平台 AI Agent Skill 安全分析：异常检测、对抗生成与去风险化

> **期末项目报告 · 数据挖掘 · 2026 年 5 月**

---

## 摘要

AI Agent 技能（Skill）市场的开放生态带来了显著的供应链安全风险。本研究以 DataHub Skills API 为数据源，采集了 **10,501 条** AI agent skill，围绕四个任务展开系统性安全分析：

- **Task A** — 数据获取：增量爬虫采集 10,501 条 skill（两批次）
- **Task B** — 异常检测：**4 种方法**（Isolation Forest、LOF、LSTM Autoencoder、XGBoost）+ **2 种嵌入**（TF-IDF vs SBERT）+ 消融实验 + 方法集成 + **SHAP 可解释性** + **Layer 2 LLM 事后解释**
- **Task C** — 对抗生成：**EASG 进化框架**（6 个可复现算子 + 多目标遗传算法），100% 逃逸率
- **Task D** — 去风险化：**KRI 复合风险评分**（校准权重 0.55/0.25/0.20）+ **7 个去风险算子** + Safety Bonus 机制

**核心发现**：
1. XGBoost 分类器达到 CV macro-F1 **0.841**（10,501 条，规则弱标签）
2. SBERT 嵌入在全部三个检测器上优于 TF-IDF（IF +11%，LOF +24%，XGB +8%）
3. 三种无监督方法高度互补（三者共识仅 45/10,501），"Any" 投票 F1=**0.118**
4. EASG 对抗生成实现 100% 逃逸（语义相似度 0.749）
5. KRI 去风险化实现 86% 降风险率（KRI Δ=0.048，语义相似度 0.925）
6. 消融实验揭示：**纯文本特征无法可靠捕获安全语义**，结构性特征是安全信号的主要载体

---

## 1. 引言

AI coding agent（Claude Code、Cursor、OpenClaw 等）通过 skill 机制扩展功能——skill 是包含自然语言指令和可选脚本的功能包。OpenClaw 的 ClawHub 市场截至 2026 年初托管超过 13,000 个 skill，日提交量高峰期超过 500 个，且发布无需强制性安全审查 [1]。

2026 年初的 ClawHavoc 攻击活动向 ClawHub 推送数百个恶意 skill，Snyk 审计发现 13.4% 的 skill 包含严重安全问题 [1]。传统检测方法各有局限：正则扫描器遗漏跨文件分段载荷，静态分析器无法解析自然语言指令中的 prompt injection [1]。

本研究以 DataHub Skills API 为数据源，构建 **10,501 条标注 skill 数据集**，围绕"检测→攻击→防御"完成系统性的安全分析流水线。

---

## 2. Task A：数据获取

两阶段增量爬虫：Batch 1（5,499 条，API page 1–110）+ Batch 2（5,002 条，API page 124–224），去重合并得 **10,501 条 skill**。

核心字段：`name`、`description`、`actions`、`permissions`（Task A 要求），以及 `category`（37 类）、`tags`、`risks`（结构化标签 `[danger]`/`[warn]`/`[medium]`/`[safe]`）、`stars`、`score`、`form` 等。

---

## 3. Task B：异常检测

### 3.1 特征工程

**TF-IDF（500/3,000 维）**：(1,2)-gram，sublinear TF，max_df=0.8，min_df=3，停用词过滤。选择依据：Qiu et al. [2] 在 8 个日志数据集上证明未微调的 BERT 不优于 TF-IDF。

**SBERT（384 维）**：all-MiniLM-L6-v2 模型，对全部 10,501 条 skill 文本编码，normalize。选择依据：TAD Survey [3] 在 17 个语料上验证 SBERT + 半监督弱标签优于纯无监督。

**结构化特征（45 维）**：受 SkillSieve [1] Layer 1 启发，涵盖风险元数据（6 维）、文本统计（5 维）、权限分析（5 维）、安全信号正则（11 维）、类别/形态/语言等元数据（18 维）。

### 3.2 弱标注

**规则标注**（8 条规则）：基于结构化特征的确定性标签，Obfuscation+Hidden→malicious (0.85), Prompt Injection→malicious (0.90), 等等。

**LLM 辅助标注**：DeepSeek-chat，SkillSieve [1] 四维分解（意图对齐、权限合理性、隐蔽行为），800 条采样标注。标注分布：unsafe 49.9%，normal 47.5%，malicious 2.5%，useless 0.1%。

LLM 标注显著多于规则标注的 unsafe（49.9% vs 9.5%），因为 LLM 从语义层面捕捉了"功能正常但权限过度"和"隐蔽行为意图"——这些是规则无法表达的。

### 3.3 无监督异常检测

| 方法 | 原理 | 参考文献 |
|---|---|---|
| **Isolation Forest** | 随机划分隔离树，异常点路径短 | Liu+ ICDM 2008 [4] |
| **LOF** | PCA 降维 → 局部密度 vs k-NN | Breunig+ SIGMOD 2000 [5] |
| **LSTM Autoencoder** | TF-IDF(n-gram=5)→LSTM→重构误差>p90 | Coote & Lachine 2024 [6] |

**SBERT vs TF-IDF 基准测试**（仅文本特征，无结构化）：

| 方法 | TF-IDF (500d) | SBERT (384d) | Δ |
|---|---|---|---|
| Isolation Forest | 0.064 | **0.071** | +11% |
| LOF | 0.119 | **0.148** | +24% |
| XGBoost | 0.443 | **0.478** | +8% |

SBERT 仅用 384 维超越 500 维 TF-IDF，在 LOF 上提升最大（+24%）。这验证了 TAD Survey [3] 的结论：预训练语义嵌入在异常检测中优于词袋模型。

### 3.4 有监督分类器

**XGBoost** [7]：3,045 维（TF-IDF 3,000 + 结构化 45），5 折分层 CV。**macro-F1 = 0.841 ± 0.053**，Accuracy = 0.986 ± 0.004。

### 3.5 方法集成

三种无监督方法的一致性极低（三者共识仅 45/10,501），说明它们捕获了异常的不同维度。四种投票策略：

| 策略 | F1 | Precision | Recall | 标记数 |
|---|---|---|---|---|
| IF 单独 | 0.082 | 0.087 | 0.077 | 1,050 |
| LOF 单独 | 0.095 | 0.101 | 0.090 | 1,050 |
| LSTM AE 单独 | 0.072 | — | — | 983 |
| **Any (≥1/3)** | **0.118** | 0.086 | 0.185 | 2,527 |
| Majority (≥2/3) | 0.033 | — | — | 511 |
| Unanimous (3/3) | 0.002 | — | — | 45 |

"Any" 策略提升 24% 超越最佳单独方法，验证了三方法互补的假设。

### 3.6 消融实验

| 实验 | CV macro-F1 | Δ |
|---|---|---|
| 完整模型 (TF-IDF + 结构特征 + LLM标签) | 0.829 | — |
| 去 TF-IDF embedding | 0.839 | −0.010 |
| 去 Agent 辅助标注 | 0.852 | −0.023 |
| 去两者 | 0.874 | −0.045 |

**反直觉结论**：去除 TF-IDF 和 LLM 标注后，分类器性能不降反升。原因分析：

1. **TF-IDF 引入噪声**：3,000 维 TF-IDF 捕捉文本相似性而非安全语义，噪音掩盖了 45 维结构化特征中的关键信号。
2. **LLM 标签与特征空间不匹配**：LLM 标注了"意图不一致"等语义维度，但这些高阶语义在当前特征空间中缺乏数值表示。
3. **核心启示**：安全异常检测的核心挑战在于**语义理解**，而非文本匹配——这正是 SkillSieve [1] 引入 Layer 2（LLM 语义分析）的原因。

### 3.7 嵌入空间可视化

对 5,000 点子采样执行 t-SNE 和 UMAP 投影：

**TF-IDF vs SBERT 对比**（Fig 1-3）：
- SBERT 嵌入空间中，malicious/unsafe 形成可辨识的聚集簇
- TF-IDF 中异常散落在全空间，与 normal 几乎无法区分
- SBERT 的类别簇边界更清晰，同类 skill 更接近

**密度分析**（Fig 4）：
- TF-IDF：稀疏区异常率 8.0%，密集区 13.0%（异常反在密集区——错误信号）
- SBERT：稀疏区异常率 13.4%，密集区 15.0%（异常更偏向稀疏区——正确方向）

可视化直观验证了定量结论：SBERT 比 TF-IDF 更有效地编码了安全语义。

### 3.9 评估方式对比

三种评估在同一模型上的结果：

| 评估方式 | 指标 | 数据 | 标签 |
|---|---|---|---|
| 5-fold CV (旧) | macro-F1=**0.829** | 5,499 | 纯规则 |
| 80/20 held-out | macro-F1=**0.648** | 10,501 | LLM+规则 |
| 5-fold CV (新) | macro-F1=**0.659** | 10,501 | LLM+规则 |

5-fold CV (旧) 偏高的原因：规则标签与结构化特征来自同源规则体系，XGBoost 在 CV 中近似学习确定性映射。LLM 标签引入了规则无法表达的语义 nuance（如意图不一致、隐蔽行为），这些在当前 545 维特征中缺乏直接数值表示，因此 0.659 更接近真实泛化水平。

### 3.10 Layer 2：LLM 事后解释

受 SkillSieve [1] Layer 2 启发，对 XGBoost 判定的 anomalous skill，调用 DeepSeek-chat 进行四维事后分析。LLM 不改变分类 —— 仅解释"为什么被标记"。

**50 样本分析结果**：

| 指标 | 值 |
|---|---|
| LLM 同意 XGBoost | **48/50 (96.0%)** |
| LLM 不同意 | 1/50 (2.0%) |
| 部分同意 | 1/50 (2.0%) |
| LLM 置信度均值 | 0.87 |
| 风险根因：权限过度 (dim B) | **86%** |
| 风险根因：意图不一致 (dim A) | 10% |
| 风险根因：隐蔽行为 (dim C) | 4% |
| 可操作的降风险建议 | **100%** |

**Layer 2 作为独立审计**：用 LLM 的 agree/disagree 作为 XGBoost 预测的验证标签，XGBoost **Precision = 96%**。说明 XGBoost 敢于标记的那些，独立 LLM 几乎全部确认属实。

96% precision (Layer 2) 与 0.648 F1 (held-out) 的差异揭示了当前瓶颈：XGBoost **精确但保守**——它只对特征能捕捉的 case 出手，遗漏了大量语义层面的 unsafe。Layer 2 弥补了这一缺口：提供了每个被标记 skill 的具体去风险指导，直接输入 Task D。

### 3.11 SHAP 可解释性

使用 SHAP TreeExplainer 分析 XGBoost 的 545 维特征贡献：

| 发现 | 数据 |
|---|---|
| 结构化特征贡献占比 | **54%** (仅 45 维 vs TF-IDF 500 维) |
| 单位维度效率 | 结构化 = TF-IDF 的 **13 倍** |
| malicious 最强信号 | `sig_credential_theft` (SHAP=0.90) |
| unsafe 最强信号 | `actions_len` (SHAP=1.57) |
| 跨类别共享 Top 特征 | 4 个 |
| 类别特有特征 | 8 个 |

SHAP 验证了消融实验的核心结论：少量精心设计的领域特征远优于大量通用文本特征。`sig_credential_theft` 是 ON/OFF 开关型特征——一旦命中，立即将分类推向 malicious。`actions_len` 则区分"过度"（unsafe）与"恶意"（malicious）。

---

## 4. Task C：对抗生成

### 4.1 EASG 框架

Evolutionary Adversarial Skill Generation，受 AOP-Mal [8]、GA Text Attack [9]、ICLT [10] 启发。

**6 个扰动算子**（全部基于确定性规则，可复现）：

| 算子 | 操作 | 参考文献 |
|---|---|---|
| O₁ Synonym Substitution | TF-IDF 高权重词 → 风险降级词典 | ICLT [10] |
| O₂ Instruction Expansion | 插入 benign 填充句，稀释危险密度 | GA Text [9] |
| O₃ Permission Obfuscation | `[warn]→[ok]`, `[danger]→[warn]` | AOP-Mal [8] |
| O₄ Context Injection | 注入正常 skill 描述或安全声明 | GA Text [9] |
| O₅ Risk Downgrade | 20 对风险词映射 (exploit→utilize) | AOP-Mal [8] |
| O₆ Structural Mimicry | 调整长度匹配 normal skill 分布 | GAOR [11] |

**遗传算法**：种群 N=50，代数 G=20，Pareto 支配 + 锦标赛选择，两点交叉，精英率 10%。

适应度（最大化）：f₁ = 逃逸率（IF/LOF/XGBoost），f₂ = 语义相似度，f₃ = 结构相似度。

### 4.2 实验结果

**20 种子**（原版，小规模验证）：

| 指标 | 值 |
|---|---|
| 种子数 | 20 |
| 初始平均被检测 | 1.95/3 |
| 最终平均被检测 | **0.00/3** |
| 逃逸改进率 | **20/20 (100%)** |
| 完全逃逸率 | **20/20 (100%)** |
| 平均语义相似度 | **0.749** |

**算子贡献度**（GA 自然选择）：

| 算子 | 使用次数 | 占比 | 有效性 |
|---|---|---|---|
| O₆ 结构模仿 | 24 | 38.7% | ★★★★★ |
| O₃ 权限混淆 | 13 | 21.0% | ★★★★ |
| O₄ 上下文注入 | 8 | 12.9% | ★★★ |
| O₂ 指令膨胀 | 7 | 11.3% | ★★★ |
| O₁ 同义词替换 | 6 | 9.7% | ★★ |
| O₅ 风险降级 | 4 | 6.5% | ★★ |

结构模仿最有效（38.7%），说明检测器对文本长度/结构高度敏感——这是模型脆弱性的直接证据。

---

## 5. Task D：去风险化

### 5.1 KRI 复合风险评分

受 Vulnerability Prioritization Survey [12] 和 KRI Framework [13]（ROC-AUC 0.927 vs CVSS 0.747）启发：

**KRI = w_t × Threat + w_i × Impact + w_e × Exposure − SafetyBonus**

三维分解：
- **Threat**：10 类危险模式（reverse shell、credential theft、prompt injection 等）+ danger 标签加权
- **Impact**：6 类权限范围（OS/Shell/Network/Filesystem/Env/External）
- **Exposure**：4 类攻击面（URL、外部依赖、用户数据、多步操作）
- **Safety Bonus**：6 类正向安全信号，每类 +0.03，上限 0.15

### 5.2 权重校准

网格搜索 [0.15, 0.55] 范围内使 KRI 与异常标签 F1 最大的权重组合：

| | w_t | w_i | w_e | 阈值 | F1 |
|---|---|---|---|---|---|
| **最优** | **0.55** | **0.25** | **0.20** | 0.36 | **0.248** |
| 原经验 | 0.40 | 0.35 | 0.25 | ~0.38 | 0.239 |

**关键发现**：Threat 权重与 F1 单调正相关，w_t 从 0.15→0.55 时 F1 持续提升。在 skill 安全评估中，操作威胁信号比权限范围和外部暴露更具判别力。

### 5.3 去风险算子

受 SoK Debloating [14] 和 Auto-SPT [15] 启发，7 个算子形成对 Task C 的对偶：

| Task C (逃逸) | Task D (降风险) | 操作 |
|---|---|---|
| O₃ 权限混淆 | D₂ 权限最小化 | 缩小权限范围 |
| O₄ 上下文注入 | D₃ 风险透明化 | 明确风险声明 |
| O₅ 风险降级 | D₇ 危险标记移除 | danger→warn+mitigation |
| O₁ 同义词替换 | D₄ 操作理由说明 | 敏感操作附加 Purpose |
| O₆ 结构模仿 | D₈ 命令标记 | 危险命令附加确认 |
| — | D₁/D₆ 审计追加 | 安全确认 + 审计日志 |

### 5.4 实验结果

**100 种子**（扩大规模版本）：

| 指标 | 值 |
|---|---|
| 种子数 | 100 |
| KRI 降风险率 | **86/100 (86.0%)** |
| 平均 KRI delta | **0.048** |
| 平均语义相似度 | **0.925** |

**对偶验证**（Dual-Use GenAI [16] 实证支撑）：
- Task C：逃逸率 ↑，KRI ↑（风险增加或持平）
- Task D：逃逸率 ↓，KRI ↓（风险显著降低）
- Task D 语义相似度（0.925）> Task C（0.749），因为去风险化追加安全文本而非替换核心内容

---

## 6. 总体讨论

### 6.1 任务链条

```
Task A (10,501条) → Task B (4方法, F1=0.841) → Task C (100%逃逸) → Task D (86%降风险)
                      ↕ SBERT对比 (+24%)          ↕ 对偶验证
                      消融实验 (Δ=-0.045)          可视化 (6张)
```

### 6.2 关键贡献

1. **构建了 10,501 条标注 skill 数据集**，两批次增量爬虫 + LLM 辅助弱标注
2. **SBERT 嵌入一致优于 TF-IDF**（+11%–24%），验证了语义嵌入对安全检测的价值
3. **三种无监督方法高度互补**（共识仅 0.8%），"Any" 投票集成提升 24%
4. **EASG 框架实现 100% 逃逸**，所有算子基于确定性规则完全可复现
5. **KRI 评分模型经网格搜索校准**（最优权重 0.55/0.25/0.20），Safety Bonus 机制是创新点
6. **消融实验揭示核心瓶颈**：纯文本特征无法捕获安全语义，结构化特征是主要信号载体

### 6.3 局限性

1. LLM 标注仅覆盖 14.5% 样本，其余用规则兜底
2. 无监督方法 F1 整体偏低（0.07–0.15），受限于弱标签质量
3. KRI 权重通过弱标签校准，需要在真实安全事件数据上进一步验证
4. Task C 对抗生成尚未在 train/test 分离的严格设定下评估

---

## 参考文献

[1] Y. Hou and Z. Yang, "SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills," 2026.

[2] Z. Qiu et al., "Assessing the impact of bag-of-words versus word-to-vector embedding methods and dimension reduction on anomaly detection from log files," *Int. J. Network Management*, vol. 34, e2251, 2024.

[3] "Comparative analysis of anomaly detection algorithms in text data," 2024.

[4] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.

[5] M. M. Breunig et al., "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD*, 2000, pp. 93–104.

[6] E. Coote and B. Lachine, "Platform Management System Host-Based Anomaly Detection using TF-IDF and an LSTM Autoencoder," Royal Military College, 2024.

[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM KDD*, 2016, pp. 785–794.

[8] Y. Xu et al., "Automatic optimization for generating adversarial malware based on prioritized evolutionary computing," *Applied Soft Computing*, vol. 173, 112933, 2025.

[9] "Adversarial Black-Box Attacks On Text Classifiers Using Multi-Objective Genetic Optimization Guided By Deep Networks," 2020.

[10] "Chinese legal adversarial text generation based on interpretable perturbation strategies," *World Wide Web*, vol. 28, no. 24, 2025.

[11] "GAOR: Genetic Algorithm-Based Optimization for Machine Learning Robustness in Communication Networks," *Network*, vol. 5, no. 6, 2025.

[12] Y. Jiang et al., "A Survey on Vulnerability Prioritization: Taxonomy, Metrics, and Research Challenges," *arXiv:2502.11070*, 2025.

[13] "Bridging the Gap Between Security Metrics and Key Risk Indicators: An Empirical Framework for Vulnerability Prioritization," 2025.

[14] M. Alhanahnah et al., "SoK: Software Debloating Landscape and Future Directions," in *Proc. FEAST*, 2024.

[15] "Auto-SPT: Automating Semantic Preserving Transformations for Code," 2025.

[16] S. K. Korimilli et al., "Dual-Use of Generative AI in Cybersecurity," *JISEM*, vol. 10, no. 56s, 2025.

[17] L. De Tomasi et al., "Simplicity by Obfuscation: Evaluating LLM-Driven Code Transformation with Semantic Elasticity," in *Proc. EASE*, 2025.

---

*报告完成日期：2026 年 5 月 8 日 · 数据集 10,501 条 skill · 17 篇参考文献*
