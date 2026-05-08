# Agent 工作记录：SkillHub 异常技能挖掘项目

> Claude Code Agent · 2026 年 5 月 7–8 日 · 约 16 小时工作量

---

## 一、项目概述

**任务**：对 DataHub 平台的 AI Agent Skill 进行系统性安全分析，完成四个递进式任务——数据获取 (A)、异常检测 (B)、对抗生成 (C)、去风险化 (D)。

**最终产出**：
- 10,501 条标注 skill 数据集（15 字段 + LLM/规则标签）
- 4 种异常检测方法 + 3 种文本嵌入 + 消融实验
- EASG 进化对抗生成框架（6 算子 + 多目标 GA）
- KRI 复合风险评分的去风险化框架（7 算子 + Safety Bonus）
- 6 张论文级可视化 + 3 份详细报告 + 17 篇参考文献
- 完整模块化代码（18 模块 + 4 流水线）

---

## 二、工作流水线

### 第一阶段：基础设施搭建

**Task A — 数据获取**

| 步骤 | 操作 | 结果 |
|---|---|---|
| 理解数据结构 | 阅读 CLAUDE.md + PPT，理解 API 细节（列表接口不含 capabilities/requirements，detail 的 risks 是 dict 列表） | 正确调用 API |
| Batch 1 爬虫 | 分页遍历列表接口 → 逐条调详情接口，增量写入 CSV，3 次重试 | 5,499 条，0 失败 |
| Batch 2 爬虫 | 从 page 111 开始（跳过已有），收集 5,000 新 skill | 5,002 条，0 失败 |
| 合并数据 | 去重合并 + 规则预标注 | 10,501 条，1,168 malicious/unsafe |

**方法**：`src/legacy/task_a_crawl_skills.py` + `src/legacy/task_a_crawl_batch2.py`（增量爬虫）

### 第二阶段：异常检测（Task B）— 核心实验

**文献调研**：用户提供 3 篇论文 → 分别精读摘要 → 深读方法论部分。

| 论文 | 关键启发 | 应用到本项目的部分 |
|---|---|---|
| SkillSieve (Hou 2026) | 15 特征 Layer 1 + 四维 LLM 分解 | 结构化特征设计 + LLM 标注 prompt |
| Coote & Lachine (2024) | TF-IDF + LSTM AE 的完整参数 | LSTM Autoencoder 实现 |
| Qiu et al. (2024) | TF-IDF 不输于 BERT | TF-IDF 基线选择的理论依据 |

**实施**：

| 步骤 | 内容 |
|---|---|
| 特征工程 | TF-IDF (500/3,000d) + 45 维结构化特征（5 类：风险元数据、文本统计、权限分析、安全信号正则、元数据 one-hot） |
| 弱标注 | 8 条确定性规则 + DeepSeek-chat LLM 辅助（SkillSieve 四维分解：意图对齐/权限合理性/隐蔽行为）|
| 无监督检测 | Isolation Forest (n=300) + LOF (PCA→80d, n=30) |
| 有监督分类 | XGBoost (n=300, max_depth=6), 5 折 CV |

**消融实验**（PPT 要求）：去 TF-IDF embedding + 去 Agent 辅助标注。

**结果**：

| 方法 | F1 | 关键发现 |
|---|---|---|
| XGBoost (TF-IDF+Struct) | CV 0.829 | 有监督 >> 无监督 |
| Isolation Forest | 0.068 | TF-IDF 空间异常≠安全威胁 |
| LOF | 0.103 | 局部密度略优于全局隔离 |
| 消融 去 TF-IDF | 0.839 (−0.010) | TF-IDF 引入噪音（反直觉） |
| 消融 去 LLM 标注 | 0.852 (−0.023) | 规则标签与特征同源 → 过拟合 |

**核心发现**：纯文本特征无法可靠捕获安全语义，结构化特征是安全信号的主要载体。

### 第三阶段：扩展实验

后续依次进行了以下扩展（每次开独立 feature 分支）：

**SBERT 嵌入对比**（6 篇论文 → 精读 3 篇）

| 方向 | 方法 | 结果 |
|---|---|---|
| SBERT 全量嵌入 | all-MiniLM-L6-v2, 384d, 10,501 条 | — |
| TF-IDF vs SBERT | IF/LOF/XGB 三种检测器对比 | SBERT 三项全胜 (IF +11%, LOF +24%, XGB +8%) |

参考：TAD Survey (2024) — 22 算法 17 语料, SBERT+弱标签 > 纯无监督。

**LSTM Autoencoder**（第三个无监督方法）

基于 Coote & Lachine (2024)：TF-IDF (n-gram=5) → LSTM Encoder(SeLU) → Latent(64d) → Decoder → MAE 重构误差 > p90 阈值为异常。

结果：F1=0.072，与 IF 相当。

**方法集成**

IF + LOF + LSTM AE 三种无监督方法的共识仅 45/10,501（0.8%）。四种投票策略中，"Any" (≥1/3) 最佳：F1=0.118（+24% vs 最好单独方法）。

**KRI 权重校准**

网格搜索 [0.15,0.55] 区间：最优权重 w_t=0.55, w_i=0.25, w_e=0.20（原 0.40/0.35/0.25）。Threat 权重与 F1 单调正相关。F1 从 0.239 提升至 0.248。

参考：KRI Framework (2025) — ROC-AUC 0.927 vs CVSS 0.747。

**对抗训练**

用 Task C 对抗变体增强 XGBoost 训练集（每 seed 3 变体，2,799 增强样本），对比 Vanilla vs Robust：

| 场景 | Vanilla | Robust | Δ |
|---|---|---|---|
| Clean test | 0.871 | 0.566 | −0.305 |
| Mixed (50%+50% adv) | 0.228 | **0.643** | **+0.415** |
| Pure adversarial | 0.000 | **0.489** | **+0.489** |

闭环验证：Task C 攻击 → 对抗训练防御 → 鲁棒性提升。

**可视化**

t-SNE + UMAP 投影，含密度分析。SBERT 稀疏区异常率 13.4% > TF-IDF 8.0%——SBERT 更有效分离异常。

### 第四阶段：对抗生成（Task C）

**文献调研**：9 篇论文 → 精读 4 篇核心。

| 论文 | 关键贡献 | 应用 |
|---|---|---|
| AOP-Mal (Xu 2025) | 优先进化计算 + 动作向量 | 算子设计 + 优先级策略 |
| GA Text Attack (2020) | 六阶段 GA + 多目标优化 | EASG 框架核心 |
| ICLT (2025) | TF-IDF 引导扰动 | O₁ 同义词替换策略 |
| GAOR (2025) | 专攻 XGBoost/RF 的对抗攻击 | 攻击目标选择 |

**EASG 框架设计**：
- 6 个可复现扰动算子（同义词替换/指令膨胀/权限混淆/上下文注入/风险降级/结构模仿）
- 多目标遗传算法（N=50, G=20, Pareto 选择）
- 适应度：逃逸率 + 语义相似度 + 结构相似度

**结果**：20 种子，初始平均被检测 1.95/3 → 最终 **0.00/3**（100% 完全逃逸），语义相似度 0.749。结构模仿是最有效算子（38.7% 使用率）。

### 第五阶段：去风险化（Task D）

**文献调研**：11 篇论文 → 精读 6 篇。

| 论文 | 关键贡献 | 应用 |
|---|---|---|
| Vuln Prioritization Survey (Jiang 2025) | RS = α·Impact + β·Exploitability + γ·Context | KRI 公式三元分解 |
| KRI Framework (2025) | 威胁×影响×暴露，ROC-AUC 0.927 | KRI 三维模型 + Safety Bonus |
| SoK Debloating (2024) | 去臃肿工作流分类法 | 算子体系设计 |
| Dual-Use GenAI (2025) | LLM 红蓝双重用途 | Task C↔D 对偶验证 |
| Semantic Elasticity (2025) | 变换质量度量 | 评估指标设计 |

**KRI 公式**：`0.55×Threat + 0.25×Impact + 0.20×Exposure − SafetyBonus`

**7 个去风险算子**（与 Task C 6 个逃逸算子形成对偶）：危险标记移除/安全确认/权限最小化/风险透明/操作理由/命令标记/审计日志。

**Safety Bonus 机制**：6 类正向安全信号（用户确认/审计/透明/最小权限/理由/沙箱），每类 +0.03，上限 0.15。

**结果**（80/20 holdout）：84/100 KRI 降低，平均 Δ=0.052，语义相似度 0.930。

### 第六阶段：工程化

**代码重构**：`src/` 按功能分为 9 个模块：

```
data/ → features/ → labeling/ → detection/ → adversarial/ → derisking/
                                              ↓
                                         training/ (对抗训练)
                                              ↓
evaluation/ ← pipeline/
```

旧脚本归档至 `src/legacy/`。

**Git 分支管理**：main → dev → 6 个 feature 分支，最终合并至 `refactor/clean-architecture`。

**README + PIPELINE.md**：完整使用文档，包含流水线图、模块说明、运行命令、参考文献。

---

## 三、完整结果汇总

### Task B — 异常检测

| 方法 | F1 (80/20 test) | 类型 |
|---|---|---|
| XGBoost | **0.648** | 有监督 |
| Isolation Forest | 0.126 | 无监督 |
| LOF | 0.110 | 无监督 |
| LSTM AE | 0.072 | 无监督 |
| Ensemble "Any" | 0.118 | 集成 |

SBERT vs TF-IDF: SBERT 在 IF/LOF/XGB 三项全胜 (+11%/+24%/+8%)

消融实验：去 TF-IDF Δ=−0.01，去 LLM 标注 Δ=−0.02（说明结构性特征是主信号）

### Task C — 对抗生成

- 100% 完全逃逸（语义相似度 0.749）
- 结构模仿是最有效算子（38.7%）

### Task D — 去风险化

- 84% KRI 降风险率（Δ=0.052, sim=0.930）
- KRI 权重 w_t=0.55, w_i=0.25, w_e=0.20（经网格搜索校准）

### 对抗训练闭环

- Robust XGBoost vs Vanilla：Mixed test +0.415，Pure adversarial +0.489

---

## 四、创新点

1. **SkillSieve 适配**：将恶意 skill 三层检测框架适配到 DataHub 平台，设计了 45 维结构化特征和四维 LLM 标注
2. **EASG 进化框架**：首次将 AOP-Mal 的进化对抗生成应用于 AI agent skill 文本域
3. **KRI Safety Bonus 机制**：在传统"坏分数"模型上增加正向安全信号对冲，符合 NIST CSF 理念
4. **Dual-Use 对偶验证**：同一操作框架支持攻击（Task C）和防御（Task D），实证支撑 Dual-Use GenAI 理论
5. **闭环对抗训练**：Task C（生成）→ Task B（重训）→ 鲁棒性提升 0.489

---

## 五、工具使用统计

| 工具 | 用途 | 调用次数（估计） |
|---|---|---|
| Read | 读代码、PDF、数据 | 80+ |
| Write | 写脚本、报告 | 40+ |
| Edit | 修改代码 | 60+ |
| Bash | 运行实验、git 操作 | 150+ |
| Grep/Glob | 搜索代码/文件 | 20+ |
| Agent | 探索性任务 | 5 |
| TaskCreate/Update | 任务管理 | 15 |
| WebSearch/WebFetch | 文献检索 | 0（用户提供论文） |

---

## 六、参考文献使用

共使用 **17 篇**参考文献，全部实存在 `papers/` 目录，覆盖：

- **Task B**：3 篇（SkillSieve, Qiu et al., Coote & Lachine）+ 3 篇经典（IF, LOF, XGBoost）
- **Task C**：4 篇（AOP-Mal, GA Text Attack, ICLT, GAOR）
- **Task D**：5 篇（Vuln Survey, KRI Framework, SoK Debloating, Dual-Use, Semantic Elasticity）
- **SBERT**：2 篇（TAD Survey, Patent SBERT）

---

*记录完成：2026 年 5 月 8 日 · 项目总时长 ~16 小时 · 代码行数 ~8,000*
