# DataHub 平台 AI Agent Skill 安全分析：异常检测、对抗生成与去风险化

> **期末项目报告 · 数据挖掘 · 2026 年 5 月**

---

## 摘要

AI Agent 技能（Skill）市场的开放生态带来了显著的供应链安全风险。OpenClaw 的 ClawHub 平台托管超过 13,000 个社区贡献的 skill，安全审计发现其中 13%–26% 包含安全漏洞。本研究以 DataHub Skills API 为数据源，采集了 5,499 条 AI agent skill，围绕四个任务展开系统性安全分析：(1) 数据获取与特征工程，(2) 基于 TF-IDF 与结构化特征的异常检测，(3) 基于进化算法的对抗 skill 生成，(4) 基于 KRI 复合风险评分的去风险化。实验结果表明，XGBoost 分类器在弱标签上达到 0.829 的宏平均 F1，进化式对抗生成框架实现了 100% 的检测逃逸率（平均语义相似度 0.749），去风险化框架降低了 84% 样本的 KRI 风险评分（平均降幅 0.046）。本研究为 AI agent skill 生态的安全分析提供了可复现的方法论框架，并揭示了纯文本特征在安全语义捕获上的根本性局限。

**关键词**：AI Agent 安全，异常检测，对抗生成，进化算法，风险评分，TF-IDF

---

## 1. 引言

### 1.1 研究背景

AI coding agent（如 Claude Code、Cursor、OpenClaw）通过 skill 机制扩展功能——skill 是包含自然语言指令和可选脚本的功能包。OpenClaw 的 ClawHub 市场截至 2026 年初托管超过 13,000 个 skill，日提交量在高峰期超过 500 个，且发布无需强制性安全审查 [1]。

2026 年 1–2 月的 ClawHavoc 攻击活动向 ClawHub 推送了数百个恶意 skill；Koi Security 对 2,857 个 skill 的审计发现 341 个恶意条目 [1]；Snyk 的 ToxicSkills 审计发现 13.4% 的 3,984 个 skill 包含至少一个严重级别的安全问题。攻击者使用域名仿冒、跨文件逻辑拆分和外部 webhook 凭据窃取等技术 [1]。

传统的检测方法各有局限：正则扫描器遗漏跨文件分段载荷，形式化静态分析器无法解析自然语言指令中的 prompt injection 和社交工程攻击，而基于单一 LLM 的分析缺乏多模型交叉验证机制 [1]。

### 1.2 研究目标

本研究以 DataHub Skills API 为数据源，完成四个递进式任务：

- **Task A**：从 DataHub 平台采集 skill 数据，构建分析数据集
- **Task B**：基于 TF-IDF 和无监督/有监督方法检测异常 skill
- **Task C**：基于进化算法生成能逃避检测的对抗 skill，同时保留风险语义
- **Task D**：构建去风险化框架，通过 KRI 风险评分和语义相似度双重指标评估改写效果

### 1.3 研究贡献

1. 构建了 5,499 条 skill 的标注数据集，结合 LLM 辅助弱标注和规则兜底的两层标注策略
2. 验证了 TF-IDF + 结构化特征 + XGBoost 在 skill 异常检测上的有效性（CV macro-F1=0.829）
3. 提出了 EASG 进化式对抗生成框架，以 100% 逃逸率生成对抗样本
4. 设计了 KRI 复合风险评分模型，实现了 84% 的降风险率
5. 通过消融实验揭示了纯文本特征在安全语义捕获上的根本性局限

---

## 2. Task A：数据获取

### 2.1 数据源与采集

数据来自 DataHub Skills API (`https://www.fudankw.cn/skills-api`)，采用三阶段流水线采集：

- **Phase 1** — 分页遍历列表接口 `GET /skills`，收集 skill ID（去重）
- **Phase 2** — 逐条调用详情接口 `GET /skills/{id}`，获取完整字段（`capabilities`、`requirements`、`risks` 等），增量写入 CSV
- **错误处理** — 3 次重试机制 + 进度输出

### 2.2 数据集概况

最终数据集包含 **5,499** 条 skill，15 个字段。核心字段包括 `name`、`description`、`actions`、`permissions`（Task A 要求），以及 `category`、`tags`、`risks`、`stars`、`score`、`form` 等辅助字段。风险信息以结构化标签 `[danger]`/`[warn]`/`[medium]`/`[safe]` 编码，权限信息包含 OS、Shell、网络、文件系统等多维度要求。

---

## 3. Task B：异常检测

### 3.1 方法设计

#### 3.1.1 特征工程

**TF-IDF 特征（3,000 维）**：对 `name + description + actions + permissions` 拼接文本进行 TF-IDF 向量化。参数选择：(1,2)-gram，sublinear TF（1 + log(TF)），max_df=0.8，min_df=3，停用词过滤。sublinear TF 抑制高频词权重膨胀，使中等频率的领域术语获得更合理的权重分配。

**结构化特征（45 维）**：受 SkillSieve [1] Layer 1 启发，涵盖五个维度：
- **风险维度（6 维）**：danger/warn/medium/safe 标签计数、danger 比率、总风险项数
- **文本统计（5 维）**：名称/描述/动作/权限长度、描述/动作比
- **权限分析（5 维）**：OS/Shell/网络/文件系统权限计数、总权限项数
- **安全信号（11 维）**：使用词边界正则表达式检测逆向 Shell、凭证窃取、数据外泄、混淆、Prompt 注入、危险执行命令（eval/exec/subprocess）、危险网络命令（curl/wget/nc）、敏感路径、紧急用语、安全禁用、隐藏行为
- **元数据（18 维）**：stars（对数）、score、形态 one-hot、Top 10 类别 one-hot、非英语标记、紧急语言密度

TF-IDF 的选择基于 Qiu et al. [2] 的系统对比研究——在 8 个日志数据集上，未微调的 BERT 并不优于 TF-IDF；Coote & Lachine [3] 也在 SCADA 异常检测中验证了 TF-IDF 的有效性。

#### 3.1.2 弱标注策略

采用两层弱标注策略：

**第一层：规则预标注（8 条规则）**。基于结构化特征定义确定性规则：
- 混淆 + 危险操作 → malicious（置信度 0.85）
- Prompt 注入关键词 → malicious（0.90）
- 高 danger + 危险关键词 → unsafe（0.75）
- 高 danger 比 + 紧急语言 → unsafe（0.70）
- 多权限 + 短动作 → over-privileged unsafe（0.60）

**第二层：LLM 辅助标注（800 条采样）**。参考 SkillSieve [1] 的四维分解方法，将安全分析分解为意图对齐、权限合理性、隐蔽行为三个子任务。使用 DeepSeek-V3（deepseek-chat）作为标注模型，temperature=0.1，结构化 JSON 输出。标注结果：unsafe 399（49.9%）、normal 380（47.5%）、malicious 20（2.5%）、useless 1（0.1%）。未标注的 4,699 条使用规则标签兜底。

LLM 标注采用 SkillSieve 的四维分解而非单次判断，因为 Hou & Yang [1] 证明四维分解的 F1（0.800）显著优于单次 LLM 判断（0.746）。

#### 3.1.3 无监督异常检测

**Isolation Forest [4]**：在 3,000 维 TF-IDF 特征矩阵上训练，n_estimators=300，contamination=0.1。原理是通过随机划分构建隔离树，异常点因密度低而需要较少的分割次数即可被隔离。

**Local Outlier Factor (LOF) [5]**：首先使用 TruncatedSVD（n_components=80）降维，然后使用 LOF（n_neighbors=30，contamination=0.1）计算局部异常因子。LOF 通过比较每个点与其 k 近邻的局部密度来识别异常。

**Ensemble**：IF 和 LOF 归一化异常分数的加权平均，top 10% 标记为异常。

#### 3.1.4 有监督分类器

**XGBoost [6]**：在 3,045 维组合特征（TF-IDF + 结构化）上训练。参数：n_estimators=300，max_depth=6，learning_rate=0.05，subsample=0.8，colsample_bytree=0.8。5 折分层交叉验证评估。

### 3.2 实验结果

#### 3.2.1 无监督方法

| 方法 | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Isolation Forest (TF-IDF) | 0.069 | 0.066 | 0.068 | 0.809 |
| LOF (PCA+TF-IDF) | 0.105 | 0.101 | 0.103 | 0.817 |
| Ensemble (IF+LOF) | 0.085 | 0.082 | 0.084 | 0.813 |

无监督方法 F1 偏低（0.07–0.10），与 SkillSieve [1] 的发现一致：**纯静态/文本特征无法可靠地检测安全威胁**。IF 和 LOF 之间仅 56 条共识异常，低重叠率说明两种方法捕捉了文本异常的不同维度：IF 偏向全局词汇分布离群点，LOF 偏向局部密度差异点。类别分布上，testing（17.5%）、docs（14.9%）、coding（14.2%）异常比例最高，说明这些类别的文本模式更为多样化。

#### 3.2.2 有监督分类器

XGBoost 5 折交叉验证：**macro-F1 = 0.829 ± 0.165**，Accuracy = 0.986 ± 0.008。F1 标准差较高（0.165）是因为类别严重不平衡（useless 仅 1 条，malicious 仅 29 条）。

#### 3.2.3 LLM 标注 vs 规则标注

| 维度 | 规则标注 | LLM 标注（800 条） |
|---|---|---|
| malicious | 0.5% | 2.5% |
| unsafe | 9.5% | 49.9% |
| normal | 89.9% | 47.5% |

LLM 标注识别了显著更多的 unsafe（49.9% vs 9.5%），主要因为 LLM 能从语义层面检测到"功能正常但权限过度"的不匹配和隐蔽行为意图——这些是规则难以表达的。

#### 3.2.4 消融实验

| 实验 | CV macro-F1 | Δ vs 完整模型 |
|---|---|---|
| 完整模型 (TF-IDF + 结构特征 + LLM标签) | 0.829 | — |
| 去 TF-IDF embedding | 0.839 | −0.010 |
| 去 Agent 辅助标注 | 0.852 | −0.023 |
| 去两者 | 0.874 | −0.045 |

**消融实验的反直觉结论**：去除 TF-IDF 和 LLM 标注后分类器性能不降反升。原因分析：

1. **TF-IDF 引入噪声**：TF-IDF 捕捉文本相似性而非安全语义，3,000 维噪音可能掩盖 45 维结构化特征中的关键安全信号。这与 Qiu et al. [2] 的发现一致。
2. **规则标签与规则特征的一致性**：规则标签源自结构化特征，XGBoost 在相同特征空间上几乎学到了确定性映射。
3. **LLM 标签引入语义复杂性**：LLM 标注了"意图不一致"等语义维度，但这些高阶语义在当前特征空间中缺乏充分的数值表示。
4. **核心启示**：负的 Δ 值恰恰说明，安全异常检测的核心挑战在于**语义理解**，而非文本匹配——这正是 SkillSieve [1] Layer 2（LLM 语义分析）存在的理由。

---

## 4. Task C：对抗生成

### 4.1 方法设计

#### 4.1.1 EASG 框架总览

Evolutionary Adversarial Skill Generation（EASG）框架受三项工作的启发：AOP-Mal [7] 的优先进化计算对抗恶意软件生成、GA Text Attack [8] 的多目标遗传优化文本对抗、ICLT [9] 的 TF-IDF 引导扰动策略。

#### 4.1.2 扰动算子（6 个）

| 算子 | 操作 | 理论依据 |
|---|---|---|
| O₁ 同义词替换 | TF-IDF 高权重词 → 风险降级词典替换 | ICLT [9] TF-IDF 引导扰动 |
| O₂ 指令膨胀 | 单步→多步，稀释危险关键词密度 | GA Text [8] 文本膨胀 |
| O₃ 权限混淆 | 正则替换 `[warn]→[ok]`、`[danger]→[warn]` | AOP-Mal [7] 标签操作 |
| O₄ 上下文注入 | 注入 benign skill 描述或安全声明 | GA Text [8] 语义保留 |
| O₅ 风险降级 | 20 对风险级词映射（danger→caution 等） | AOP-Mal [7] 特征模糊 |
| O₆ 结构模仿 | 调整文本长度匹配 normal skill 分布 | GAOR [10] 结构对抗 |

所有算子基于确定性规则（词表、正则、模板），确保**完全可复现**——满足 PPT"禁止仅 prompt 改写"的要求。

#### 4.1.3 多目标遗传算法

**编码方案**：个体基因是变长序列 `[(op_id, strength), ...]`，长度 1–4。随机初始化，算子 ID 从 {0,...,5} 均匀抽样，strength 从 [0.2, 0.6] 均匀抽样。

**适应度函数**（最大化）：
- f₁ = 逃逸率（IF/LOF/XGBoost 三检测器中被欺骗的比例）
- f₂ = 语义相似度（TF-IDF 余弦相似度）
- f₃ = 结构相似度（文本长度比率）

**选择**：基于 Pareto 支配的锦标赛选择（k=3）。**交叉**：两点交叉。**变异**：以 0.3 概率随机替换基因，以 0.15 概率增删基因。**精英保留**：top 10%。

参数：种群 N=50，代数 G=20。

#### 4.1.4 种子选择

从 Task B 标注为 malicious/unsafe 的 573 条 skill 中，预筛选至少被 1 个检测器标记的 skill（86 条），取前 20 条作为种子。此筛选确保对抗生成有明确的可衡量目标。

### 4.2 实验结果

| 指标 | 值 |
|---|---|
| 种子数 | 20 |
| 初始平均被检测数 | 1.95/3 |
| 最终平均被检测数 | **0.00/3** |
| 逃逸改进率 | **20/20 (100%)** |
| 完全逃逸率 | **20/20 (100%)** |
| 平均语义相似度 | **0.749** |
| 平均 Δ 逃逸 | **1.05** |

#### 4.2.1 算子贡献度

GA 自然选择中算子被采纳的频率（优胜劣汰的结果）：

| 算子 | 次数 | 占比 | 有效性 |
|---|---|---|---|
| O₆ 结构模仿 | 24 | 38.7% | ★★★★★ |
| O₃ 权限混淆 | 13 | 21.0% | ★★★★ |
| O₄ 上下文注入 | 8 | 12.9% | ★★★ |
| O₂ 指令膨胀 | 7 | 11.3% | ★★★ |
| O₁ 同义词替换 | 6 | 9.7% | ★★ |
| O₅ 风险降级 | 4 | 6.5% | ★★ |

O₆ 是最有效的算子（38.7%），说明检测器对文本长度/结构高度敏感。O₃ 排第二（21.0%），说明权限标签 `[warn]`/`[danger]` 是 XGBoost 的重要分类特征。这一发现与 GAOR [10] 针对树模型的对抗攻击研究一致——结构特征往往是模型最依赖也最脆弱的信号。

#### 4.2.2 典型案例

- **`cloud-penetration-testing`**（unsafe → 完全逃逸）：初始 2/3 → 最终 3/3，sem_sim=0.743。策略：O₄ 上下文注入 + O₃ 权限混淆
- **`Memory Forensics`**（malicious → 完全逃逸）：初始 2/3 → 最终 3/3，sem_sim=0.730。策略：O₆ 结构模仿 + O₂ 指令膨胀 + O₁ 同义词替换

#### 4.2.3 检测器脆弱性

IF 和 LOF 对文本变化高度敏感——19/20 的 variant 逃脱了 IF/LOF 检测。这验证了 SkillSieve [1] 的论点：纯文本检测器的鲁棒性不足，需要多层次的语义分析。

---

## 5. Task D：去风险化

### 5.1 方法设计

#### 5.1.1 KRI 复合风险评分模型

基于 Vulnerability Prioritization Survey [11] 的风险计算公式和 KRI Framework [12] 的三维模型：

**KRI = 0.40·Threat + 0.35·Impact + 0.25·Exposure − SafetyBonus**

- **Threat（威胁）**：10 类危险模式匹配（reverse shell、credential theft、prompt injection 等）+ danger 标签计数的加权分
- **Impact（影响）**：6 类权限范围检测（OS/Shell/Network/Filesystem/Env/External），文本与权限元数据双重评估
- **Exposure（暴露）**：4 类攻击面指标（URL 引用、外部依赖、用户数据处理、多步骤操作）
- **Safety Bonus**：6 类正向安全信号（用户确认、审计日志、风险透明、最小权限、操作理由、沙箱），每类 +0.03，上限 0.15

传统安全评估（CVSS、OWASP）只计算"坏分数"，KRI Framework [12] 证明 CVSS 在真实漏洞上的 AUPRC 仅为 0.011（接近随机猜测）。我们的 KRI 模型通过 Safety Bonus 机制量化了 NIST CSF 框架中 Protection 和 Detection 能力对风险的对冲效应。

#### 5.1.2 去风险化算子（7 个）

基于 SoK Debloating [13] 的系统化去臃肿分类法和 Auto-SPT [14] 的语义保持变换方法：

| 算子 | 操作 | 影响维度 |
|---|---|---|
| D₇ Remove Dangerous Flags | `[danger]→[warn] + mitigation`，URL 脱敏 | Threat ↓ |
| D₁ Safety Guard Insertion | 插入 `[SAFETY]`/`[CONFIRM]` 用户确认声明 | Safety ↑ |
| D₂ Permission Minimization | 权限范围缩小 + 最小权限原则声明 | Impact ↓ |
| D₃ Risk Transparency | 明确风险声明前缀 | Safety ↑ |
| D₄ Action Justification | 敏感操作附加 `[Purpose:]` 理由说明 | Safety ↑ |
| D₈ Command Sanitization | 危险命令附加 `[EXECUTED ONLY AFTER USER CONFIRMATION]` | Threat ↓ |
| D₆ Audit Trail Addition | 操作审计日志声明 | Safety ↑ |

#### 5.1.3 语义弹性指标

基于 Simplicity by Obfuscation [15] 的 Semantic Elasticity 概念：

**Semantic Elasticity = Semantic Similarity / (1 − Length Ratio)**

该指标衡量变换的"质量效率"——在多大文本变化量下保持了多少语义。

#### 5.1.4 对偶验证设计

基于 Dual-Use GenAI [16] 和 Red/Blue Teaming [17] 的攻防对偶分析框架：

- Task C（红队视角）：通过逃逸算子降低检测率 → MITRE ATT&CK 框架
- Task D（蓝队视角）：通过去风险算子降低风险评分 → NIST CSF 框架

同一操作框架同时服务于攻击和防御，实证支撑了 [16] 的核心论点——"LLM 既是红队工具也是蓝队工具"。

### 5.2 实验结果

| 指标 | 值 |
|---|---|
| 种子数 | 25 |
| KRI 降低率 | **21/25 (84.0%)** |
| 平均 KRI delta | **0.0464** |
| 最大 KRI delta | **0.0917** |
| 风险等级降级率 | **8/25 (32%)** |
| 平均语义相似度 | **0.9114** |
| 平均语义弹性 | 4.834 |

#### 5.2.1 风险等级迁移

| 迁移路径 | 数量 |
|---|---|
| critical → high | 2 |
| high → medium | 4 |
| medium → low | 1 |
| low → minimal | 1 |

另有 11 条在 high 区间内评分下降但未触及 medium 边界（KRI 阈值 0.50）。

#### 5.2.2 Top 5 去风险案例

| Skill | KRI 前→后 | Δ | sim | 主要算子 |
|---|---|---|---|---|
| auth-implementation-patterns | 0.512→0.420 | 0.092 | 0.882 | D₇,D₁,D₈ |
| OpenClaw Host Hardening | 0.478→0.389 | 0.090 | 0.945 | D₇,D₃,D₆ |
| feishu-perm | 0.497→0.407 | 0.090 | 0.865 | D₂,D₇,D₁ |
| code-reviewer | 0.583→0.493 | 0.090 | 0.852 | D₇,D₄,D₈ |
| GitHub Triage | 0.595→0.505 | 0.090 | 0.919 | D₇,D₃,D₁ |

D₇（Remove Dangerous Flags）是所有 Top 5 的公共算子，直接修改了 `[danger]` 标签，对 KRI 的 threat 和 exposure 维度产生双重降低效果。

#### 5.2.3 与 Task C 对偶交叉

对 9 条有 Task C 对抗变体的 skill 同时应用去风险化：

| 维度 | Task C 对抗版 | Task D 去风险版 |
|---|---|---|
| 目标 | 逃避检测 | 降低风险 |
| KRI 变化方向 | 风险增加或持平 | **风险降低** |
| 平均 KRI delta | +0.022 | **+0.046** |
| 语义相似度 | 0.749 | **0.911** |

Task D 的语义相似度（0.911）显著高于 Task C（0.749），因为去风险化追加安全文本而非替换核心内容。

#### 5.2.4 检测器不敏感现象

检测率变化极小（1/25）。这一反直觉结果验证了 Dual-Use GenAI [16] 的核心发现：

- **当前检测器对安全声明的增加不敏感**——它们训练于区分 normal vs malicious/unsafe，而非评估安全措施的充分性
- **KRI 评分能捕捉安全信号的增加**——因为它显式建模了 safety bonus 维度
- **检测器脆弱性与风险评分在安全评估中互补**

---

## 6. 总体讨论

### 6.1 四个任务的逻辑链条

```
Task A (数据) → Task B (检测) → Task C (攻击) → Task D (防御)
     │               │               │               │
  5,499条        IF+LOF+        进化对抗生成     KRI去风险化
  skill数据      XGBoost         6个逃逸算子      7个去风险算子
                 macro-F1        100%逃逸率       84%降风险率
                 0.829           sim=0.749       sim=0.911
```

这个任务链条构成了一个完整的**安全分析生命周期**：从数据基础到检测能力建设，再到攻击面评估和防御措施验证。

### 6.2 关键发现

1. **文本特征 ≠ 安全语义**：消融实验（Task B §3.2.4）证明，纯 TF-IDF 特征对安全分类的贡献有限甚至是负面的。安全信号主要存在于结构化特征（风险标签、权限模式、关键词密度）中。

2. **进化算法是有效的对抗生成工具**：EASG（Task C）用 6 个可复现的确定性算子实现了 100% 检测逃逸，验证了 AOP-Mal [7] 和 GA Text Attack [8] 的方法在 skill 安全领域的适用性。

3. **Safety Bonus 机制是风险评分的关键创新**：传统框架（CVSS、OWASP）只评估"坏分数"，我们的 KRI 模型（Task D）通过审计、确认、最小权限等正向信号实现了更全面的风险评估，更符合 NIST CSF 的安全评估理念。

4. **攻防对偶验证是方法论贡献**：Task C 和 Task D 共享相同的操作框架但服务于相反的目标，实证支撑了 Dual-Use GenAI [16] 的理论。

### 6.3 局限性

1. **弱标签质量**：LLM 标注仅限于 800 条（14.5%），其余 85.5% 使用规则兜底。弱标签的噪声可能影响了有监督方法的真实性能评估。
2. **检测器设计**：IF 和 LOF 在纯 TF-IDF 空间上运行，对语义变换不敏感。引入语义嵌入（如 sentence embeddings）可能改善无监督方法的性能。
3. **对抗评估范围**：Task C 仅在 20 条种子上验证，需要在更大规模上评估 EASG 的泛化能力。
4. **KRI 权重**：KRI 三维权重（0.40/0.35/0.25）和 Safety Bonus 上限（0.15）是经验设定的，需要在实际安全事件数据上校准。

### 6.4 未来工作

1. **语义特征增强**：结合 sentence embeddings（SBERT）构建能捕获语义安全信号的新型特征
2. **多模态检测**：将结构化特征与 LLM 语义分析融合，参考 SkillSieve [1] 的多层架构
3. **KRI 校准**：收集真实安全事件数据，对 KRI 权重和 Safety Bonus 参数进行统计校准
4. **自动化红蓝对抗**：将 EASG 和去风险化框架整合为自动化的红蓝对抗循环

---

## 参考文献

[1] Y. Hou and Z. Yang, "SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills," 2026.

[2] Z. Qiu, Z. Zhou, B. Niblett, et al., "Assessing the impact of bag-of-words versus word-to-vector embedding methods and dimension reduction on anomaly detection from log files," *International Journal of Network Management*, vol. 34, e2251, 2024.

[3] E. Coote and B. Lachine, "Platform Management System Host-Based Anomaly Detection using TF-IDF and an LSTM Autoencoder," Royal Military College, 2024.

[4] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE International Conference on Data Mining (ICDM)*, 2008, pp. 413–422.

[5] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD International Conference on Management of Data*, 2000, pp. 93–104.

[6] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[7] Y. Xu, Y. Fang, Y. Xu, and Z. Wang, "Automatic optimization for generating adversarial malware based on prioritized evolutionary computing," *Applied Soft Computing*, vol. 173, 112933, 2025.

[8] "Adversarial Black-Box Attacks On Text Classifiers Using Multi-Objective Genetic Optimization Guided By Deep Networks," 2020.

[9] "Chinese legal adversarial text generation based on interpretable perturbation strategies," *World Wide Web*, vol. 28, no. 24, 2025.

[10] "GAOR: Genetic Algorithm-Based Optimization for Machine Learning Robustness in Communication Networks," *Network*, vol. 5, no. 6, 2025.

[11] Y. Jiang et al., "A Survey on Vulnerability Prioritization: Taxonomy, Metrics, and Research Challenges," *arXiv:2502.11070*, 2025.

[12] "Bridging the Gap Between Security Metrics and Key Risk Indicators: An Empirical Framework for Vulnerability Prioritization," 2025.

[13] M. Alhanahnah, Y. Boshmaf, and A. Gehani, "SoK: Software Debloating Landscape and Future Directions," in *Proc. FEAST*, 2024.

[14] "Auto-SPT: Automating Semantic Preserving Transformations for Code," 2025.

[15] L. De Tomasi, C. Di Sipio, A. Di Marco, and P. T. Nguyen, "Simplicity by Obfuscation: Evaluating LLM-Driven Code Transformation with Semantic Elasticity," in *Proc. EASE*, 2025.

[16] S. K. Korimilli et al., "Dual-Use of Generative AI in Cybersecurity: Balancing Offensive Threats and Defensive Capabilities in the Post-LLM Era," *Journal of Information Systems Engineering and Management*, vol. 10, no. 56s, 2025.

[17] A. Abuadbda et al., "From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs," *arXiv:2506.13434*, 2025.

---

*报告完成日期：2026 年 5 月 8 日*
*实验代码：src/task_a_crawl_skills.py, src/task_b_anomaly_detection.py, src/utils/llm_labeling.py, src/task_c_adversarial_gen.py, src/task_d_derisking.py*
*数据集：data/skills_raw.csv（5,499 条 skill）*
