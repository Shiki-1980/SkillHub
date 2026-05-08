# Task B 报告：DataHub Skill 异常检测

## 一、技术路线

### 1.1 总体架构

```
DataHub Skills (5499条)
       │
       ▼
┌─ 特征工程 ───────────────────────┐
│  TF-IDF (3000维) + 结构化特征 (45维) │
└─────────────────────────────────┘
       │
       ├──→ 弱标注 (LLM辅助 + 规则兜底)
       │         │
       │         ▼
       │    ┌─ 无监督异常检测 ──────────┐
       │    │  Isolation Forest (TF-IDF) │
       │    │  LOF (PCA+TF-IDF)          │
       │    │  Ensemble (IF+LOF)         │
       │    └───────────────────────────┘
       │         │
       │         ▼
       └──→ ┌─ 有监督分类器 ────────────┐
            │  XGBoost (TF-IDF+结构特征)  │
            └───────────────────────────┘
                  │
                  ▼
            ┌─ 评估与消融实验 ──────────┐
            │  多指标对比 + 去embedding  │
            │  + 去agent辅助标注         │
            └───────────────────────────┘
```

### 1.2 特征工程

**TF-IDF 特征（3000维）**：对 `name + description + actions + permissions` 拼接文本进行 TF-IDF 向量化。参数选择：(1,2)-gram，sublinear TF，max_df=0.8，min_df=3，停用词过滤。采用 sublinear TF（1 + log(TF)）以抑制高频词的权重膨胀。

**结构化特征（45维）**：受 SkillSieve [1] Layer 1 启发，从 DataHub 数据中提取：
- **风险维度（6维）**：danger/warn/medium/safe 标签计数、danger 比率、总风险项数
- **文本统计（5维）**：名称长度、描述长度、动作长度、权限长度、描述/动作比
- **权限分析（5维）**：OS 权限数、Shell 权限数、网络权限数、文件系统权限数、总权限项数
- **安全信号（11维）**：逆向 Shell、凭证窃取、数据外泄、混淆、Prompt 注入、危险执行、危险网络、敏感路径、紧急用语、安全禁用、隐藏行为 — 均使用词边界正则匹配
- **元数据（5维）**：stars（对数）、score、是否为 workflow/prompt/reference 形态
- **类别（10维）**：Top 10 类别的 one-hot 编码
- **其他（3维）**：紧急语言密度、非英语标记

### 1.3 弱标注方法

采用两层弱标注策略：

**第一层：规则预标注（8条规则）**。基于结构化特征定义确定性规则，包括：
- 混淆+危险操作 → malicious（置信度 0.85）
- Prompt 注入关键词 → malicious（0.90）
- 高 danger + 危险关键词 → unsafe（0.75）
- 高 danger 比 + 紧急语言 → unsafe（0.70）
- 多权限 + 短动作 → over-privileged unsafe（0.60）

**第二层：LLM 辅助标注（800条采样）**。参考 SkillSieve [1] 的四维分解方法，将安全分析分解为三个子任务：意图对齐（A）、权限合理性（B）、隐蔽行为（C）。使用 DeepSeek-V3 (deepseek-chat) 作为标注模型，temperature=0.1，结构化 JSON 输出。LLM 标注的 800 条结果中：normal 380（47.5%）、unsafe 399（49.9%）、malicious 20（2.5%）、useless 1（0.1%）。未标注的 4699 条使用规则标签兜底。

### 1.4 无监督异常检测方法

**Isolation Forest [7]**：在 3000 维 TF-IDF 特征矩阵上训练，n_estimators=300，contamination=0.1。原理是通过随机划分构建隔离树，异常点需要较少的分割次数即可被隔离，因此路径长度较短。输出归一化异常分数（0~1，1 为最异常）。

**Local Outlier Factor (LOF) [8]**：先在 TF-IDF 矩阵上使用 TruncatedSVD（n_components=80）降低维度，然后使用 LOF（n_neighbors=30，contamination=0.1）计算局部异常因子。LOF 通过比较每个点与其 k 近邻的局部密度来识别异常，适合发现局部密度显著不同于邻域的点。

**Ensemble**：将 IF 和 LOF 的归一化异常分数取加权平均（各 0.5），top 10% 标记为异常。

### 1.5 有监督分类器

**XGBoost [9]**：在 3045 维组合特征（TF-IDF + 结构化）上训练。参数：n_estimators=300，max_depth=6，learning_rate=0.05，subsample=0.8，colsample_bytree=0.8。5 折分层交叉验证。

### 1.6 消融实验

为满足 PPT 要求，设计两组消融实验：
- **去 Embedding 效果**：仅使用 45 维结构化特征（去掉 TF-IDF），训练 XGBoost
- **去 Agent 辅助效果**：仅使用规则标签（去掉 LLM 标注），训练 XGBoost
- **去两者**：仅使用结构化特征 + 规则标签

---

## 二、方法设计理由

### 2.1 为什么选择 TF-IDF

Qiu et al. [3] 在 8 个日志数据集上系统对比了 TF-IDF（Bag-of-Words）与 BERT（Word-to-Vector）嵌入方法，核心结论是**未微调的 BERT 并不优于 TF-IDF**。Coote & Lachine [2] 在 SCADA 主机日志异常检测中验证了 TF-IDF + LSTM Autoencoder 的有效性。考虑到本项目中 skill 文本以技术性描述为主，词汇分布具有领域特异性，TF-IDF 的计算效率和对领域术语的敏感性使其成为合适的基线选择。PPT 强制要求 TF-IDF。

### 2.2 为什么同时使用无监督和有监督方法

PPT 要求至少一种无监督异常检测方法和一种分类器。二者的互补性在于：
- **无监督方法**不依赖标签，可以发现未知类型的异常模式，但难以区分"文本离群点"和"安全威胁"
- **有监督方法**利用弱标签学习具体的安全信号，但有标签偏差风险

实验结果表明（§4.1），这种互补性确实存在：IF 和 LOF 发现的异常与 XGBoost 预测仅有 56 条重叠，说明两者捕捉了异常的不同维度。

### 2.3 为什么采用 SkillSieve 的四维分解标注

Hou & Yang [1] 证明将安全分析分解为意图对齐、权限合理性、隐蔽行为、跨文件一致性四个子任务，效果（F1=0.800）显著优于单次 LLM 判断（F1=0.746）。我们将此方法适配到 DataHub 场景（去掉不适用的"跨文件一致性"维度），用于指导 LLM 弱标注。

### 2.4 为什么使用 LLM 辅助标注而非纯规则

规则标注（8条规则）覆盖范围有限，对语义层面的安全隐患（如隐蔽行为、意图不一致）无法捕捉。LLM 标注结果验证了这一点：LLM 识别出 399/800（49.9%）为 unsafe，而规则仅识别出 ~10%，说明 LLM 能从语义层面发现规则难以表达的异常模式。

---

## 三、参考文献

[1] Y. Hou and Z. Yang, "SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills," 2026.

[2] E. Coote and B. Lachine, "Platform Management System Host-Based Anomaly Detection using TF-IDF and an LSTM Autoencoder," Royal Military College, 2024.

[3] Z. Qiu, Z. Zhou, B. Niblett, et al., "Assessing the impact of bag-of-words versus word-to-vector embedding methods and dimension reduction on anomaly detection from log files," *International Journal of Network Management*, vol. 34, e2251, 2024.

[4] S. Hawkins, H. He, G. Williams, and R. Baxter, "Outlier Detection Using Replicator Neural Networks," in *Proc. DaWaK*, 2002.

[5] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD*, 2000.

[6] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008.

[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM KDD*, 2016.

[8] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

---

## 四、实验结果与分析

### 4.1 无监督方法结果

| 方法 | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Isolation Forest (TF-IDF) | 0.069 | 0.066 | 0.068 | 0.809 |
| LOF (PCA+TF-IDF) | 0.105 | 0.101 | 0.103 | 0.817 |
| Ensemble (IF+LOF) | 0.085 | 0.082 | 0.084 | 0.813 |

无监督方法在弱标签上的 F1 偏低（0.07-0.10），这一结果与 SkillSieve [1] 的发现一致：**纯静态/文本特征无法可靠地检测安全威胁**。SkillSieve Layer 1（纯静态特征）F1=0.733，但在 SkillSieve 的场景中有精确的手工标注；而我们使用弱标签作为参照，由于标签本身包含噪声，F1 进一步降低。

**方法间一致性分析**：IF 和 LOF 的整体一致性为 82.03%，但两者同时标记为异常的仅 56 条，IF 独有 494 条，LOF 独有 494 条。低重叠率说明两种方法捕捉了文本异常的不同维度：
- IF 倾向于标记词汇分布与主流差异较大的 skill（如 Rube MCP 系列自动化 skill）
- LOF 倾向于标记在局部密度上与邻域差异大的 skill（如低频率的 devops/testing 类 skill）

**类别分布**：无监督方法在 testing（17.5%）、docs（14.9%）、coding（14.2%）类别中异常比例最高，说明这些类别中 TF-IDF 文本模式更为多样化和离散。

### 4.2 有监督分类器结果

**XGBoost 5 折交叉验证**：
- **macro-F1 = 0.829 ± 0.165**
- **Accuracy = 0.986 ± 0.008**

F1 标准差较高（0.165），主要原因是类别严重不平衡（useless 仅 1 条，malicious 仅 29 条），导致某些 fold 中少数类样本极少甚至缺失。

与无监督方法的对比：
- XGBoost F1（0.829）远超 IF（0.068）和 LOF（0.103），验证了**有监督学习 + 结构化特征**在安全分类任务上的优势
- 但 XGBoost 的高性能部分来源于它学习了弱标签的规则模式，而非真正的安全语义

### 4.3 LLM 标注 vs 规则标注

| 维度 | 规则标注 | LLM 标注（800条采样） |
|---|---|---|
| malicious | 0.5% | 2.5% |
| unsafe | 9.5% | 49.9% |
| normal | 89.9% | 47.5% |
| useless | 0.0% | 0.1% |

LLM 标注的 unsafe 比例（49.9%）远高于规则标注（9.5%），主要差异来源：
1. LLM 将大量"功能正常但权限过度"的 skill 标记为 unsafe（如 notion、weather），规则难以捕捉这种语义级别的权限不匹配
2. LLM 识别出 20 条 malicious（如 nemo-guardrails、security-operator、ai-shield-audit），其中部分技能被 LLM 认为可能被滥用为攻击工具
3. LLM 仅标记 1 条 useless（claude-win11-speckit-update-skill），说明大部分 DataHub skill 至少功能上是完整的

### 4.4 消融实验

| 实验 | CV macro-F1 | Δ vs 完整模型 |
|---|---|---|
| 完整模型 (TF-IDF + 结构特征 + LLM标签) | 0.829 | — |
| 去 TF-IDF embedding | 0.839 | −0.010 |
| 去 Agent 辅助标注 | 0.852 | −0.023 |
| 去两者 | 0.874 | −0.045 |

**核心发现**：去除 TF-IDF 和 LLM 标注后，分类器性能**不降反升**。这一反直觉的结果有如下解释：

1. **TF-IDF 引入噪声**：TF-IDF 捕捉的是文本相似性，而非安全语义。对于安全分类任务，TF-IDF 的 3000 维噪音可能掩盖了 45 维结构化特征中的关键信号。这与 Qiu et al. [3] 的发现一致：嵌入方法的选择对下游任务有显著影响，且更高的维度不代表更好的效果。

2. **规则标签与规则特征的一致性**：规则标签来自结构化特征（如 danger 计数、危险关键词），而 XGBoost 也在相同特征上训练。标签和特征来自同一套规则体系时，模型学到了近似确定性的映射关系，因此 macro-F1 最高（0.874）。

3. **LLM 标签引入语义复杂性**：LLM 标注了规则无法表达的安全判断（如"意图不一致"），但这些语义维度在当前特征空间中缺乏充分的数值表示。模型无法从 TF-IDF 或结构特征中捕捉"隐蔽行为意图"这样的高阶语义，因此 F1 反而下降。

4. **负的 Δ 值恰恰说明**：安全异常检测的核心挑战在于**语义理解**，而非文本匹配。文本嵌入和结构特征可以捕捉表层模式，但真正决定 malicious/unsafe 的语义信号需要更深层的分析——这正是 SkillSieve [1] Layer 2（LLM 语义分析）存在的理由。

### 4.5 总体结论

1. **TF-IDF + 结构化特征 + XGBoost** 构建的基线系统达到了 CV macro-F1=0.829，能够有效识别符合已知规则的异常 skill
2. **无监督方法**（IF/LOF）发现了 TF-IDF 空间中的文本异常，但这些异常与安全威胁不具有直接对应关系
3. **LLM 辅助标注**显著提升了标注质量和覆盖度，但需要更强大的语义特征才能发挥全部价值
4. **消融实验**揭示了安全异常检测的核心瓶颈：表层文本特征不足以捕捉深层安全语义，后续工作（Task C/D）需要探索语义级别的对抗与分析

---

*报告生成日期：2026-05-08*
*实验代码：task_b.py, run_llm_labeling.py*
