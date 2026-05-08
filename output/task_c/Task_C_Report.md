# Task C 报告：进化式对抗 Skill 生成 (EASG)

## 一、技术路线

### 1.1 总体架构

```
Task B 识别出的 anomalous skills (29 malicious + 544 unsafe, 至少被1个检测器标记)
       │
       ▼
┌─ Phase 1: 扰动算子池 (6个) ──────────────────────────┐
│ O₁ 同义词替换   TF-IDF高权重词 → 风险降级词典替换      │
│ O₂ 指令膨胀     单步→多步, 稀释危险关键词密度          │
│ O₃ 权限混淆     重构 [warn]/[danger] 标签模式          │
│ O₄ 上下文注入   注入 benign skill 描述片段或安全声明    │
│ O₅ 风险降级     danger→caution 等软性措辞替换         │
│ O₆ 结构模仿     调整文本长度以匹配正常skill分布         │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌─ Phase 2: 多目标遗传算法 ────────────────────────────┐
│                                                        │
│  个体: 基因 = [(算子ID, 强度), ...]                    │
│        variant_text = 依次应用基因中的算子             │
│                                                        │
│  适应度 (最大化):                                      │
│    f₁ = 逃逸率 (3个检测器中被欺骗的比例)               │
│    f₂ = 语义相似度 (TF-IDF余弦相似度)                  │
│    f₃ = 结构相似度 (长度比率)                          │
│                                                        │
│  ┌── 初始化种群 (N=50) ──┐                            │
│  │         │              │                            │
│  │    ┌─ 适应度评估 ─┐    │                            │
│  │    │ IF+LOF+XGBoost │  │                            │
│  │    │  集成判断      │  │                            │
│  │    └──────────────┘    │                            │
│  │         │              │                            │
│  │    ┌─ 锦标赛选择 ─┐    │  × G=20 代                │
│  │    │ Pareto支配判断 │   │                            │
│  │    └──────────────┘    │                            │
│  │         │              │                            │
│  │    ┌─ 两点交叉 ───┐    │                            │
│  │    └─ 基因突变 ───┘    │                            │
│  │         │              │                            │
│  └── 新一代种群 ─────────┘                            │
│                                                        │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌─ Phase 3: 评估 ──────────────────────────────────────┐
│  改写前后逐条对照: IF/LOF/XGBoost检测率变化            │
│  语义保留: TF-IDF cosine similarity                   │
│  结构保留: 长度比率                                    │
│  算子贡献度: GA自然选择的算子使用频率                   │
└──────────────────────────────────────────────────────┘
```

### 1.2 扰动算子详细设计

**O₁ 同义词替换**：先用 TF-IDF 计算原文中各词的权重，按权重降序选择 top-N 词（比例由 strength 参数控制，默认 0.3），然后将匹配词替换为`RISK_DOWNGRADE_MAP`中的软性替代词（如 `exploit→utilize`、`exfiltrate→transfer`）。此方法借鉴 ICLT [4] 的 TF-IDF 引导扰动策略。

**O₂ 指令膨胀**：将文本按句分割，在随机位置插入 benign 填充句（如 "Check if input parameters are valid"、"Log the operation for auditing purposes"）。膨胀因子由 strength 控制，默认 0.3（插入原句数 30% 的填充句）。

**O₃ 权限混淆**：正则表达式模式匹配替换。将 `[warn]→[ok]`、`[danger]→[warn]`，并将具体权限描述泛化（如 "需要 bash, curl, python" → "需要 standard shell access"）。

**O₄ 上下文注入**：从两类来源注入 benign 上下文 — (a) 10 条预定义安全声明模板（如 "This skill follows best practices for security and transparency"），(b) 从 Task B 正常 skill 描述中随机采样首句。注入位置随机（开头/结尾各 50%）。

**O₅ 风险降级**：使用 20 对风险级词映射词典，将 security-sensitive 词替换为中性措辞。例如：`danger→caution`、`urgent→timely`、`bypass→streamline`、`backdoor→alternate access`。

**O₆ 结构模仿**：调整文本长度以接近正常 skill 的统计分布（描述 ~157 chars，动作 ~256 chars）。过长文本截断，过短文本追加安全声明。

### 1.3 遗传算法设计

**编码方案**：个体基因是一个变长序列 `[(op_id₁, strength₁), (op_id₂, strength₂), ...]`，序列长度 1-4。随机初始化时，每个基因的算子 ID 从 {0,...,5} 均匀抽样，strength 从 [0.2, 0.6] 均匀抽样。

**适应度函数（多目标最大化）**：
- *f₁ 逃逸率*：对每个 variant，调用 Task B 三个检测器（IF、LOF、XGBoost），计算未被检测出的比例。f₁ ∈ [0, 1]，1 表示完全逃逸
- *f₂ 语义相似度*：variant 与原文本的 TF-IDF 余弦相似度，约束对抗样本不偏离原意
- *f₃ 结构相似度*：variant 与原文本的长度比（min/max），约束文本长度不剧烈变化

**选择算子**：基于 Pareto 支配关系的锦标赛选择（k=3）。若双方互不支配，则按 f₁（逃逸率）决胜。

**交叉算子**：两点交叉。在父代基因序列上随机选两个切点，交换中间片段。子代基因长度截断至 1-5。

**变异算子**：对基因序列中的每个 (op_id, strength) 以 0.3 概率随机替换；以 0.15 概率随机增删一个基因。

**精英保留**：每代保留 top 10%（按逃逸率排序）的个体直接进入下一代。

### 1.4 种子选择策略

从 Task B 标注为 malicious 或 unsafe 的 573 条 skill 中，预筛选出被至少 1 个检测器（IF/LOF/XGBoost）标记为异常的 skill。筛选结果：86 条被 1 个检测器标记，3 条被 2 个检测器标记。取前 20 条作为 GA 种子。此筛选确保对抗生成有明确的"初始检测 → 逃逸"变化可衡量。

---

## 二、方法设计理由

### 2.1 为什么使用进化算法

Xu et al. [1] 在对抗恶意软件生成中证明，进化计算可以通过动作优先级和稀疏优化，在巨大的扰动空间中高效搜索。与基于梯度的方法（FGSM、C&W [6]）不同，进化算法：
- **不依赖模型梯度**：适用于黑盒场景，只需目标模型的输出标签即可工作
- **天然支持多目标优化**：通过 Pareto 支配可以同时优化逃逸率、语义相似度、结构相似度
- **搜索多样性**：种群机制避免了贪心策略陷入局部最优的问题

GA Text Attack [2] 进一步验证了遗传算法在文本对抗生成中的有效性：在 SST 和 IMDB 数据集上分别达到 65.7% 和 36.5% 的攻击成功率，94% 的对抗样本人类不可区分。

### 2.2 为什么使用多目标优化

单目标（仅最大化逃逸率）会导致极端变异——例如把所有词替换为随机字符虽然能逃逸检测，但语义完全破坏。多目标 Pareto 优化在逃逸率和语义保留之间寻找均衡解，确保生成的对抗样本同时满足：
- **逃避检测**（对下游安全系统构成真实威胁）
- **保留含义**（证明修改并非简单的破坏性操作）
- **保持可读**（经过改写的 skill 仍然看起来合法）

### 2.3 为什么算子基于规则而非 LLM

PPT 要求"生成策略必须可复现（迭代/规则/进化），禁止仅 prompt 改写"。我们的 6 个扰动算子全部基于确定性规则：
- O₁ 和 O₅ 基于预定义词表（`RISK_DOWNGRADE_MAP`）
- O₂ 基于填充句模板
- O₃ 基于正则表达式
- O₄ 基于预定义安全声明
- O₆ 基于统计阈值

这确保了**任何人使用相同种子和参数都能复现完全相同的对抗样本**。LLM prompt 改写由于模型版本和温度的随机性无法保证这一点。

### 2.4 为什么选择这些种子

GAOR [3] 表明，针对树模型（XGBoost、RF）的对抗攻击最有效的场景是目标样本**初始被检测**。如果种子本身已经逃逸了所有检测器（如 Task B 中 484/573 的 unsafe skill 未被任何检测器检测），对抗生成没有意义。因此我们预筛选了至少被 1 个检测器标记的 skill，确保每个种子都有明确的逃逸目标。

### 2.5 GA 超参数选择

- **种群大小 N=50**：参考 GA Text Attack [2] 的设置，在搜索空间覆盖度和计算开销间取得平衡
- **代数 G=20**：AOP-Mal [1] 使用 50 代，考虑到我们的算子空间较小（6 个离散算子 vs 大量连续动作），20 代已足够收敛
- **变异率 0.3**：避免过早收敛，同时防止随机搜索化
- **精英率 10%**：保留最优解，防止退化

---

## 三、参考文献

[1] Y. Xu, Y. Fang, Y. Xu, and Z. Wang, "Automatic optimization for generating adversarial malware based on prioritized evolutionary computing," *Applied Soft Computing*, vol. 173, 112933, 2025.

[2] "Adversarial Black-Box Attacks On Text Classifiers Using Multi-Objective Genetic Optimization Guided By Deep Networks," 2020.

[3] "GAOR: Genetic Algorithm-Based Optimization for Machine Learning Robustness in Communication Networks," *Network*, vol. 5, no. 6, 2025.

[4] "Chinese legal adversarial text generation based on interpretable perturbation strategies," *World Wide Web*, vol. 28, no. 24, 2025.

[5] "A Survey on Adversarial Attacks for Malware Analysis," 2024.

[6] N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks," in *Proc. IEEE Symposium on Security and Privacy*, 2017.

[7] C. Szegedy, W. Zaremba, I. Sutskever, et al., "Intriguing properties of neural networks," in *Proc. ICLR*, 2014.

[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008.

[9] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD*, 2000.

[10] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM KDD*, 2016.

---

## 四、实验结果与分析

### 4.1 总体对抗效果

| 指标 | 值 |
|---|---|
| 种子数 | 20 |
| 初始平均被检测数 | 1.95/3 检测器 |
| 最终平均被检测数 | **0.00/3 检测器** |
| 逃逸改进率 | **20/20 (100%)** |
| 完全逃逸率 | **20/20 (100%)** |
| 平均语义相似度 | **0.749** |
| 平均结构相似度 | 0.887 |

所有 20 个种子均实现了完全逃逸（3/3 检测器均不再标记为异常），同时保持平均 0.749 的语义相似度。这表明 EASG 框架能有效生成既逃避检测、又保留原意的对抗 skill。

### 4.2 语义相似度分析

语义相似度的分布范围为 0.589-0.991，中位数 0.76。低语义相似度（<0.65）的样本集中在种子技能文本较短的情况下——因为短文本对少量扰动更敏感，cosine 相似度下降更快。

**语义保留最好**（sim > 0.9）：
- `gdpr-data-handling`：sim=0.991，通过 O₂ 指令膨胀 + O₄ 上下文注入实现，原文本语义完整保留
- `Web3 Smart Contract Testing`：sim=0.981，O₃ 权限混淆 + O₆ 结构模仿

**语义保留最差**（sim < 0.65）：
- `ethical-hacking-methodology`：sim=0.589，O₆ 截断了大量原文本
- `Remote Browser Automation`：sim=0.632，O₄ 注入了较多上下文

### 4.3 算子贡献度分析

GA 自然选择中各算子被采纳的次数：

| 算子 | 使用次数 | 占比 | 有效性评级 |
|---|---|---|---|
| O₆ 结构模仿 | 24 | 38.7% | ★★★★★ |
| O₃ 权限混淆 | 13 | 21.0% | ★★★★ |
| O₄ 上下文注入 | 8 | 12.9% | ★★★ |
| O₂ 指令膨胀 | 7 | 11.3% | ★★★ |
| O₁ 同义词替换 | 6 | 9.7% | ★★ |
| O₅ 风险降级 | 4 | 6.5% | ★★ |

**O₆ 结构模仿是最有效的算子**（24/62 = 38.7%）。这说明 Task B 检测器对文本长度/结构敏感——调整 skill 长度使其接近正常 skill 的统计特征（数百字符的中等长度）即可有效绕过检测。这一发现与对抗攻击的常见模式一致：**结构性特征往往是分类器最依赖也最脆弱的信号**。

**O₃ 权限混淆排第二**（21.0%），说明权限标签（[warn]/[danger]/[ok]）是 XGBoost 的重要特征，改变这些标签能直接影响分类决策。

**O₅ 风险降级使用最少**（6.5%），可能原因是风险级词在 TF-IDF 向量中的权重不够高，替换后对整体向量影响较小。

### 4.4 典型案例分析

**Case 1: `cloud-penetration-testing`（unsafe → 完全逃逸）**
- 初始检测：2/3（被 IF 和 XGBoost 检测）
- 最终逃逸：3/3（语义相似度 0.743）
- 策略：O₄ 上下文注入 "Designed for legitimate development and testing purposes only" + O₃ 权限混淆
- 分析：原文本中 "penetration-testing" 和一系列网络操作触发了 IF 的异常检测和 XGBoost 的分类。通过注入合法用途声明和调整权限标签，成功使三个检测器均将其识别为 normal。

**Case 2: `Memory Forensics`（malicious → 完全逃逸）**
- 初始检测：2/3
- 最终逃逸：3/3（语义相似度 0.730）
- 策略：O₆ 结构模仿 + O₂ 指令膨胀 + O₁ 同义词替换
- 分析：这是本次实验中唯一的 malicious 标签种子。GA 通过三个算子的组合，调整文本长度 + 插入填充句 + 替换敏感词，实现了对三个检测器的完全逃逸。

**Case 3: `Secrets Management`（unsafe → 完全逃逸）**
- 初始检测：2/3
- 最终逃逸：3/3（语义相似度 0.782）
- 策略：O₆ 结构模仿 + O₄ 上下文注入
- 分析："secrets" 相关技能天然被 XGBoost 高概率分类为 unsafe，但通过加入 "All operations are logged and can be reviewed by the user" 等声明降低了风险感知。

### 4.5 检测器脆弱性分析

| 检测器 | 初始检测率（20种子） | 最终检测率 | 被逃逸次数 |
|---|---|---|---|
| Isolation Forest | 95% | 0% | 19 |
| LOF | 95% | 0% | 19 |
| XGBoost | 5% | 0% | 1 |

IF 和 LOF 对文本变化高度敏感——几乎所有经过修改的 variant 都能逃脱 IF/LOF 检测。这一方面说明 IF/LOF 在 TF-IDF 空间中有明显的脆弱性，另一方面也说明这 20 个种子中的异常信号主要集中在 TF-IDF 特征上。

XGBoost 仅被 1 个种子触发（5%），说明当前弱标签体系下 XGBoost 对 unsafe 类别的区分能力有限。但在消融实验中（Task B §4.4），XGBoost 对规则标签的 unsafe 类别有很高的识别率——差异在于 LLM 标注的 unsafe 包含大量"语义不安全但文本正常"的 skill，XGBoost 缺乏捕获这些语义信号的输入特征。

### 4.6 总体结论

1. **EASG 框架成功生成对抗 skill**：20/20 种子完全逃逸 3 个检测器，平均语义相似度 0.749
2. **结构模仿是最有效的扰动策略**：表明当前检测器过度依赖表面文本特征，对结构化扰动鲁棒性不足
3. **进化算法在多目标对抗生成中表现出色**：Pareto 优化自然地平衡了逃逸率和语义保留
4. **所有算子基于确定性规则**：满足 PPT"生成策略必须可复现"的要求，随机种子固定后结果完全可复现
5. **IF 和 LOF 对文本扰动的脆弱性**高于 XGBoost，但三者在组合扰动面前都难以有效防御

---

*报告生成日期：2026-05-08*
*实验代码：task_c.py*
*对抗样本数据：output/task_c/adversarial_results.json*
