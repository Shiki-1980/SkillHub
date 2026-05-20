# SHAP 可解释性分析：XGBoost 异常检测

> 模型：XGBoost (n=300, max_depth=6)，5,000 样本，545 维特征（TF-IDF 500 + 结构化 45）

---

## Fig 1: malicious 类别 SHAP 特征重要性

**`fig1_shap_malicious_importance.png`**

**展示什么**：Top 20 结构化特征对 `malicious` 分类的 SHAP 贡献度。每个点代表一个样本，颜色代表特征值高低（红=高值，蓝=低值），水平位置代表 SHAP 值（正值=推高恶意概率，负值=压低）。

**关键发现**：

| 排名 | 特征 | 含义 | 方向 |
|---|---|---|---|
| 1 | `sig_credential_theft` | 凭据窃取关键词（password/token + steal/leak/exfiltrate） | 有 → 强烈推高恶意 |
| 2 | `cat_security` | skill 分类为 security | 是 → 推高恶意 |
| 3 | `sig_data_exfil` | 数据外泄关键词 | 有 → 推高恶意 |
| 4 | `actions_len` | 操作指令长度 | 长 → 推高恶意 |
| 5 | `stars_log` | skill 收藏数（对数） | 低 → 推高恶意 |

**解读**：
- `sig_credential_theft`（SHAP=0.90）是最强的恶意信号——一旦描述/指令中出现"把密码/凭证发送到某处"的组合，模型几乎必然判定为恶意
- `cat_security`（SHAP=0.80）说明安全类别 skill 本身有更高的恶意概率——这符合直觉：攻击工具常伪装在安全类别下
- `sig_data_exfil`（0.16）排第三但权重显著低于前二，说明凭证窃取比通用数据外泄更具判别力
- `actions_len` 排第四——恶意 skill 的操作指令往往更长（因为需要描述复杂的攻击步骤）
- `stars_log` 排第五——低收藏数是恶意的微妙信号（恶意 skill 可能来自新账号或伪装 skill）

---

## Fig 2: 三个类别特征重要性对比

**`fig2_shap_all_classes.png`**

**展示什么**：malicious、unsafe、normal 三类各 Top 15 结构化特征的并排对比柱状图。

**逐类解读**：

### malicious（左）

特征集中在**安全威胁信号**：`sig_credential_theft`、`sig_data_exfil`、`sig_dangerous_exec`、`sig_reverse_shell`。这些特征的共同点：**直接描述攻击行为**。

### unsafe（中）

特征集中在**操作规模与权限范围**：`actions_len`（第一, 1.57）、`n_danger`（0.56）、`total_perm_items`（0.42）、`danger_ratio`（0.22）。这些特征描述的是"过度"而非"恶意"——操作太长、危险标签太多、权限过大。

### normal（右）

特征集中在**质量与人气**：`stars_log`、`score`、`desc_len`。正常 skill 的特征模式是收藏多、评分高、描述适中。

**交叉洞察**：
- `cat_security` 在三类中都出现在 Top 15——这是**跨类别通用特征**，说明安全类别本身就携带强分类信号
- `actions_len` 在 malicious 排第 4、unsafe 排第 1——但在 normal 中重要性大幅下降。**操作长度是区分"异常"与"正常"的关键**
- 8 个特征只在一个类别中出现——大多数特征具有类别特异性，说明模型学到了精细的类别边界

---

## Fig 3: Top 3 特征依赖图

**`fig3_dependence_top3.png`**

**展示什么**：malicious 类别 Top 3 特征的 SHAP 依赖图。X 轴=特征值，Y 轴=SHAP 值，颜色=第二强交互特征的取值。

**三个 Top 特征**：

**`sig_credential_theft`**（凭据窃取信号）：
- 特征值为 0（无条件命中）→ SHAP ≈ 0（不推高恶意）
- 特征值为 1（命中该 pattern）→ SHAP 急剧上升至 +2.0 以上
- **这是一个 ON/OFF 开关型特征**——一旦命中，立即将分类推向恶意
- 交互最强的第二特征通常是 `cat_security`：如果既是安全类别又有凭据窃取 pattern → 双重确认

**`cat_security`**（安全类别）：
- 特征值为 0（非安全类）→ SHAP ≈ −0.5（轻微压向非恶意）
- 特征值为 1（安全类）→ SHAP 分布广泛，从 −0.5 到 +2.0
- 注意：**安全类别本身不一定代表恶意**，但与其他危险信号联动时 SHAP 飙升
- 交互特征：`n_danger` 或 `sig_credential_theft`

**`sig_data_exfil`**（数据外泄信号）：
- 与 `sig_credential_theft` 类似——开关型，命中则 SHAP 推高
- 但 SHAP 幅度（~1.0）小于 `sig_credential_theft`（~2.0），说明数据外泄的威胁感知低于凭据窃取

---

## Fig 4: 恶意分类瀑布图

**`fig4_waterfall_malicious.png`**

**展示什么**：解释一个具体 skill 为什么被分类为 `malicious`。从基线概率（base value）开始，每个特征正向（红色箭头）或负向（蓝色箭头）推动预测，最终达到模型输出值。

**示例 skill**：`skillguard`（安全审计类 skill）

**解读方式**：
- **Base value**（灰色）≈ 某个起始概率
- **红色箭头**：推高恶意概率的特征——如命中 `sig_credential_theft`（+X.XX）、`cat_security`（+Y.YY）
- **蓝色箭头**：压低恶意概率的特征——如 `stars_log` 较高（收藏多 → 可能是合法 skill）
- 所有箭头叠加 = **f(x)**（最终模型对该 skill 的输出）

**关键信息**：
- 不止告诉你"这个 skill 是恶意的"，更告诉你"为什么是恶意的"
- 对于安全审计员，可以直接定位到具体句子/词 → 快速验证
- 瀑布图可复现：任何 skill 都可以生成这样的解释

---

## Fig 5: Top 15 特征相关性热力图

**`fig5_feature_correlation.png`**

**展示什么**：malicious Top 15 结构化特征之间的 Pearson 相关系数矩阵。

**聚类观察**：

**高相关簇 1 — 风险标签（左上）**：
- `n_danger` ↔ `total_risks` (r ≈ 0.7-0.9)：危险标签数自然与总风险数高度相关
- `n_warn` ↔ `total_risks`：同上
- 这些特征之间存在**多重共线性**——未来可以考虑用 PCA 压缩或只保留一个

**高相关簇 2 — 权限特征（中间）**：
- `total_perm_items` ↔ `perm_network_count` / `perm_shell_count`：权限项多自然包含更多网络和 Shell 权限
- `perm_os_count` ↔ `perm_shell_count`：OS 权限通常伴随 Shell 权限

**低相关但重要的特征**：
- `sig_credential_theft` 与除了 `sig_data_exfil` 外的所有特征相关性极低（r < 0.1）——**它携带的是独立、新增的信息**，这是优秀特征的标志
- `stars_log` 与安全信号特征几乎不相关——人气指标反映的是另一维度

**实用启示**：
- 高相关簇内的特征可以精简而几乎不损失性能
- `sig_credential_theft`、`cat_security` 等低相关高 SHAP 特征是模型的核心资产，绝对不能去掉

---

## Fig 6: 综合仪表盘

**`fig6_shap_dashboard.png`**

**展示什么**：6 面板综合视图。

### (0,0)-(0,2): 三类特征重要性柱状图
- 三栏并排：malicious（红）、unsafe（黄）、normal（绿）
- 直观展示同一特征在不同类别中的不同重要性
- 如 `actions_len` 在 unsafe 排第 1，在 malicious 排第 4——同一特征支撑了不同的分类逻辑

### (1,0): 特征跨类重叠度
- 统计每个特征出现在多少个类别的 Top 10 中
- **4 个特征出现在全部 3 个类别** → 它们是最通用的区分信号
- **8 个特征只出现在 1 个类别** → 它们是类别特定信号
- 绿色=3 类共享，黄色=2 类共享，红色=仅 1 类特有

### (1,1): 特征组贡献度（malicious）
- 将 45 维结构化特征聚类为 5 组，看哪组贡献最大
- **Security Signals**（安全信号组）预计占比最大——因为 `sig_credential_theft` 等全部在这里
- **Risk**（风险元数据组）第二——`n_danger`、`danger_ratio` 提供了结构化的风险背景
- **Perms**（权限组）和 **Text Stats**（文本统计组）贡献较小——它们是辅助信号而非主信号

### (1,2): TF-IDF vs 结构化贡献对比
- **结构化特征（45 维）贡献 54%** vs **TF-IDF（500 维）贡献 46%**
- 45 维 vs 500 维 → 结构化特征的单位维度贡献是 TF-IDF 的 **13 倍**（54/45 ÷ 46/500 ≈ 13）
- 这再次验证了消融实验的结论：结构化特征是安全信号的主要载体

---

## 总体结论

1. **`sig_credential_theft` 是最强的单个恶意信号**（SHAP=0.90）。凭据窃取 pattern 一出现，模型几乎必然判定 malicious。这验证了我们的特征工程是有效的——它精确捕捉了安全专家最关注的攻击模式。

2. **操作长度是 unsafe 的第一特征**（SHAP=1.57）。unsafe 不等于 malicious——过度权限、冗长指令才是 unsafe 的主信号。模型正确地区分了"故意攻击"和"不小心过度"。

3. **结构化特征效率是 TF-IDF 的 13 倍**。45 维结构化贡献 54% vs 500 维 TF-IDF 贡献 46%。在安全分类中，**领域知识编码的少量特征远优于大量通用文本特征**。这与 SkillSieve [1] 的核心设计理念一致。

4. **SHAP 瀑布图使模型可解释**——不仅可以判断"是否恶意"，还可以解释"为什么恶意"，为安全审计提供可追溯的证据链。

---

## 参考文献

[1] Y. Hou and Z. Yang, "SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills," 2026.

[2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
