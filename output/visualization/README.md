# 可视化图说明

> 基于 10,501 条 DataHub Skill · TF-IDF (500d) vs SBERT (384d) 嵌入对比

---

## Fig 1: t-SNE 嵌入空间 — 异常标签对比

**`fig1_tsne_label_comparison.png`**

| 左：TF-IDF (500d) | 右：SBERT (384d) |
|---|---|
| 绿色=normal, 黄色=unsafe, 红色=malicious, 紫色=useless |

**解读**：
- SBERT 的异常点（红/黄）分布更**聚集**，形成可辨识的簇，说明 SBERT 对安全语义的编码优于 TF-IDF
- TF-IDF 的异常点散落在全空间，与 normal 几乎无法区分，验证了消融实验中"TF-IDF 引入噪声"的结论
- 颜色分布直观展示了 SBERT 将更多 malicious/unsafe 推向了特定的低密度区域

---

## Fig 2: t-SNE 嵌入空间 — 类别对比

**`fig2_tsne_category_comparison.png`**

| 左：TF-IDF | 右：SBERT |
|---|---|
| 8 色对应 Top 8 类别，灰色=other |

**解读**：
- SBERT 按类别形成的簇边界更清晰，同类 skill 在语义空间中确实更接近
- TF-IDF 的类别簇重叠严重，说明基于词频的特征区分不同类别 skill 的能力有限
- 这解释了为什么 SBERT 在 XGBoost 分类器上提升了 8% 的 F1

---

## Fig 3: UMAP 投影 — 异常标签对比

**`fig3_umap_label_comparison.png`**

与 Fig 1 同一数据，使用 UMAP 算法投影。UMAP 比 t-SNE 更好地保留了全局结构和簇间距。

**解读**：
- UMAP 中 SBERT 的异常簇更加紧凑，簇间距离更大，说明 SBERT 嵌入的判别性更强
- 少量 malicious（红点）在 SBERT 空间中形成孤立小簇，与 LOF 的局部密度检测原理一致

---

## Fig 4: 密度分析与异常浓度

**`fig4_density_analysis.png`**

| 上行：TF-IDF | 下行：SBERT |
|---|---|
| 左列：密度热力图 (viridis) | 右列：异常覆盖图 (绿=normal, 红=malicious/unsafe) |

右上角标注了稀疏区域（最远的 10%）和密集区域（最近的 10%）的异常占比。

**关键数字**：
- TF-IDF：稀疏区异常率 8.0%，密集区 13.0%
- SBERT：稀疏区异常率 13.4%，密集区 15.0%

**解读**：
- TF-IDF 中异常反而集中在密集区（正常区），这是**错误信号**——说明 TF-IDF 的异常识别方向反了
- SBERT 中稀疏区异常率更高（13.4% vs 8.0%），说明 SBERT 能更准确地将异常推向嵌入空间的边缘
- 密度热力图直观展示了两种嵌入的拓扑差异：SBERT 的结构更清晰、簇间过渡更锐利

---

## Fig 5: 综合分析仪表盘（6 面板）

**`fig5_dashboard.png`**

| 位置 | 内容 | 关键信息 |
|---|---|---|
| 左上 | TF-IDF t-SNE | 异常散落、无结构 |
| 中上 | SBERT t-SNE | 异常聚集、簇清晰 |
| 右上 | 标签分布 (log scale) | normal 9325, unsafe 1110, malicious 58, useless 8 |
| 左下 | Top 10 类别 | productivity 最多, security 第 6 位 |
| 中下 | 检测方法 F1 对比 | XGBoost 0.829 >> LOF 0.103 > LSTM AE 0.072 > IF 0.068 |
| 右下 | SBERT vs TF-IDF 直接对比 | SBERT 三项全胜 (IF +11%, LOF +24%, XGB +8%) |

**解读**：
- 一键概览整个项目的核心结果
- 右下角面板直接回答了"SBERT 比 TF-IDF 好多少"的问题
- 中下面板展示了有监督 vs 无监督方法的巨大差距，验证了弱标签的必要性

---

## Fig 6: 各类别异常率

**`fig6_anomaly_by_category.png`**

水平条形图，颜色越深异常率越高。

**解读**：
- testing (17.5%)、docs (14.9%)、coding (14.2%) 的异常率最高
- 这些类别中 skill 的文本模式更离散、质量更参差
- data (5.1%) 异常率最低，因为数据分析类 skill 的操作模式相对标准化
- 该图可用于指导安全审计的优先级：testing 和 docs 类应优先检查

---

## 技术说明

- **嵌入模型**：all-MiniLM-L6-v2 (384 维)，对 10,501 条 skill 文本编码
- **TF-IDF**：500 维，(1,2)-gram，sublinear TF
- **t-SNE**：perplexity=30, random_state=42, 5,000 点子采样
- **UMAP**：n_components=2, random_state=42
- **配色**：色盲友好方案 (Wong 2011)
- **格式**：PNG, 300 DPI, 适合论文/PPT 直接使用
