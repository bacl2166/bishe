# 实验一编码器：实现范围与复现边界

入口 `exp1.tokenizers.build_codes(name, items, semantic, collaborative, config, seed, output_dir)`
返回 `dict[str, tuple[int, ...]]`。两矩阵行顺序必须与 items 一致。所有方法先按字符串商品 ID
排序并按行 L2 归一化（零向量保持零），避免输入排列影响随机训练和相同距离的处理。
空目录、重复 ID、非有限矩阵、维度不匹配会报错。无需下载模型或安装 Faiss。

## 范围声明

默认 `mode="controlled"` 是统一特征、计算预算和搜索接口下的**编码器受控适配**，
不能据此宣称完整复现原论文推荐系统或论文指标。特别是 SEATER 的联合生成训练、
LETTER 的下游 ranking-guided generation、BONSAI 的 SFT/GRPO 均不在本编码接口中。
这些边界写入运行目录的 `tokenizer_manifest.json`，论文结论必须区分 native 和 controlled。

所有方法均可指定 `mode="native", native_mapping="original_export.json"`，从原作者
程序导入 `{"item-id": [1,2,3]}`；ID 集合必须与当前目录完全一致，token 必须为非负整数。
没有提供映射时 native 模式直接报错，不会静默用适配算法替代。
映射导入证明使用了外部编码，但不自动证明其训练过程忠实；原始配置和数据拆分应另外保存。
该接口每商品仅允许一条路径；原生多路径需在外部先确定单路径，不能宣称保留了多路径训练。

## 十二种方法

| 名称 | 输入与核心算法/损失 | 实现边界 |
| --- | --- | --- |
| semantic_hierarchical | semantic；每节点递归 KMeans，分支 branching，固定 depth，单叶补零 | 普通层次语义基线；与 collaborative_hierarchical 共享分支和深度 |
| collaborative_hierarchical | collaborative；同一递归 KMeans | 协同矩阵由数据模块提供，编码器不使用验证/测试交互 |
| rqkmeans | semantic；每层拟合上一层残差，选择中心后逐层相减 | 真正残差量化，非重复对原矩阵聚类；保存每层 codebook |
| opq | semantic；分为 depth 个子空间，PQ 与正交 Procrustes 旋转交替优化平方重建误差 | NumPy/sklearn 实现 OPQ 核心优化；非 Faiss 二进制兼容实现；维度不足/不整除时补零 |
| rqvae | semantic；MLP encoder/decoder + 可学习残差码本；MSE reconstruction + codebook + commitment | 硬最近邻、straight-through 梯度、残差 KMeans 初始化；小预算 tokenizer 训练 |
| letter | semantic + collaborative；RQ-VAE + CF 对齐 + codebook diversity | 保留论文三项核心目标；CF 维度经可训练线性映射匹配 latent；码本使用容量约束分组 |
| cost | semantic；RQ + 输入与 decoder 重建向量间 batch InfoNCE | 采用论文 contrastive quantization 目标，替代 MSE 重建；不将 CF 近邻伪装成原文正例 |
| seater | semantic；容量约束平衡语义树 | **只实现树构造适配**；没有 SEATER 联合序列模型、alignment/ranking contrastive loss；严格原生比较需导入原作者映射 |
| topotok | semantic；RQ-VAE + 第一层组间、第二层组内、更深层商品间的拓扑蒸馏 | batch 内平方距离 softmax 教师/学生 KL；组来自对应层硬码，不是固定 KMeans 标签 |
| bonsai | 元数据文本 terms；递归 greedy minimum set cover 建 trie，特征耗尽停止 | 保留文本词与集合覆盖结构，不使用 KMeans；DF 词筛选代替原文 Qwen3-32B 抽词，仅取单路径，无 SFT/GRPO |
| random_balanced | seed 随机排列后递归等容量分组 | 与层次基线相同 branching/depth；容量不足的叶子最终消歧 |
| attribute_first | category/brand/price/specs；按配置字段优先生成离散前缀 | 规则基线；缺失值明确编码，价格单位继承输入，不推测币种 |

### 神经损失细节

RQ-VAE 每层计算 `MSE(code, stopgrad(residual)) + commitment * MSE(residual, stopgrad(code))`，
默认 commitment=0.25。decoder 重建时使用 straight-through。码本损失训练码向量，encoder
通过 straight-through 和 commitment 接收梯度。没有额外高斯潜变量 KL；这里遵循语义 ID
文献中 RQ-VAE 的确定性 residual-quantized autoencoder 用法。

LETTER 采用论文式 (3)/(4) 的负正例 softmax 概率（无额外 log），
CF 对齐比较量化累计向量与协同向量的线性投影；默认权重 0.02。
每个码本每轮重新做容量约束分组，同组随机选正例、排除当前码，diversity 权重默认 0.001。
较小目录或 k=1 无正例时 diversity 项为零。本文实现不包含原文推荐器的 ranking-guided loss。

CoST 使用 `contrastive_weight * CE(cos(input_i, decoded_j)/temperature, i) + RQ`，
没有额外 MSE 项。TopoTok 使用行 softmax 的平方距离分布 KL，第一层
教师为每组 semantic 均值、学生为码向量；第二层以累计码和为学生并只比较上一层同组商品；
之后比较全部商品；组内项排除对角，组间和商品间按原式包含自距离。
`topology_temperature=1.0` 对应原式的负距离 softmax。
深度 2 采用组间+商品间，深度 1 退化为组间+商品间。
单商品或没有有效比较对时拓扑项为零。该实现保留原文三类蒸馏目标，
未复现原文大规模训练协议；不可将其指标直接当作原文 TopoTok。

SEATER 小节点使用 Hungarian 槽位分配精确满足每组容量差 <=1，超过
`balance_exact_limit=1500` 时使用按距离 regret 排序的确定性容量贪心分配。
这保证容量约束，但大型节点的贪心并非原作者完整优化算法。

BONSAI 的 `item_terms` 可提供 `{item_id: [term, ...]}`，使用原作者抽词替换本地词筛选。
默认最多 16 个词，以较低目录 DF 优先保留区分性词；每节点贪心选择覆盖最多尚未分配商品
的词（稳定词序打破平局），选中词在后续路径移除。同一商品只分配到第一个覆盖它的词。
无可分特征时停止并用统一消歧逻辑保证唯一，不强制固定深度。

## 复现配置和产物

默认：`branching=8, depth=3, codebook_size=16, epochs=3, device="cpu"`。
神经训练默认 `latent_dim=min(32, semantic_dim), hidden_dim=max(32, 2*latent_dim),
batch_size=128, learning_rate=0.001, temperature=0.2`；`threads=1`。
OPQ `opq_iterations=5`。这些默认值是 smoke 预算，不代表论文收敛预算。
设置 epochs=0 可检查模型和编码管线，但输出是初始化状态，不能报告为训练结果。

`attribute_first` 可用：

```python
config = {"attributes": ["category", "brand", "price", "specs.color"],
          "price_bins": [1000, 5000, 10000, 50000]}
```

`category_level` 可选择某一级类目；否则编码完整类目路径。price=None 独立编码，
price_bins 必须严格递增，边界值进入上一个边界之后的箱。`specs` 编码完整规范 JSON，
`specs.KEY` 或 `spec:KEY` 编码一个规格字段。

每方法目录输出 `mapping.json`、`tokenizer_manifest.json`；RQ-KMeans/OPQ 保存 npz
模型，神经方法保存含模型 state_dict/维度/config 的 pt checkpoint。
manifest 记录实际长度、碰撞商品数、种子、模式及神经每轮损失。
所有碰撞组按商品 ID 排序，对**组内所有商品**追加一个确定性 token。
消歧 token 大于原始映射全部 token 的最大值，避免与已有变长路径混淆。
因此最终长度可以超过 depth；depth 表示基础编码深度，不应误报为最终固定长度。
EOS 由搜索模块统一追加，这里不保留/生成特殊 EOS token。

随机性由 NumPy 独立 RNG、固定 sklearn random_state 与 torch fork_rng 控制，CPU
结果可复现且不污染调用者的 torch RNG 状态。GPU 跨设备/版本逐位一致性不作保证。
CPU 单测覆盖十二方法完整映射、排列不变、全相同/零向量、单商品、碰撞、native 模式、
属性前缀、容量平衡和残差量化误差下降。

```powershell
python -m unittest discover -s tests -p test_exp1_tokenizers.py -v
```

## 核查的一手来源

- [LETTER 原论文 §3.1, 式 (2)–(5)](https://arxiv.org/html/2405.07314v2)
- [CoST 原论文 §3.1, 式 (1)–(5)](https://arxiv.org/html/2404.14774v2)
- [TopoTok 原论文 §3.2](https://arxiv.org/html/2607.18600v1)
- [BONSAI 原论文 §5](https://arxiv.org/html/2607.16633v1) 与 [官方代码](https://github.com/Liu-Jingzhe/BONSAI)
- [SEATER 官方实现](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/blob/main/model/SEATER.py)
- [Optimized Product Quantization 原论文](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html)

统一数据、特征和小训练预算有助于分析编码结构，但会改变这些方法原本的假设。
实验报告应优先解释结构与约束搜索行为，并显式披露受控适配，避免推导论文间完整性能排名。
