# 实验一评分器

`fit_scorer(name, data, config, seed, output_dir)` 支持 `most_popular`、`item_knn`、`sasrec`。
返回对象提供 `scores(history)`、`item_embeddings()` 与 `metadata`。分数长度等于
`len(data['items'])`，严格按目录顺序排列。SASRec 返回 logits，另两种返回对数偏好权重。
实验驱动对同一历史只计算一次分数并在所有编码间共享；需要概率时统一计算稳定的
`softmax(scores / temperature)`。评分器不读取任何编码树或约束可行集合。

## 数据边界

训练只使用 `train_sequences`；缺少该字段才回退到 `histories.values()`。
商品 ID 转为字符串，目录 ID 必须唯一。训练及历史中目录外 ID 忽略。
空历史或全部目录外历史返回训练流行度，明确记录在 metadata。
验证集只用于 SASRec 最优轮选择，不进行梯度更新；test 字段完全不读取。
所有目录商品（包括没有训练交互的商品）始终保留输出位置。

* MostPopular：训练事件出现次数取自然对数（计数下限 `1e-12`），不将验证或测试正例加入计数。
  因此统一 softmax 且温度为 1 时恢复训练频率分布，避免将大计数再指数化。
* ItemKNN：训练二元用户商品矩阵上的商品余弦相似度。删除自身边后，每个源商品保留
  最多 `knn_neighbors=100` 个邻居；相似度同分时按目录索引排序。历史商品邻居得分求和，
  加上最大值不超过 `1e-10` 的训练流行度以打破同分，再取自然对数（下限 `1e-12`）。
  因此温度为 1 时，统一 softmax 恢复邻居偏好权重比例。
* SASRec：PyTorch 多层 Transformer，商品与位置嵌入、因果注意力、padding mask、
  LayerNorm 和 dropout。每个训练序列的每个非首位置产生一个“严格前缀 → 下一商品”样本。
  全目录交叉熵训练，padding 索引 0 不参与输出或损失；使用全部目录负类，因此无需负采样。
  每轮以验证集逐样本平均 CE 选择最优模型，连续 `patience` 轮不改善早停。
  验证标签必须属于目录，且历史至少包含一个目录商品。没有有效验证样本时，以训练 CE
  选模型，并明确记录 `selection=train_ce_fallback`。最优权重写入 `sasrec_best.pt`，
  并恢复到返回模型；文件包括商品顺序、配置、seed 与最佳轮次。

## 配置与复现

默认配置：`epochs=10, batch_size=128, d_model=64, heads=2, layers=2,
max_history=50, lr=0.001, patience=3, dropout=0.1, max_train_samples=5000,
torch_threads=2`。历史保留最近 `max_history` 项。训练样本超过上限时，以指定 seed
从所有前缀中均匀无放回抽样；`max_train_samples=null` 或非正数表示不设上限。
`device` 支持 `cpu`、`cuda`、`cuda:0`、`auto`；默认自动选择 GPU（可用时）。
GPU 请求无法满足时 PyTorch 会明确报错，不静默更改设备。`d_model` 必须可被 `heads` 整除。
同一 seed 固定模型初始化、dropout 与样本顺序，并启用 cuDNN 确定性配置；跨硬件、
跨 PyTorch 版本不承诺逐位一致。metadata 包含实际设备、每轮损失、样本数和 checkpoint 路径。

`collaborative_embeddings(data, dim=64, seed=0)` 独立生成只依赖训练交互的协同向量：
二元用户商品矩阵做 TruncatedSVD，返回商品端 `V * singular_values`，维度不足补零。
单行或单列矩阵使用精确一维范数表示，空矩阵返回零向量。MostPopular/ItemKNN 的
`item_embeddings()` 使用这些向量；SASRec 返回训练后的商品嵌入。比较编码时应在同一实验
配置与 seed 下生成一次协同向量，随后复用。

轻量验证：`python -m unittest discover -s tests -p test_exp1_scorers.py -v`。
包含目录顺序、训练数据边界、KNN 余弦、因果不可见未来、padding 稳定性、SASRec
训练/最佳 checkpoint、测试标签不干扰模型和 CPU 复现检查。
