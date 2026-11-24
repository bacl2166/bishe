# 实验一服务器运行说明

本实验冻结商品评分，改变商品编码和购物条件，测量分布偏移、搜索损失、商品级推荐指标和计算开销。初始推荐模型为 MostPopular、ItemKNN 和因果 SASRec。实验二的生成模型训练属于后续工作。

## 快速验证

```bash
git clone git@github.com:bacl2166/bishe.git
cd bishe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest discover -v
python -B scripts/smoke_exp1.py
```

离线联调用模拟的 Amazon 格式记录，CPU、TF-IDF+SVD 语义向量、一个训练周期，遍历十二种编码、三个评分器、三种解码目标。充分搜索要恢复各方法自身精确排序，并验证续跑。此结果用于验证接口和数值。

## 下载与数据审计

```bash
python scripts/download_amazon.py --category Musical_Instruments --output data
# 主站不通时可指定可访问的镜像
python scripts/download_amazon.py --category Musical_Instruments --output data --endpoint https://hf-mirror.com
python -m exp1 --config configs/experiment1.json --prepare-only
```

下载器先写 `.partial`，成功并核对响应长度后原子重命名。重复下载会重新获取文件。已有 JSONL/gz 可直接在配置里填写路径。固定数据版本可使用 `--revision` 指定发布方提交号。数据指纹会写入运行 manifest。

准备阶段输出 `audit.json` 和 `requests.json`。检查品牌、价格、品类及规格的单项和联合覆盖率、实际测试用户数量、有效正例数、各约束族人数。若某类属性完全缺失，audit 记录跳过原因。商品资料是抓取时快照；评论时间用于交互时间代理。

完整文件默认严格验证每行 JSON，截断文件会报错。之前的 128 MiB 试跑样本需在配置增加 `"review_prefix_bytes": 134217728`，以明确只读取该前缀内完整行。

训练截止默认为 2021-01-01 UTC，验证期截至 2022-01-01，之后为测试期。训练目录按训练交互数量选择。所有推荐历史冻结为训练历史，模型验证仅用于 checkpoint 选择。训练期评分、热度与协同向量由训练交互生成。文本特征只使用训练目录商品的静态元数据。

## 正式运行

```bash
python -u -m exp1 --config configs/experiment1.json --resume 2>&1 | tee experiment1.log
```

如果跳过 prepare-only，首次运行去掉 `--resume` 即可。配置默认使用 CUDA 训练 SASRec 和神经量化器，语义模型使用 CPU。CPU 服务器将 `scorer.device`、`tokenizer.device` 设为 `cpu`，可将 `embeddings.backend` 设为 `tfidf_svd` 并记录该变更。SentenceTransformer 首次运行需要下载模型，也可将 embeddings.model 指向服务器已有模型目录。

运行目录按配置、输入文件字节和实验源码生成指纹。同指纹使用 `--resume` 可重用数据、语义向量、编码映射和已完成的整组评测结果。中断发生在某组结果写完前，会重新计算该组。原子写入的完整 JSON 是恢复标志。改变配置、输入或源码时选择新 output 目录。并发任务必须用不同目录。

商品概率矩阵会在每个评分器训练后原子保存。恢复时读取同一份概率，保证跨中断的编码比较使用相同商品分数。原生映射文件内容也计入输入指纹。

默认配置对三个评分种子和三个编码种子做交叉比较。规则和用户抽样使用独立固定 request_seed。先用较小 max_items/max_users/max_requests 测量运行规模，再冻结正式配置。max_requests 包含结构请求及其随机对照，可能使实际用户数小于 max_users，应检查 audit.constraints.unique_users。max_requests 足够大时可覆盖更多用户。

1200条请求的完整矩阵最多约700万结果行。逐任务落盘和汇总，配对分析一次载入一个基线与一个比较编码的结果。服务器需为行级JSON和模型文件预留磁盘空间。配置中的100轮编码训练、30轮SASRec训练是起始预算，收敛和超参数选择应在验证集上完成。

## 已选约束

- no_filter 作为完整目录参考。
- price 包含价格上限和中心价格区间，各有三种强度，阈值来自训练目录分位数。
- brand、category 是单值或多值允许集合。同维度取并集，维度间取交集。
- rating 使用训练期平均评分和评价数。
- spec 使用 constraints.spec_field，默认 string_gauge，可换 material/size/color/compatible_devices/instrument。
- price_brand、price_brand_spec 组合筛选。
- random_matched 为每条结构规则生成同数量随机候选，对照引用 matched_request。

未知的受约束属性视为不可行。所有模型与编码使用完全相同的可行集合。规则生成不读取测试正例，正例被排除时保留覆盖率统计。目录外正例同样保留。

## 三种评分与精确参照

固定评分器输出 logits 后，经统一 temperature 的稳定 softmax 得到每件商品正概率 q。MostPopular 和 ItemKNN 输出对数权重，温度1恢复对应权重的比例。每条历史的概率在所有编码间缓存并共享。

每个编码唯一对应商品，搜索统一追加 EOS。原前缀质量为全部后代商品概率之和。

- preserve 屏蔽无可行后代路径，保留原条件概率，完整路径恢复原商品概率。
- local 每个前缀只在当前合法子节点间重新归一化，路径评分可能改变。
- feasible_mass 汇总完整可行后代质量，恢复 q 在可行目录上的全局条件分布。

每种模式先枚举所有可行商品的完整路径分数，得到自身精确 Top-K。另保留所有编码共享的原 q 精确 Top-K。统一计入 EOS，无长度平均。

Beam 限制每层保留的未完成前缀，完成商品进入独立候选池，最终按完整路径分数取 Top-K。每层评分是原子操作，若该层超出剩余 node_budget，停止该层前的扩展，因此预算是严格上限，可能出现返回不足。充分 beam 必须与自身枚举一致。

node_budget 统计有效子节点的评分次数。时间分别记录完整分布与 mask 准备、搜索解码。由于本实验预先枚举商品概率并构造前缀，CPU 搜索耗时应作为机制测量；上线生成模型的真实服务延迟需要实验二单独测量。

## 输出与分析

`summary.json/csv` 为每组配置的均值、样本数、可行正例数和命中丢失率。`paired_comparisons.json/csv` 在同请求、同用户、同种子和预算下，相对 random_balanced 做用户级配对 bootstrap，报告95%区间。三个种子的结果保留独立行，避免把所有种子当作独立用户。

`rows/*.json` 含逐请求的实际/原目标/自身目标商品列表、正例原排名、首次剪枝深度、参考前缀数量、节点数和耗时。用户在结果行中哈希，prepared/requests 仍含数据集用户标识，应保留在本地运行目录。

主要指标定义如下。

| 指标 | 定义 |
|---|---|
| hr/ndcg | 单历史正例仍可行时计算；被排除为null |
| positive_feasible | 正例可行率的逐请求0/1 |
| oracle_recall | 实际列表与原q精确列表重合比例 |
| own_oracle_recall | 实际列表与自身模式精确列表重合比例 |
| objective_topk_overlap | 两份精确列表重合比例 |
| distribution_kl | 可行条件分布q_A相对自身完整路径分布的KL |
| root_kl | 根层理想可行质量分布相对局部mask分布的KL |
| root_invalid_mass | 根层仍合法分支内不可行商品的原概率总量 |
| lost_hit_rate | 原精确命中但实际漏掉的次数/原精确命中数 |
| own_lost_hit_rate | 自身目标下对应的命中丢失率 |
| shortfall | 返回数小于 min(K,可行目录数) |

空目录的参考召回为null。分母零的命中丢失率为null。两份精确排序已经变化时，先分析目标偏移，再用自身参考解释搜索遗漏。HR提升但参考召回降低时，检查新增命中与丢失命中的逐请求变化。相同beam的节点数可能不同，利用预算曲线比较成本。

所有编码的受控适配与原生映射说明见 [TOKENIZERS.md](TOKENIZERS.md)，评分器说明见 [SCORERS.md](SCORERS.md)。原生完整论文结果需要相应作者映射和协议。
