# bishe

当前价格专项采用[实验一分阶段方案](docs/PRICE_EXP1.md)，配置 `configs/price-exp1/experiment1_musical.json` 覆盖全部 12 种编码。使用 `python -m exp1 --config <配置> --stage prepare|encode|score|search --resume` 分步推进，首轮单种子核验流程与成本，价格上限/区间各三档并保留无筛选和配对随机对照。

新增可选的[扩展预处理配置](docs/EXPANDED_DATA.md)：最多 10000 件商品、2000 位测试用户、20000 条请求，使用 `--config-dir configs/expanded`。先运行 `--prepare-only`，再用 `python -m exp1.audit --run <输出目录> --split validation` 检查各约束与强度的独立用户和有效正例数量。规模扩展不保证每种筛选都具有足够标签；审计后再固定正式模型评测预算。

## 三个数据集与实验二

新增 Musical Instruments、Amazon Fashion、Appliances 三个域，原始评论规模约为 300/250/210 万。三个域统一最多 3000 件商品、200 个有测试事件的评估用户、800 条请求和 50000 个模型训练样本，清洗后的实际规模与属性覆盖率分别审计。请求先保留无筛选基线，再轮转分配约束及成对随机对照；当前配置输出到 `runs/<domain>-experiment{1,2}-v2`，保留旧版审计结果。数据获取与完整配置见 [三域说明](docs/DATASETS.md)，修复背景见 [用户覆盖说明](docs/EVALUATION_COHORT.md)。

实验二训练真正的 Transformer 生成推荐模型，支持全部十二种编码、原始概率与局部归一化两种解码、固定诊断请求的逐商品精确路径评分、HR/NDCG、搜索损失和断点续跑。方法和指标见 [实验二说明](docs/EXPERIMENT2.md)。论文方法的受控适配范围沿用 [编码器说明](docs/TOKENIZERS.md)。

```bash
python -m pip install -r requirements.txt
python scripts/smoke_exp2.py
python scripts/download_amazon.py --category Musical_Instruments --output data
python scripts/download_amazon.py --category Amazon_Fashion --output data
python scripts/download_amazon.py --category Appliances --output data
python scripts/run_suite.py --experiment 1 --prepare-only
python scripts/run_suite.py --experiment 1 --resume
python scripts/run_suite.py --experiment 2 --resume
```

正式配置使用 CUDA，三域共训练 324 个生成模型。先检查数据审计；需缩小试跑时另存配置并减少编码、种子及请求预算。新增两个域的完整数据与正式 GPU 实验由服务器执行，本地流程检查不代表论文效果结论。

## 完整实验一

已新增 `exp1` 服务器入口，包含十二种编码入口、九类约束/对照、MostPopular/ItemKNN/SASRec、三种概率处理、各自的精确排序、逐请求诊断和用户级配对区间。运行步骤见 [实验一服务器说明](docs/EXPERIMENT1.md)，方法适配范围见 [编码器说明](docs/TOKENIZERS.md)。

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -v
python -B scripts/smoke_exp1.py
python scripts/download_amazon.py --output data
python -u -m exp1 --config configs/experiment1.json
```

默认配置需要CUDA，CPU运行修改 `scorer.device` 和 `tokenizer.device`。论文编码方法使用明确标注的受控适配，并支持导入原作者的商品映射。正式训练与论文指标应依据相应协议核验。

## Amazon Reviews 2023 乐器数据初始实验

使用 [Amazon Reviews 2023 的 Musical_Instruments 子集](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#grouped-by-category)。本次运行读取[原始评论文件](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw/review_categories/Musical_Instruments.jsonl)前 128 MiB 中的完整记录，并使用[完整商品元数据](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw/meta_categories/meta_Musical_Instruments.jsonl)。评论文件按原始顺序截取，随后对符合条件的用户按种子抽样。需要 Python 3.10 或以上版本，无第三方依赖。

在 PowerShell 中运行下面的命令，从镜像下载两个输入文件并执行实验。

```powershell
$dataDir = '../amazon-2023-musical-instruments'
New-Item -ItemType Directory -Force $dataDir | Out-Null
$source = 'https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw'
curl.exe -fLsS --range 0-134217727 -o "$dataDir/reviews-128m.raw" "$source/review_categories/Musical_Instruments.jsonl"
curl.exe -fLsS -o "$dataDir/meta_Musical_Instruments.jsonl" "$source/meta_categories/meta_Musical_Instruments.jsonl?download=true"
python -B amazon_experiment.py --reviews "$dataDir/reviews-128m.raw" --metadata "$dataDir/meta_Musical_Instruments.jsonl" --review-prefix-bytes 134217728 --output results/amazon_musical_instruments.json
python -B -m unittest -v
```

程序也支持完整 `.jsonl` 或 `.jsonl.gz` 输入。移除 `--review-prefix-bytes` 可处理完整评论文件。复现时可用以下 SHA-256 校验输入文件。

- `reviews-128m.raw` `a8111672f00e00b71c631dd8232f6fff4cc9e07a421e2e0b91c7f895edab2c31`
- `meta_Musical_Instruments.jsonl` `96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c`

### 评测协议

- **目录与请求**。以 `parent_asin` 匹配评论与商品。默认以 2022-01-01 UTC 为训练截止，选出截止前评论最多且有有效价格的 2,000 件商品。对训练期至少交互过 3 件目录商品、且截止后有评论的用户，以固定种子抽取最多 400 人。用户截止后的第一条评论是测试正例。正例落在目录外时仍计入覆盖率。
- **约束**。唯一的购物约束是价格不超过训练目录的价格中位数。价格使用抓取时的商品快照。目录和阈值在读取测试结果前确定。
- **两种推荐模型**。MostPopular 按训练评论数评分。ItemKNN 用训练用户的商品共现计算余弦相似度，每件商品保留最相似的 100 个邻居。推荐时对用户最近 50 次训练交互中的不同商品累加近邻相似度，并用微量热度平滑。两种模型使用同一目录、请求和可行集合。
- **两种商品编码**。商品按价格从低到高分组，另以固定种子随机分组。默认 32 个组，两种方法的对应组容量相同。分别运行 beam 宽度 8、32，并以每个模型在可行集合上的精确 Top-10 为参考。
- **指标**。在正例可行的请求上计算 HR@10、NDCG@10，同时报告正例可行率。OracleRecall@10 和节点数用于分析搜索。两种模型使用全可行集合评估，交互历史中已出现过的商品也可作为后续交互目标。

### 试跑结果

共读取 245,281 条评论。目录包含 2,000 件有价格的商品，价格上限为 25.90 美元，筛选后剩余 1,000 件。400 条测试请求中有 94 条的正例在目录内，53 条的正例仍可行。HR 和 NDCG 的分母为这 53 条请求。

| 模型 | 搜索方式 | HR@10 | NDCG@10 | OracleRecall@10 | 平均评分节点数 |
|---|---|---:|---:|---:|---:|
| MostPopular | 精确排序 | 0.0566 | 0.0257 | 1.0000 | — |
| MostPopular | 价格有序编码，beam 8 | 0.0755 | 0.0314 | 0.9000 | 520.0 |
| MostPopular | 随机平衡编码，beam 8 | 0.0755 | 0.0300 | 0.6000 | 265.0 |
| ItemKNN | 精确排序 | 0.0000 | 0.0000 | 1.0000 | — |
| ItemKNN | 价格有序编码，beam 8 | 0.0000 | 0.0000 | 0.7985 | 517.6 |
| ItemKNN | 随机平衡编码，beam 8 | 0.0189 | 0.0055 | 0.5208 | 286.7 |

Beam 扩大到 32 时，两套编码都复现了各自模型的精确 Top-10。Beam 为 8 时，价格有序编码找回了更多模型高分商品，同时评分的节点也更多。ItemKNN 的精确排序在这 53 条请求中未命中正例。结果来自单个样本和种子，适合作为流程验证。完整输出见 [结果 JSON](results/amazon_musical_instruments.json)。

## 价格约束下的合成数据演示

本实验用五件模拟商品、一条购物请求和一套固定商品分数，比较不同商品编码在有限搜索预算下的表现。唯一的筛选条件是商品价格不超过预算上限。默认上限为 20 美元，历史交互正例设为 P3。

### 运行

需要 Python 3.10 或以上版本，代码只使用标准库。在仓库根目录运行：

```bash
python experiment.py
python -B -m unittest -v
```

可以修改预算上限（单位为美分）、beam 宽度、返回数量和随机种子：

```bash
python experiment.py --max-price-cents 1800 --beam-widths 1 2 --top-k 2 --seed 42
```

### 实验步骤

1. 按价格上限筛选可行商品，再按固定商品分数精确排序，得到参考 Top-K。
2. 给同一目录生成三套两段式编码。第一段是组号，第二段用于区分组内商品。
   - `product_type` 根据商品类型分组。
   - `price_band` 按固定的 10 美元价格带分组，分组边界独立于命令行指定的预算上限。
   - `random_balanced` 用固定种子打乱商品，再尽量均匀地分组。
3. 从编码构建前缀树。前缀分数等于它下方全部商品的固定分数之和。搜索屏蔽没有可行后代的路径，保留其余前缀的原分数。beam 宽度限制每层继续搜索的未完成前缀数，已到达的商品按原分数返回 Top-K。
4. 报告 HR@K、NDCG@K、OracleRecall@K 和搜索阶段评分的节点数。OracleRecall 的分母为参考列表实际长度。正例不符合预算时，HR 和 NDCG 显示 `NA`。可行集合为空时，OracleRecall 也显示 `NA`。

### 默认结果

预算为 20 美元时，可行商品为 P2 和 P3，精确排序为 `[P3, P2]`。

| 编码 | Beam | 搜索结果 | HR@2 | NDCG@2 | OracleRecall@2 | 节点数 |
|---|---:|---|---:|---:|---:|---:|
| 商品类型前缀 | 1 | P2 | 0.000 | 0.000 | 0.500 | 3 |
| 价格带前缀 | 1 | P3、P2 | 1.000 | 1.000 | 1.000 | 3 |
| 随机平衡编码 | 1 | P3 | 1.000 | 1.000 | 0.500 | 3 |
| 商品类型前缀 | 2 | P3、P2 | 1.000 | 1.000 | 1.000 | 4 |
| 价格带前缀 | 2 | P3、P2 | 1.000 | 1.000 | 1.000 | 3 |
| 随机平衡编码 | 2 | P3、P2 | 1.000 | 1.000 | 1.000 | 4 |

商品类型前缀把高分但超预算的 P1 和可行的 P2 放在一组。该组仍能参与搜索，其前缀分数高于 P3 所在组。Beam 宽度为 1 时，P3 所在组被提前剪掉。默认预算下，价格带前缀将 P2 和 P3 放在同一条可继续搜索的分支。

这些结果用于验证搜索机制。商品分数和正例均为人工设定。节点数仅统计进入搜索评分的可行子节点，真实数据上的推荐效果和模型推理成本需要后续实验测量。
