# 仅价格约束、全部编码的实验一

入口配置为 `configs/price-exp1/experiment1_musical.json`，输出到独立目录 `runs/musical-price-exp1`。复用完整原始文件；不直接复制旧的冻结请求，因为本轮的约束范围和预算已改变。使用最多 10000 件训练目录商品、2000 位有测试事件的用户，训练时间划分和用户资格与 expanded 配置一致，不根据正例商品或可行性选人。

## 请求与编码

每位用户都有 13 条请求：一条无筛选基线、六条价格请求，以及六条候选数量完全匹配的随机对照。达到 2000 位用户时共 26000 条。

| 价格变体 | 宽松 | 中等 | 严格 |
|---|---|---|---|
| 最高价格 | 训练目录价格 Q75 | Q50 | Q25 |
| 价格区间 | 最小值到最大值 | Q25 到 Q75 | Q25 到 Q50 |

分位数仅用训练目录的已知价格计算。未知价格不满足价格规则，无筛选基线仍保留这类商品。价格是静态元数据快照，规则是模拟购物筛选，不代表用户实际操作或历史价格。

全部 12 种编码参与：semantic_hierarchical、collaborative_hierarchical、rqvae、rqkmeans、opq、letter、cost、seater、topotok、bonsai、random_balanced、attribute_first。属性优先对照本轮只按价格构造前缀。论文方法的受控适配范围见 `TOKENIZERS.md`，不宣称完整复现各论文系统。

评分器为 MostPopular、ItemKNN、SASRec；同一评分器的商品概率只计算一次，所有编码共享。首轮编码与评分器均固定 seed=11，用于走通全方法流程和测量成本；不能据单种子结果作最终稳健性结论。后续三种子配置应在独立目录预先固定。

## 分阶段运行

`--stage` 是运行阶段选项，不改变配置指纹。前三个阶段分别保存 `prepared`、`encoded`、`scored`，搜索完成为 `complete`。`--prepare-only` 仍兼容旧入口，默认未指定 stage 时仍执行全部阶段。

| 阶段 | 输入 | 保存产物 | 使用资源 |
|---|---|---|---|
| prepare | 完整评论和元数据 | prepared.json、requests.json、audit.json | CPU与磁盘 |
| encode | 冻结训练目录和训练交互 | semantic.npy、collaborative.npy、12套映射和结构记录 | 文本嵌入CPU；神经编码器按配置使用CUDA |
| score | 冻结训练序列和评测历史 | 三评分器记录、固定商品概率NPZ | MostPopular/ItemKNN用CPU；SASRec用CUDA |
| search | 编码映射和固定商品概率 | 逐请求结果、汇总、配对比较 | CPU |

后续阶段必须带 `--resume`，缺少前置缓存时直接报错。搜索阶段只加载编码和商品概率，不隐式训练。语义/协同表示、编码映射与商品概率均可复用；续跑前不得修改源码、配置或输入文件。

### 第一步：准备与验证集审计

```bash
CUDA_VISIBLE_DEVICES="" python -u -m exp1 \
  --config configs/price-exp1/experiment1_musical.json \
  --stage prepare --resume

CUDA_VISIBLE_DEVICES="" python -m exp1.audit \
  --run runs/musical-price-exp1 --split validation
```

重点检查六种价格条件各自的独立用户数、有效正例用户数、空候选率；不要把六个变体汇总后的有效人数当作每个变体的人数。审计数量检查线不是统计功效保证，空集合和目录外正例继续保留。

### 第二步：表示与全部编码

根据当前节点实际分配选择 GPU；例如分配到物理 GPU 6 时执行 `export CUDA_VISIBLE_DEVICES=6`。该编号仅为示例，不能沿用历史占用截图。文本模型首次下载可通过公开镜像设置 `HF_ENDPOINT=https://hf-mirror.com`，必要时设置 `HF_HUB_DISABLE_XET=1`，也可在首次准备前另存配置指定本地模型目录。

```bash
python -u -m exp1 \
  --config configs/price-exp1/experiment1_musical.json \
  --stage encode --resume
```

检查 manifest 状态为 encoded，`encodings/` 下有 12 套 `mapping.json`、`tokenizer_manifest.json`、`structure.json`；检查碰撞消歧、码长、节点量和词表规模。

### 第三步：固定商品评分

```bash
python -u -m exp1 \
  --config configs/price-exp1/experiment1_musical.json \
  --stage score --resume
```

检查状态为 scored，`scorers/most_popular-11`、`item_knn-11`、`sasrec-11` 均有 `metadata.json` 和 `probabilities.npz`。概率列顺序与固定商品目录一致，同一用户历史不因筛选或编码而重新评分。SASRec仅用训练前缀训练，以验证集选择权重。

### 第四步：精确参照与搜索

确认已有全部缓存且计算/输出预算适合后运行：

```bash
CUDA_VISIBLE_DEVICES="" python -u -m exp1 \
  --config configs/price-exp1/experiment1_musical.json \
  --stage search --resume
```

三种模式为 preserve、local、feasible_mass，beam 为 10 和 100。每个请求先在同一可行集合中按原始商品概率精确排序，再与有限搜索及当前模式自己的精确排序对照。该阶段包含精确参照构造和诊断时间，不能将其整体耗时当作纯推荐延迟。

首轮最大输出为 `26000×12×3×3×2=5616000` 行，分 36 个评分器/编码作业保存。运行一个完整作业后可实测耗时和文件大小；中断后复用已完成作业，未完成作业重新计算。不要在运行中改配置以混接结果，也不自动删除旧数据。

## 结果解释

报告 HR/NDCG 及有效正例用户数、OracleRecall、自身目标 OracleRecall、目标 Top-K 重合率、KL、可行输出率、返回不足率、候选不足率、节点量和延迟。`search_score_loss` 是原始目标精确 Top-n 平均 log 概率减实际 n 项平均 log 概率；`own_search_score_loss` 对当前模式的精确目标进行相同计算，其中 n 为实际返回数与 K 的较小值。空输出记 null，返回不足另报。模式变化与有限搜索误差分别解释，不直接跨不同评分器比较绝对分数。

训练商品评分只需每个评分器一次，价格约束在请求评测阶段加入。当前不运行实验二。正式性能结论还需要后续多种子确认和其他数据域验证。

## 本地离线验证

```bash
python -B -m unittest -q
python -B scripts/smoke_exp1.py \
  --config configs/price-exp1/experiment1_musical.json --staged
```

集成检查使用合成 Amazon 格式小样本、CPU单轮训练，检验所有编码、三评分器、三概率模式及阶段缓存，不提供真实数据性能结论。
