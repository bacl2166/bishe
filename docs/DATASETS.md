# 三域 Amazon Reviews 2023 实验

数据来源为 McAuley Lab 的 [Amazon Reviews 2023 官方 Hugging Face 数据卡](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#grouped-by-category)，核对日期为 2026-11-25。下表沿用官方四舍五入统计，商品数按评论中出现的商品计算，部分商品没有元数据。

| suite 名称 | 官方分类 | 原始交互 | 原始用户 | 原始商品 |
| --- | --- | ---: | ---: | ---: |
| `musical` | `Musical_Instruments` | 3.0M | 1.8M | 213.6K |
| `fashion` | `Amazon_Fashion` | 2.5M | 2.0M | 825.9K |
| `appliances` | `Appliances` | 2.1M | 1.8M | 94.3K |

原始交互数接近，因此选这三个域进行相同预算的对照。用户活跃度、商品分布和属性完整率不同，相同原始规模不能保证相同的有效训练规模或测试覆盖率。正式报告必须使用各域输出中的实际审计数字。

## 获取完整文件

在仓库根目录运行以下命令。下载脚本保留原来的 `--category`、`--output`、`--endpoint`、`--revision` 接口，默认从官方 Hugging Face 仓库获取未压缩 JSONL。

```bash
python scripts/download_amazon.py --category Musical_Instruments --output data
python scripts/download_amazon.py --category Amazon_Fashion --output data
python scripts/download_amazon.py --category Appliances --output data
```

每域需要 `data/<Category>.jsonl` 和 `data/meta_<Category>.jsonl`。各域来源 URL 合并保存在 `data/download_sources.json`。下载先写 `.partial`，完整接收后才替换目标文件；重复调用会重新下载对应文件。需要固定上游版本时，为三个下载命令传入相同的 `--revision <commit>`。

提交的六份配置均指向完整文件，不设置 `review_prefix_bytes`。本地已有的 128MiB 乐器评论前缀仅供独立 smoke 测试；它与完整数据的正式实验不可混用。添加这些配置和运行入口不会启动下载或训练。

## 统一预算与数据边界

`configs/experiment1_{musical,fashion,appliances}.json` 采用完全相同的设置，仅数据路径与输出目录不同：

- 训练评论早于 `2021-01-01`；验证标签来自 `[2021-01-01, 2022-01-01)`；测试标签来自 `2022-01-01` 起，日期均为 UTC。
- 有效目录最多 3000 个商品，商品至少有 5 次训练交互，按训练热度排序；`evaluation_split=test` 仅从存在测试期事件、且至少有 5 条目录内训练历史的用户中按固定哈希抽取最多 200 人，最多保留最近 50 条历史。测试正例的商品身份、目录归属和是否可行不参与用户选择。旧配置未指定该键时保持 `either` 行为，允许只有验证事件的用户。
- 验证与测试均使用冻结的训练历史。最早的标签即使在目录外也保留，用审计记录覆盖率。
- 总请求预算最多 800，包含结构约束和随机对照。先给用户分配无筛选基线，再每轮每人分配一条结构约束，起始模板按用户次序轮转，避免总是截取前几种约束；随机对照与结构请求保持完整配对。当前预算可覆盖 200 位测试用户的基线及至少一组约束/随机对照；每个用户不再穷尽所有约束模板。各约束族和有效正例实际独立用户数仍须报告。
- 固定请求种子 2026，编码与 scorer 种子均为 `[11, 22, 33]`。三个 scorer 为 `most_popular`、`item_knn`、`sasrec`；神经 scorer 的训练样本上限为 50000。
- 12 个编码与原 `configs/experiment1.json` 一致。约束族保持 `no_filter`、`price`、`brand`、`category`、`rating`、`spec`、`price_brand`、`price_brand_spec`、`random_matched`。

配置不固定 `spec_field`；约束构造器从训练目录中选取覆盖商品最多的规范规格字段，实际字段写入审计。评分约束来自训练评论的评分均值与次数，最低次数为 5，忽略元数据中的全期评分统计。

缺失属性不会被填造。若某域（包括 Fashion）的品牌或价格全缺失，依赖该属性的约束族会跳过，并记录 `constraints.skipped_families`；只有部分商品缺失时，这些商品不能满足涉及该属性的规则。规格、类别和评分约束采用相同原则。空可行集保留。跨域比较须列明各域实际支持的约束族，并在同一约束族与预算下对照。

`max_items=3000` 和 `max_users=200` 是上限。执行准备阶段后，至少检查 `audit.json` 中的 `review_rows`、`train_review_rows`、`catalog_size`、`train_users`、`evaluation_users`、`test_labels`、`test_catalog_coverage`，以及 `constraints.unique_users`、`constraints.family_unique_users`、`constraints.family_counts`、`constraints.family_positive_feasible_requests`、`constraints.family_positive_feasible_users`、`constraints.joint_attribute_coverage` 与 `constraints.skipped_families`。目录、属性联合覆盖率、可行集大小和目录外标签率不同，均可能影响最终指标。元数据仍是 2023 年静态快照，不能视为交互当时的历史属性。

## 运行顺序

先仅准备三域数据，检查审计；这一步不训练编码或模型。

```bash
python scripts/run_suite.py --experiment 1 --datasets musical fashion appliances --prepare-only
```

确认数据后，续跑实验一，再运行实验二：

```bash
python scripts/run_suite.py --experiment 1 --datasets musical fashion appliances --resume
python scripts/run_suite.py --experiment 2 --datasets musical fashion appliances --resume
```

也可一次串行完成。每个域按实验一、实验二的顺序执行，再进入下个域；任一子进程失败即停止。

```bash
python scripts/run_suite.py --experiment both --datasets musical fashion appliances --resume
```

`--prepare-only` 仅允许配合 `--experiment 1`。实验二需要实验一生成的目录、请求及全部编码映射；仅准备数据后直接运行实验二不会具备这些产物。suite 在启动实验二前检查 `source_run` 目录存在，实验二 runner 继续验证具体产物。

从其他目录调用时，应使用脚本的绝对路径；suite 将每个子进程的工作目录固定为本仓库根目录，因此配置中的相对数据路径与输出路径不会随调用位置变化。`--config-dir` 可切换配置目录，相对路径同样以仓库根目录为基准。

## 实验二配置

`configs/experiment2_{musical,fashion,appliances}.json` 的 `source_run` 分别引用 `runs/<domain>-experiment1-v2`，输出至 `runs/<domain>-experiment2-v2`。实验二使用相同的全部 12 个编码，编码与生成模型种子均为 `[11, 22, 33]`。版本 v2 修正用户选择与请求分配，旧版输出保留；应重新准备数据，不能复用旧 `prepared.json` 或 `requests.json`，原始下载文件可以复用。

生成模型配置为 CUDA、`d_model=64`、2 个 attention heads、1 层 encoder、2 层 decoder、batch size 128、学习率 0.001、最多 20 个 epoch、patience 3、训练样本上限 50000、历史上限 50。三个域使用相同配置；该配置不承诺某一显卡上的运行时长。

诊断请求预算为 80，抽样种子为 2026，`top_k=10`，模式为 `preserve` 与 `local`，预算为 beam 10、30、100，以及 beam 100 配合 2000 节点上限。Bootstrap 次数为 1000。诊断请求量也是上限，实际使用规模由 runner 产物报告。
