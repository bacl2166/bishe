# 实验二：真正的编码器—解码器推荐模型

`exp2.model.fit_generator(data, codes, config, seed, output_dir)` 训练并返回 `Generator`。
用户历史先以商品 ID 嵌入和位置嵌入进入 Transformer encoder，随后 Transformer decoder
使用因果自注意力与对历史的 cross-attention，逐 token 生成商品编码。
训练目标为 teacher forcing 全词表交叉熵，包含每个商品编码末尾的 EOS，忽略 PAD。
没有将实验一商品分数转换为 token 分数，也不在模型内进行树约束或分数重归一化。

## 接口与 token 约定

* `item_ids`：与 `data['items']` 顺序完全一致的字符串 ID。
* `token_codes`：商品 ID 到实际生成 token 元组，包含末尾 EOS。
* `encode_item(item_id)`：返回对应实际 token 元组，包含 EOS。
* `log_probs(history, prefixes)`：同一商品历史、多个已生成 token 前缀，返回
  `[len(prefixes), vocab_size]` 的完整词表 log-softmax。前缀不含 BOS/EOS/PAD，方法内部
  自动添加 BOS；每行概率和为 1，特殊 token 也保留有限概率。空前缀用于预测首 token。
  对商品路径计分时逐步累加每个实际 token（包括 EOS）的 log 概率，不做长度归一化。
* `model`：可访问的 PyTorch `nn.Module`；`metadata`：普通可 JSON 序列化字典。

保留 `PAD=0, BOS=1, EOS=2`，输入编码中所有原始整数 token 按全局升序重映射为
`3,4,...`，原 token 即使为 0/1/2 也不会与保留 token 冲突。
映射在 `raw_token_mapping` 属性和 checkpoint manifest 中公开。
输入编码须覆盖整个目录且彼此唯一，允许变长编码、互为前缀的编码和已添加碰撞后缀的编码。
每个输入编码必须非空，token 必须为非负整数（支持 NumPy 整数，拒绝 bool、浮点数与字符串）；
前缀也须由实际整数 token 构成，不会静默截断浮点值。
EOS 区分完整短编码与具有相同前缀的长编码。

## 数据与训练

只从 `train_sequences` 构造严格前缀到下一商品的样本；缺少该字段时才使用
`histories.values()`。目录外 ID 忽略，历史保留最后 `max_history` 个目录商品。
每个非首位置的目标只看到它之前的商品，不读取任何验证或测试事件来补训练历史。
`build_training_samples(data, max_history)` 可直接审计这些样本。
空历史在推理/验证时通过一个学习到的 encoder memory token 表示，避免全 padding 注意力。

验证数据仅使用 `validation` 提供的历史与目录内 positive，计算包括 EOS 的逐 token 平均
CE，选择最佳轮次并早停；不参与梯度更新。不读取 `test`。没有目录内验证正例时记录
`selection=train_ce_fallback` 并使用训练 CE 选择模型；没有训练前缀则明确报错。
训练损失按有效 token 求平均仅用于优化与模型选择；推理始终返回未经长度处理的逐 token
log 概率。所有目录商品均有编码目标，无候选集/可行性过滤。

默认参数：`d_model=32, heads=2, encoder_layers=1, decoder_layers=2, max_history=50,
dropout=0.1, epochs=10, batch_size=128, lr=0.001, weight_decay=0, patience=3,
max_train_samples=50000, torch_threads=2, device=auto`。`max_code_len` 自动取最长编码加 EOS
的实际长度。`max_train_samples=0`、负值或 null 表示不限制；超限时按 seed 均匀无放回
抽取前缀样本。设备支持 `cpu/auto/cuda/cuda:0`，不可用的显式 GPU 请求由 PyTorch 报错。
随机种子控制初始化、dropout、样本抽取与洗牌；跨设备/版本不承诺逐位相同。

## 保存与恢复

训练恢复最佳权重后写入 `generator_best.pt`，其中包含 `state_dict` 和完整 manifest；
同时写 `generator_manifest.json` 供人工检查。manifest 记录目录顺序、实际 token 编码、
原始 token 映射、架构配置、seed、逐轮损失和最佳轮次。
两份文件分别在目标同目录写入唯一 `.tmp` 文件后使用 `os.replace` 原子替换；写入失败
保留上一份完整目标并清理临时文件。两份文件不构成跨文件事务，恢复以 checkpoint 内嵌
manifest 为准。metadata 另记录 `train_seconds`（含模型初始化和样本构造，不含保存时间）、
`parameter_count` 及 `peak_cuda_memory_bytes`（CPU 为 null；GPU 为训练期间峰值已分配字节数）。
每轮输出 seed、轮次、训练/验证 CE 和经过秒数，并立即刷新以便远程查看训练日志。

`load_generator(data, codes, config, seed, output_dir)` 直接读取该 checkpoint，不构造训练
样本也不训练。传入 seed 必须等于保存值；仅允许 `device`/`torch_threads` 改变执行环境，
其他显式传入的配置键必须在保存配置中存在且相等，不传入的键使用保存值。
严格检查目录顺序、实际编码和原始 token 映射是否一致。metadata 标记
`loaded_from_checkpoint=True`。迁移 checkpoint 目录后可正常恢复。

运行 `python -m unittest discover -s tests -p test_exp2_model.py -v` 验证因果性、历史
cross-attention、完整词表归一化、EOS 与 padding、批量/单条 prefix 分数一致、训练样本
严格过去、最佳 checkpoint 恢复、同 seed 复现、实际权重更新和测试集不干扰训练。
