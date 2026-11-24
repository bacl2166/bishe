# 2026-11-24 交付验证

- `python -B -m unittest discover -v` 通过40项测试。
- `python -B scripts/smoke_exp1.py` 遍历12编码、3评分器、3概率模式，生成25,704行，充分搜索恢复自身精确排名，原样续跑成功。
- 真实 Amazon Musical_Instruments 的128 MiB评论前缀和完整元数据用于小规模集成检查。训练目录150件商品，实际2位用户、44条含随机对照的请求，单种子、编码与SASRec均1轮，TF-IDF/SVD维度8，生成9,504行。
- 真实检查中，beam500的自身参考召回均为1或空集合null。preserve与feasible_mass的精确目标重合率均为1或空集合null。所有非空输出满足约束。
- GPU训练和正式规模实验由服务器运行配置执行。上述小样本用于接口与数值验证，其样本量不足以推断编码优劣。

真实验证使用同一CLI，主要覆盖配置为 max_items=150、min_history=2、max_users=3、max_requests=45、encoding_seeds=[11]、scorer_seeds=[11]、budgets=[{"beam":5},{"beam":500}]。请求上限保持结构请求与随机对照成对，实际返回44条。原始数据、用户记录、训练权重与运行目录均由 .gitignore 排除。
