# STRING External Graph Residual GNN Pipeline

## 目标

这条路线把 PoPS 作为统计前端保留，把 STRING 外部图建模限定为残差修正模块：

1. 先用 PoPS 风格流程得到 trait-specific feature space
2. 再用独立的 STRING gene-gene 外部图做消息传递
3. 最终输出 `base linear score + graph residual score`

## 输入

1. `MAGMA` 输出
   - `.genes.out`
   - `.genes.raw`
2. `PoPS` 特征矩阵
   - `.mat.*.npy`
   - `.cols.*.txt`
   - `.rows.txt`
3. STRING 外部图边文件
   - 前两列为 gene id
   - 第三列可选为 edge weight

## 实现步骤

### Step 1. 读取 MAGMA 与统计控制项

1. 读取 `ZSTAT` 作为训练目标 `Y`
2. 读取 MAGMA covariates
3. 读取 error covariance
4. 对 error covariance 做正则化

### Step 2. 构造 PoPS 风格监督目标

1. 从 `Y` 中投影掉 covariates
2. 得到 `Y_proj`
3. `Y_proj` 作为后续训练标签

### Step 3. PoPS 风格特征筛选

1. 计算每个 feature 与 `Y_proj` 的边际关联
2. 按 `p-value` 和 `top-k` 过滤
3. 保留 trait-specific selected features

### Step 4. 全量特征矩阵预处理

1. 载入 selected feature matrix
2. 对所有能与 MAGMA 对齐的基因执行：
   - GLS whitening
   - centering / covariate projection
3. 避免训练节点和邻居节点处于不同特征空间

### Step 5. 图构建

1. 读取 STRING ENSG 边文件
2. 映射到 `rows.txt` 对应基因顺序
3. 过滤不存在的基因
4. 构建无向图
5. 根据参数执行 hub 控制
6. 添加 self-loop，保证孤立基因至少保留自身路径

### Step 5.1 Hub 控制参数化方案

当前工程把 hub 控制做成可切换参数，支持：

1. `none`
   - 不做拓扑裁剪
2. `topk_by_score`
   - 对每个节点只保留 top-k strongest neighbors
3. `mutual_topk_by_score`
   - 仅保留双方都互相进入 top-k 的边

另外还支持：

1. `--hub_topk`
2. `--edge_dropout`
3. `--graph_self_loops / --no_graph_self_loops`

### Step 6. 模型

模型由两个分支组成：

1. `base branch`
   - Ridge warm start
   - 对 selected features 直接打分
2. `graph residual branch`
   - 2-layer mean aggregation graph residual network
   - 学习 STRING 外部图上的结构化增量
   - 仅对图内基因生效，图外基因 residual 强制为 0

最终：

`final_score = base_score + residual_score`

### Step 7. 数据划分

1. 训练染色体由 `--training_chromosomes` 指定
2. 验证染色体由 `--validation_chromosomes` 指定
3. 若未指定验证染色体，则默认取训练集合中的最后一条染色体

### Step 8. 输出

1. `.preds`
   - gene score
2. `.marginals`
   - feature marginal association table
3. `.coefs`
   - route metadata

## 适用场景

1. 想优先验证“STRING 外部图结构是否能补充 PoPS”
2. 想把图增益解释成 residual effect
3. 想先得到最稳健的一版 GNN 增强方案

## 关键优点

1. 不破坏 PoPS 主体统计框架
2. 图信息来源独立
3. 对孤立基因更友好
4. 最容易和 MLP / shuffled graph 做公平对照
