# PIPELINE

## 路线定义

`string_graph_direct_magma_gnn` 是一条 direct pipeline：

- 输入：PoPS features + STRING graph + MAGMA score
- 输出：direct gene score
- 不使用 PoPS 线性主干
- 不做 residual 拟合

当前工程已经扩展为统一结构框架，通过 `--fusion_mode` 控制：

- `pure`
  - 纯 direct 路线
  - 不使用 PoPS 风格线性主干

- `add`
  - 线性主干与分支输出直接相加
  - 共同拟合最终目标

- `residual`
  - 线性主干 warm-start 后提供基线
  - 分支显式拟合剩余误差

## 主流程

1. 读取 gene annotation
2. 读取 MAGMA `.genes.out/.genes.raw`
3. 按 `target_protocol` 生成监督目标
4. 按 `feature_selection_mode` 选择或保留特征
5. 读取 PoPS feature matrix
6. 读取 STRING ENSG 图并做 hub 控制
7. 构建 `MLP / GraphSAGE / GAT / GCN / APPNP`
8. 根据 `fusion_mode` 决定纯 direct / add / residual 训练方式
9. 用 `loss_type` 指定的损失训练
10. 通过 validation `monitor_loss` 做早停与模型选择
11. 输出 `.preds / .marginals / .meta.tsv / .history.tsv`

另外新增一条非图分支：

- `XGBoost`

它只使用 PoPS features，不使用 STRING 图，用于提供一个可自动调参的强非线性 baseline。

## target_protocol

- `zstat`
  - 直接拟合 MAGMA 原始 `ZSTAT`
- `cov_projected`
  - 先做 covariate projection
- `cov_projected_gls`
  - 做 covariate projection + error covariance 处理

## feature_selection_mode

- `none`
  - 不做 train-only marginal selection
- `train_marginal`
  - 仅在 training chromosomes 上做 marginal feature selection

## loss_type

- `mse`
  - 直接回归 MAGMA target
- `huber`
  - 更鲁棒的回归
- `pairwise_rank`
  - 在同一 MAGMA correlation block 内，让高 target 基因分数高于低 target 基因
- `hybrid_mse_rank`
  - `MSE + pairwise_rank`

## hub 控制

- `hub_strategy = none`
- `hub_strategy = topk_by_score`
- `hub_strategy = mutual_topk_by_score`

`hub_topk` 支持：

- `32`
- `128`
- `512`
- `-1` 表示不限制 top-k

## 关键边界条件

- ranking loss 不再依赖 benchmark TP
- `pairwise_rank / hybrid_mse_rank` 依赖 MAGMA `.genes.raw` 中的相关性结构
- 默认相关性阈值为 `0.05`
- 如果某个 MAGMA block 在当前 split 内只有 0 或 1 个有效基因，则该 block 会被跳过
- 如果某个 block 内所有 target 几乎相同，则不会产生有效排序 pair
- 图外基因不会删除，`MLP` 或图模型中的 node encoder 仍然可以输出分数

## XGBoost 分支

- `model_type = xgboost`
- 不使用图结构
- 只允许 `fusion_mode = pure`
- 只支持：
  - `loss_type = mse`
  - `loss_type = huber`
- 支持：
  - `--xgb_auto_tune`
  - `--xgb_search_iter`

当前自动调参实现基于固定 train/validation split 做随机搜索，而不是普通 K-fold。

## 输出兼容性

为了兼容 compare 脚本：

- `.preds` 第一列保持 `ENSGID`
- 分数字段保持 `PoPS_Score`
- 同时保留 `.coefs` 作为 metadata alias
