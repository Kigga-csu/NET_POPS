# SEARCH_PLAN

## 目标

在 validation set 上用 `monitor_loss` 选择 direct pipeline 的更优配置。

## 第一阶段建议搜索空间

- `model_type`
  - `mlp`
  - `graphsage`
  - `gcn`
  - `appnp`
- `loss_type`
  - `mse`
  - `huber`
- `hub_topk`
  - `32`
  - `128`
  - `-1`

## 第二阶段可扩展

- `gat`
- `pairwise_rank`
- `hybrid_mse_rank`
- `hidden_dim`
- `dropout`
- `edge_dropout`
- `target_protocol`
- `feature_selection_mode`

## 输出

- `search_results.tsv`
- 各配置输出目录
- 对应 `.meta.tsv` 和 `.history.tsv`
