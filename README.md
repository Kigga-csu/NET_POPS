# string_graph_direct_magma_gnn

基于 STRING 外部图与 PoPS 特征，直接拟合 MAGMA gene score 的新 pipeline。

## 目标

- 不使用 PoPS 线性主干
- 只保留 PoPS 特征作为 node feature
- 使用 STRING gene-gene 图作为 graph structure
- 直接训练 `MLP / GraphSAGE / GAT / GCN / APPNP` 拟合 MAGMA 分数

## 主要特性

- `target_protocol` 可选：
  - `zstat`
  - `cov_projected`
  - `cov_projected_gls`
- `feature_selection_mode` 可选：
  - `none`
  - `train_marginal`
- `model_type` 可选：
  - `mlp`
  - `graphsage`
  - `gat`
  - `gcn`
  - `appnp`
  - `xgboost`
- `fusion_mode` 可选：
  - `pure`
  - `add`
  - `residual`
- `loss_type` 可选：
  - `mse`
  - `huber`
  - `pairwise_rank`
  - `hybrid_mse_rank`

## 安装

### pip

```bash
cd /Users/wangshixian/Documents/project/POP_NETS/pops/string_graph_direct_magma_gnn
pip install -r requirements.txt
```

### conda

```bash
cd /Users/wangshixian/Documents/project/POP_NETS/pops/string_graph_direct_magma_gnn
conda env create -f environment.yml
conda activate string_graph_direct_magma_gnn
```

## 运行示例

```bash
python /Users/wangshixian/Documents/project/POP_NETS/pops/string_graph_direct_magma_gnn/main.py \
  --gene_annot_path /Users/wangshixian/Documents/project/POP_NETS/pops/example/data/utils/gene_annot_jun10.txt \
  --feature_mat_prefix /Users/wangshixian/Documents/project/POPS_data_information/feature_munge/pops_features \
  --num_feature_chunks 116 \
  --magma_prefix /Volumes/WSX19819083255_data/data_genomic/MAGMA_result/pops_hg19_results_ExWAS/GCST006979/GCST006979_magma_res \
  --out_prefix /tmp/string_graph_direct_magma_gnn_demo/GCST006979 \
  --external_graph_path /Users/wangshixian/Documents/project/POPS_data_information/string_PPI/graph_build_outputs/run_ensembl_gene_20260414_113748/03_graph_score_ge_700/gene_graph_edges_score_gte_700.tsv \
  --model_type graphsage \
  --fusion_mode pure \
  --loss_type mse \
  --target_protocol zstat \
  --feature_selection_mode train_marginal \
  --hub_strategy topk_by_score \
  --hub_topk 128 \
  --verbose
```

## 融合模式

当前 pipeline 支持三种结构：

- `--fusion_mode pure`
  - 纯 direct 路线
  - 不使用 PoPS 风格线性主干
  - `score = branch_model(x, graph)`

- `--fusion_mode add`
  - 线性主干 + 分支直接相加
  - `score = base_score + branch_score`
  - 主干和分支共同拟合最终目标

- `--fusion_mode residual`
  - 线性主干 + 残差分支
  - 先 warm-start 一个 ridge 风格线性主干
  - 分支只拟合 `target - base_score`
  - 最终 `score = base_score + residual_score`

注意：

- `xgboost` 当前只支持 `--fusion_mode pure`
- `mlp / graphsage / gat / gcn / appnp` 支持三种融合方式

## 排序 loss 当前实现

当前实现中：

- `pairwise_rank`
- `hybrid_mse_rank`

不再使用 benchmark 的 `locus` / `TP` 作为训练监督。

现在的做法是：

1. 从 `MAGMA .genes.raw` 中解析 gene-gene correlation
2. 将基因划分为多个相关性 block
3. 在每个 block 内，根据 MAGMA target score 的相对大小构造 pairwise ranking 约束

也就是说，排序 loss 的监督来自：

- MAGMA block 结构
- MAGMA target 的局部相对顺序

而不是测试 benchmark 的因果标签。

默认使用：

- `--magma_block_corr_threshold 0.05`

这是因为 `0.0` 通常会把基因连成过大的粗块，不利于局部排序训练。

## XGBoost

现在新增了：

- `--model_type xgboost`

它是一个只使用特征、不使用图的强非线性 baseline。

### 自动调参

可以开启：

- `--xgb_auto_tune`

当前实现使用的是 `scikit-learn` 的随机搜索框架，配合固定的 train/validation split 做参数搜索，不是普通 K-fold。

相关参数：

- `--xgb_search_iter`
- `--xgb_n_jobs`
- `--xgb_tree_method`

目前 `xgboost` 分支只支持：

- `loss_type=mse`
- `loss_type=huber`

## 输出

- `.preds`
- `.marginals`
- `.meta.tsv`
- `.coefs`
- `.history.tsv`

`.preds` 保持和 compare 脚本兼容：

- 第一列：`ENSGID`
- 分数字段：`PoPS_Score`

## 搜索

- 常规批量运行：`run_exwas.sh`
- 小规模验证集搜索：`run_search.sh`
