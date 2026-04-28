# STRING External Graph Residual GNN

这个工程实现的是一条固定路线：

- 外部 STRING ENSG 图
- PoPS 特征作为 node feature
- 图只做 residual
- 2-layer residual GraphSAGE
- hub 控制可通过参数切换

## 当前支持的 hub 控制方式

- `--hub_strategy none`
- `--hub_strategy topk_by_score`
- `--hub_strategy mutual_topk_by_score`

可配参数：

- `--hub_topk`
- `--edge_dropout`
- `--graph_self_loops / --no_graph_self_loops`

## 当前默认推荐参数

- `--hub_strategy topk_by_score`
- `--hub_topk 64`
- `--edge_dropout 0.1`

## 环境要求

推荐 Python 3.10。

这个工程当前依赖：

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`

安装方式二选一：

### pip

```bash
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate string_external_graph_residual_gnn
```

说明：

- 当前实现不依赖 `torch-geometric`
- 没有 GPU 也可以用 CPU 运行
- 如果你已经有 `net_pops` 环境，并且其中包含上述依赖，也可以直接复用

## 当前路线约束

- 图只做 residual
- 图外基因走 `base only`
- feature selection 只在 train split 内做
- validation 按 chromosome
- graph 固定，不随 split 改变
