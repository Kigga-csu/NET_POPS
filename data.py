import logging

import numpy as np
import pandas as pd
import torch


def _build_neighbor_dict(edge_df, row_map, min_weight):
    source_col = edge_df.columns[0]
    target_col = edge_df.columns[1]
    weight_col = edge_df.columns[2] if edge_df.shape[1] >= 3 else None
    neighbors = {idx: [] for idx in range(len(row_map))}
    for _, row in edge_df.iterrows():
        if weight_col is not None and min_weight is not None and row[weight_col] < min_weight:
            continue
        src = str(row[source_col])
        dst = str(row[target_col])
        if src not in row_map or dst not in row_map or src == dst:
            continue
        score = float(row[weight_col]) if weight_col is not None else 1.0
        src_idx = row_map[src]
        dst_idx = row_map[dst]
        neighbors[src_idx].append((dst_idx, score))
        neighbors[dst_idx].append((src_idx, score))
    return neighbors


def _apply_hub_strategy(neighbors, hub_strategy, hub_topk):
    if hub_strategy == "none":
        return neighbors

    ranked = {}
    for src_idx, items in neighbors.items():
        dedup = {}
        for dst_idx, score in items:
            dedup[dst_idx] = max(score, dedup.get(dst_idx, float("-inf")))
        ranked[src_idx] = sorted(dedup.items(), key=lambda x: (-x[1], x[0]))

    if hub_strategy == "topk_by_score":
        if hub_topk == -1:
            return ranked
        if hub_topk is None or hub_topk <= 0:
            raise ValueError("hub_topk must be -1 or > 0 for topk_by_score.")
        return {src_idx: items[:hub_topk] for src_idx, items in ranked.items()}

    if hub_strategy == "mutual_topk_by_score":
        if hub_topk == -1:
            topk_sets = {src_idx: {dst_idx for dst_idx, _ in items} for src_idx, items in ranked.items()}
            return {
                src_idx: [(dst_idx, score) for dst_idx, score in items if src_idx in topk_sets.get(dst_idx, set())]
                for src_idx, items in ranked.items()
            }
        if hub_topk is None or hub_topk <= 0:
            raise ValueError("hub_topk must be -1 or > 0 for mutual_topk_by_score.")
        topk_sets = {src_idx: {dst_idx for dst_idx, _ in items[:hub_topk]} for src_idx, items in ranked.items()}
        pruned = {}
        for src_idx, items in ranked.items():
            kept = []
            for dst_idx, score in items[:hub_topk]:
                if src_idx in topk_sets.get(dst_idx, set()):
                    kept.append((dst_idx, score))
            pruned[src_idx] = kept
        return pruned

    raise ValueError(f"Unsupported hub_strategy: {hub_strategy}")


def _neighbor_dict_to_edge_index(neighbors, num_nodes, device, add_self_loops):
    edge_pairs = []
    graph_mask = np.zeros(num_nodes, dtype=bool)
    for src_idx, items in neighbors.items():
        if items:
            graph_mask[src_idx] = True
        for dst_idx, _ in items:
            edge_pairs.append((src_idx, dst_idx))
    if add_self_loops:
        for node_idx in range(num_nodes):
            edge_pairs.append((node_idx, node_idx))
    if not edge_pairs:
        raise ValueError("External graph is empty after filtering.")
    edge_index = torch.tensor(np.asarray(edge_pairs, dtype=np.int64).T, device=device)
    graph_mask = torch.from_numpy(graph_mask.astype(np.float32)).to(device)
    return edge_index, graph_mask


def load_external_graph(
    rows,
    edge_path,
    device,
    delimiter="\t",
    add_self_loops=True,
    min_weight=None,
    hub_strategy="topk_by_score",
    hub_topk=128,
):
    row_map = {gid: idx for idx, gid in enumerate(rows)}
    if edge_path is None:
        logging.warning("No external graph file provided. Using self-loops only.")
        empty_neighbors = {idx: [] for idx in range(len(rows))}
        edge_index, graph_mask = _neighbor_dict_to_edge_index(empty_neighbors, len(rows), device, add_self_loops)
        metadata = {
            "GRAPH_GENES": 0,
            "GRAPH_OVERLAP_GENES": 0,
            "GRAPH_EDGE_COUNT_AFTER_CONTROL": int(edge_index.shape[1]),
        }
        return edge_index, graph_mask, metadata

    edge_df = pd.read_csv(edge_path, sep=delimiter)
    if edge_df.shape[1] < 2:
        raise ValueError("External graph file must contain at least two columns for source and target.")
    raw_neighbors = _build_neighbor_dict(edge_df, row_map, min_weight)
    pruned_neighbors = _apply_hub_strategy(raw_neighbors, hub_strategy, hub_topk)
    edge_index, graph_mask = _neighbor_dict_to_edge_index(pruned_neighbors, len(rows), device, add_self_loops)
    metadata = {
        "GRAPH_GENES": int(pd.unique(pd.concat([edge_df.iloc[:, 0], edge_df.iloc[:, 1]], ignore_index=True)).shape[0]),
        "GRAPH_OVERLAP_GENES": int(graph_mask.sum().item()),
        "GRAPH_EDGE_COUNT_AFTER_CONTROL": int(edge_index.shape[1]),
    }
    return edge_index, graph_mask, metadata
