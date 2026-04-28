import argparse
import logging

import numpy as np
import torch

from data import load_external_graph
from model import ExternalGraphResidualModel
from trainer import predict_scores, train_model
from utils import (
    align_targets_to_rows,
    build_split_masks,
    compute_marginal_assoc,
    get_device,
    get_gene_indices_to_use,
    load_feature_matrix,
    natural_key,
    process_feature_matrix_gls_center,
    project_out_covariates,
    read_gene_annot_df,
    read_magma,
    regularize_error_cov,
    save_run_outputs,
    select_features_from_marginal_assoc_df,
)


def parse_args():
    parser = argparse.ArgumentParser(description="STRING external graph residual GraphSAGE on top of PoPS-style features.")
    parser.add_argument("--gene_annot_path", required=True)
    parser.add_argument("--feature_mat_prefix", required=True)
    parser.add_argument("--num_feature_chunks", type=int, required=True)
    parser.add_argument("--magma_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--external_graph_path", required=True)
    parser.add_argument("--external_graph_delimiter", default="\t")
    parser.add_argument("--external_graph_min_weight", type=float)
    parser.add_argument(
        "--hub_strategy",
        default="topk_by_score",
        choices=["none", "topk_by_score", "mutual_topk_by_score"],
        help="Topology-side hub control strategy.",
    )
    parser.add_argument("--hub_topk", type=int, default=64, help="Used by hub control strategies based on top-k pruning.")
    parser.add_argument("--no_graph_self_loops", dest="graph_self_loops", action="store_false")
    parser.set_defaults(graph_self_loops=True)
    parser.add_argument("--control_features_path")
    parser.add_argument("--subset_features_path")
    parser.add_argument("--feature_selection_p_cutoff", type=float, default=0.05)
    parser.add_argument("--feature_selection_max_num", type=int)
    parser.add_argument("--training_chromosomes", nargs="*")
    parser.add_argument("--validation_chromosomes", nargs="*")
    parser.add_argument("--ignore_magma_covariates", dest="use_magma_covariates", action="store_false")
    parser.set_defaults(use_magma_covariates=True)
    parser.add_argument("--ignore_magma_error_cov", dest="use_magma_error_cov", action="store_false")
    parser.set_defaults(use_magma_error_cov=True)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge_dropout", type=float, default=0.1)
    parser.add_argument("--residual_alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = get_device()

    gene_annot_df = read_gene_annot_df(args.gene_annot_path)
    all_chromosomes = sorted(gene_annot_df.CHR.unique(), key=natural_key)
    training_chromosomes = args.training_chromosomes or all_chromosomes

    y, covariates, error_cov, y_ids = read_magma(
        args.magma_prefix, args.use_magma_covariates, args.use_magma_error_cov
    )
    if error_cov is not None:
        error_cov = regularize_error_cov(error_cov, y, y_ids, gene_annot_df)

    cov_keep_inds = get_gene_indices_to_use(y_ids, gene_annot_df, all_chromosomes, True)
    y_proj = project_out_covariates(y, covariates, error_cov, y_ids, gene_annot_df, cov_keep_inds) if covariates is not None else y

    feature_selection_inds = get_gene_indices_to_use(y_ids, gene_annot_df, training_chromosomes, True)
    marginal_assoc_df = compute_marginal_assoc(
        args.feature_mat_prefix,
        args.num_feature_chunks,
        y_proj,
        y_ids,
        None,
        error_cov,
        gene_annot_df,
        feature_selection_inds,
    )
    selected_features = select_features_from_marginal_assoc_df(
        marginal_assoc_df,
        args.subset_features_path,
        args.control_features_path,
        args.feature_selection_p_cutoff,
        args.feature_selection_max_num,
    )
    logging.info("Selected %d features", len(selected_features))

    x_all_np, full_cols, rows = load_feature_matrix(
        args.feature_mat_prefix, args.num_feature_chunks, selected_features
    )
    x_all_np = process_feature_matrix_gls_center(x_all_np, rows, y_ids, gene_annot_df, error_cov, covariates)
    y_full, matched_mask = align_targets_to_rows(rows, y_ids, y_proj)
    train_mask, val_mask, used_val_chrs = build_split_masks(
        rows,
        gene_annot_df,
        matched_mask,
        training_chromosomes,
        args.validation_chromosomes,
        True,
    )
    if train_mask.sum() == 0:
        raise ValueError("No training genes available after split construction.")

    x_all = torch.from_numpy(x_all_np.astype(np.float32)).to(device)
    edge_index, graph_mask, graph_metadata = load_external_graph(
        rows,
        args.external_graph_path,
        device,
        delimiter=args.external_graph_delimiter,
        min_weight=args.external_graph_min_weight,
        add_self_loops=args.graph_self_loops,
        hub_strategy=args.hub_strategy,
        hub_topk=args.hub_topk,
    )
    batch = {"x_all": x_all, "edge_index": edge_index, "graph_mask": graph_mask}

    model = ExternalGraphResidualModel(
        in_dim=x_all.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        residual_alpha=args.residual_alpha,
    ).to(device)
    model, best_loss = train_model(
        model,
        batch,
        y_full,
        train_mask,
        val_mask,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
    )
    scores = predict_scores(model, batch)
    metadata = {
        "ROUTE": "string_external_graph_residual_gnn",
        "NUM_SELECTED_FEATURES": len(selected_features),
        "NUM_NODES": len(rows),
        "NUM_EDGES": edge_index.shape[1],
        "TRAIN_GENES": int(train_mask.sum()),
        "VAL_GENES": int(val_mask.sum()),
        "VALIDATION_CHROMOSOMES": ",".join(used_val_chrs),
        "BEST_MONITOR_LOSS": best_loss,
        "EXTERNAL_GRAPH_PATH": args.external_graph_path,
        "HUB_STRATEGY": args.hub_strategy,
        "HUB_TOPK": args.hub_topk,
        "GRAPH_SELF_LOOPS": args.graph_self_loops,
        "EDGE_DROPOUT": args.edge_dropout,
        "RESIDUAL_ALPHA": args.residual_alpha,
        "HIDDEN_DIM": args.hidden_dim,
        "WARMUP_EPOCHS": args.warmup_epochs,
        "PATIENCE": args.patience,
        "EVAL_EVERY": args.eval_every,
        "GRAPH_MASK_GENES": int(graph_mask.sum().item()),
        **graph_metadata,
    }
    save_run_outputs(args.out_prefix, rows, scores, marginal_assoc_df, selected_features, metadata)


if __name__ == "__main__":
    main()
