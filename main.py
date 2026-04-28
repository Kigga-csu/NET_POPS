import argparse
import logging

import numpy as np
import torch

from data import load_external_graph
from model import build_model
from trainer import predict_scores, train_model
from utils import (
    align_targets_to_rows,
    align_block_ids_to_rows,
    build_magma_corr_blocks,
    build_pairwise_groups_from_block_ids,
    build_split_masks,
    compute_marginal_assoc,
    get_device,
    get_gene_indices_to_use,
    load_feature_matrix,
    natural_key,
    process_feature_matrix_gls_center,
    read_gene_annot_df,
    read_magma,
    regularize_error_cov,
    resolve_target,
    save_run_outputs,
    select_features_from_marginal_assoc_df,
)
from xgb_utils import train_xgboost_model


def parse_args():
    parser = argparse.ArgumentParser(description="Direct STRING graph + PoPS feature pipeline for MAGMA score fitting.")
    parser.add_argument("--gene_annot_path", required=True)
    parser.add_argument("--feature_mat_prefix", required=True)
    parser.add_argument("--num_feature_chunks", type=int, required=True)
    parser.add_argument("--magma_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--external_graph_path", required=True)
    parser.add_argument("--external_graph_delimiter", default="\t")
    parser.add_argument("--external_graph_min_weight", type=float)
    parser.add_argument(
        "--target_protocol",
        default="zstat",
        choices=["zstat", "cov_projected", "cov_projected_gls"],
    )
    parser.add_argument(
        "--feature_selection_mode",
        default="train_marginal",
        choices=["none", "train_marginal"],
    )
    parser.add_argument("--control_features_path")
    parser.add_argument("--subset_features_path")
    parser.add_argument("--feature_selection_p_cutoff", type=float, default=0.05)
    parser.add_argument("--feature_selection_max_num", type=int)
    parser.add_argument("--training_chromosomes", nargs="*")
    parser.add_argument("--validation_chromosomes", nargs="*")
    parser.add_argument(
        "--model_type",
        default="graphsage",
        choices=["mlp", "graphsage", "gat", "gcn", "appnp", "xgboost"],
    )
    parser.add_argument(
        "--fusion_mode",
        default="pure",
        choices=["pure", "add", "residual"],
    )
    parser.add_argument(
        "--loss_type",
        default="mse",
        choices=["mse", "huber", "pairwise_rank", "hybrid_mse_rank"],
    )
    parser.add_argument(
        "--hub_strategy",
        default="topk_by_score",
        choices=["none", "topk_by_score", "mutual_topk_by_score"],
    )
    parser.add_argument("--hub_topk", type=int, default=128)
    parser.add_argument("--no_graph_self_loops", dest="graph_self_loops", action="store_false")
    parser.set_defaults(graph_self_loops=True)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge_dropout", type=float, default=0.1)
    parser.add_argument("--appnp_steps", type=int, default=10)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--pairwise_margin", type=float, default=0.2)
    parser.add_argument("--pairwise_min_target_diff", type=float, default=0.0)
    parser.add_argument("--pairwise_sample_mode", choices=["all", "balanced"], default="balanced")
    parser.add_argument("--pairwise_max_pairs_per_locus", type=int, default=256)
    parser.add_argument("--magma_block_corr_threshold", type=float, default=0.05)
    parser.add_argument("--reg_loss_weight", type=float, default=1.0)
    parser.add_argument("--rank_loss_weight", type=float, default=0.2)
    parser.add_argument("--xgb_auto_tune", action="store_true")
    parser.add_argument("--xgb_search_iter", type=int, default=20)
    parser.add_argument("--xgb_n_jobs", type=int, default=1)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
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
    if args.model_type == "xgboost" and args.loss_type not in {"mse", "huber"}:
        raise ValueError("model_type=xgboost currently only supports loss_type=mse or huber.")
    if args.model_type == "xgboost" and args.fusion_mode != "pure":
        raise ValueError("model_type=xgboost currently only supports fusion_mode=pure.")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = get_device()

    gene_annot_df = read_gene_annot_df(args.gene_annot_path)
    all_chromosomes = sorted(gene_annot_df.CHR.unique(), key=natural_key)
    training_chromosomes = args.training_chromosomes or all_chromosomes

    need_covariates = args.target_protocol in {"cov_projected", "cov_projected_gls"}
    need_error_cov = args.target_protocol == "cov_projected_gls"
    y, covariates, error_cov, y_ids = read_magma(
        args.magma_prefix,
        need_covariates=need_covariates,
        need_error_cov=need_error_cov,
    )
    if error_cov is not None:
        error_cov = regularize_error_cov(error_cov, y, y_ids, gene_annot_df)

    target_values = resolve_target(
        args.target_protocol,
        y,
        y_ids,
        covariates,
        error_cov,
        gene_annot_df,
        training_chromosomes,
    )

    marginal_assoc_df = None
    if args.feature_selection_mode == "train_marginal":
        feature_selection_inds = get_gene_indices_to_use(y_ids, gene_annot_df, training_chromosomes, True)
        marginal_assoc_df = compute_marginal_assoc(
            args.feature_mat_prefix,
            args.num_feature_chunks,
            target_values,
            y_ids,
            None,
            error_cov if args.target_protocol == "cov_projected_gls" else None,
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
    else:
        if args.subset_features_path is not None:
            selected_features = np.loadtxt(args.subset_features_path, dtype=str).flatten().tolist()
        else:
            selected_features = None
    logging.info("Feature selection mode=%s selected=%s", args.feature_selection_mode, "ALL" if selected_features is None else len(selected_features))

    x_all_np, full_cols, rows = load_feature_matrix(
        args.feature_mat_prefix, args.num_feature_chunks, selected_features
    )
    if x_all_np.shape[1] == 0:
        raise ValueError("No features remain after feature loading/selection.")
    if args.target_protocol in {"cov_projected", "cov_projected_gls"}:
        proc_error_cov = error_cov if args.target_protocol == "cov_projected_gls" else None
        x_all_np = process_feature_matrix_gls_center(x_all_np, rows, y_ids, gene_annot_df, proc_error_cov, covariates)

    y_full, matched_mask = align_targets_to_rows(rows, y_ids, target_values)
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

    graph_metadata = {
        "GRAPH_GENES": 0,
        "GRAPH_OVERLAP_GENES": 0,
        "GRAPH_EDGE_COUNT_AFTER_CONTROL": 0,
    }

    block_gene_ids, block_ids_y, block_metadata = build_magma_corr_blocks(
        args.magma_prefix + ".genes.raw",
        corr_abs_threshold=args.magma_block_corr_threshold,
    )
    block_ids_full = align_block_ids_to_rows(rows, block_gene_ids, block_ids_y)
    pairwise_groups = build_pairwise_groups_from_block_ids(block_ids_full)
    logging.info(
        "Magma correlation blocks: blocks=%d singleton_blocks=%d max_size=%d",
        block_metadata["MAGMA_BLOCK_NUM_BLOCKS"],
        block_metadata["MAGMA_BLOCK_SINGLETON_BLOCKS"],
        block_metadata["MAGMA_BLOCK_MAX_SIZE"],
    )

    if args.model_type == "xgboost":
        scores_model, scores, xgb_metadata, history = train_xgboost_model(
            x_all_np=x_all_np.astype(np.float32),
            y_full=y_full.astype(np.float32),
            train_mask=train_mask,
            val_mask=val_mask,
            loss_type=args.loss_type,
            random_seed=args.random_seed,
            n_jobs=args.xgb_n_jobs,
            tree_method=args.xgb_tree_method,
            auto_tune=args.xgb_auto_tune,
            search_iter=args.xgb_search_iter,
        )
        best_loss = xgb_metadata["BEST_MONITOR_LOSS"]
        best_epoch = -1
        graph_mask_genes = 0
        num_edges = 0
    else:
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
        batch = {
            "x_all": x_all,
            "edge_index": edge_index,
            "graph_mask": graph_mask,
            "pairwise_groups": pairwise_groups,
        }
        model = build_model(
            model_type=args.model_type,
            in_dim=x_all.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            edge_dropout=args.edge_dropout,
            num_layers=args.num_layers,
            appnp_steps=args.appnp_steps,
            appnp_alpha=args.appnp_alpha,
            fusion_mode=args.fusion_mode,
        ).to(device)
        model, best_loss, best_epoch, history = train_model(
            model,
            batch,
            y_full,
            train_mask,
            val_mask,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            pairwise_margin=args.pairwise_margin,
            pairwise_min_target_diff=args.pairwise_min_target_diff,
            pairwise_sample_mode=args.pairwise_sample_mode,
            pairwise_max_pairs_per_locus=args.pairwise_max_pairs_per_locus,
            reg_loss_weight=args.reg_loss_weight,
            rank_loss_weight=args.rank_loss_weight,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            eval_every=args.eval_every,
            fusion_mode=args.fusion_mode,
        )
        scores = predict_scores(model, batch)
        xgb_metadata = {}
        graph_mask_genes = int(graph_mask.sum().item())
        num_edges = int(edge_index.shape[1])
    metadata = {
        "ROUTE": "string_graph_direct_magma_gnn",
        "MODEL_TYPE": args.model_type,
        "FUSION_MODE": args.fusion_mode,
        "LOSS_TYPE": args.loss_type,
        "TARGET_PROTOCOL": args.target_protocol,
        "FEATURE_SELECTION_MODE": args.feature_selection_mode,
        "NUM_SELECTED_FEATURES": x_all_np.shape[1],
        "NUM_NODES": len(rows),
        "NUM_EDGES": num_edges,
        "TRAIN_GENES": int(train_mask.sum()),
        "VAL_GENES": int(val_mask.sum()),
        "VALIDATION_CHROMOSOMES": ",".join(used_val_chrs),
        "BEST_MONITOR_LOSS": best_loss,
        "BEST_EPOCH": best_epoch,
        "EXTERNAL_GRAPH_PATH": args.external_graph_path,
        "HUB_STRATEGY": args.hub_strategy,
        "HUB_TOPK": args.hub_topk,
        "GRAPH_SELF_LOOPS": args.graph_self_loops,
        "EDGE_DROPOUT": args.edge_dropout,
        "HIDDEN_DIM": args.hidden_dim,
        "NUM_LAYERS": args.num_layers,
        "XGB_AUTO_TUNE": args.xgb_auto_tune,
        "XGB_SEARCH_ITER": args.xgb_search_iter,
        "XGB_TREE_METHOD": args.xgb_tree_method,
        "PAIRWISE_MAX_PAIRS_PER_LOCUS": args.pairwise_max_pairs_per_locus,
        "PAIRWISE_MIN_TARGET_DIFF": args.pairwise_min_target_diff,
        "RANK_LOSS_WEIGHT": args.rank_loss_weight,
        "REG_LOSS_WEIGHT": args.reg_loss_weight,
        "MAGMA_BLOCK_CORR_THRESHOLD": args.magma_block_corr_threshold,
        "MAGMA_BLOCK_NUM_GROUPS_IN_ROWS": int(len(pairwise_groups)),
        "GRAPH_MASK_GENES": graph_mask_genes,
        **graph_metadata,
        **block_metadata,
        **xgb_metadata,
    }
    save_run_outputs(
        args.out_prefix,
        rows,
        scores,
        marginal_assoc_df,
        selected_features if selected_features is not None else full_cols,
        metadata,
        history,
    )


if __name__ == "__main__":
    main()
