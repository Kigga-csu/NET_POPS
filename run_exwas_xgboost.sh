#!/bin/bash

source activate net_pops

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAGMA_BASE="/data/lab/wangshixian/MAGMA/result/pops_hg19_results_ExWAS"
FEATURE_PREFIX="/data/lab/wangshixian/POPS/results/AD_2019/pops_features/feature"
GENE_ANNOT="/data/lab/wangshixian/POPS/pops/example/data/utils/gene_annot_jun10.txt"
EXTERNAL_GRAPH="/data/lab/wangshixian/PryoGraph/data/string_PPI/gene_graph_edges_score_gte_700.tsv"

# 当前 xgboost 分支不使用 benchmark / trait_map 训练，但保留在这里方便后续扩展排序评估。
BENCHMARK="/Users/wangshixian/Documents/project/POP_NETS/pops/compare/data/results_9_pheno.tsv"
TRAIT_MAP="/data/lab/wangshixian/data/FLAMES_benchmark/ExWAS/GWAS_catalog_code_number.txt"
OUTPUT_BASE="${SCRIPT_DIR}/result/direct_graph_magma_xgboost_Adjust_parameters"

NUM_CHUNKS=116
MODEL_TYPE="xgboost"
LOSS_TYPE="mse"
TARGET_PROTOCOL="zstat"
FEATURE_SELECTION_MODE="train_marginal"
FEATURE_SELECTION_MAX_NUM=512

# XGBoost 自动调参
XGB_AUTO_TUNE=1
XGB_SEARCH_ITER=20
XGB_N_JOBS=4
XGB_TREE_METHOD="hist"

TRAITS=(
    "GCST006979"
    "GCST90002412"
    "GCST90013977"
    "GCST90019500"
    "GCST90019505"
    "GCST90026417"
    "GCST90029022"
    "GCST90179149"
    "GCST90245992"
)

for TRAIT in "${TRAITS[@]}"; do
    MAGMA_PREFIX="${MAGMA_BASE}/${TRAIT}/${TRAIT}_magma_res"
    OUT_DIR="${OUTPUT_BASE}/${TRAIT}"
    OUT_PREFIX="${OUT_DIR}/${TRAIT}"
    mkdir -p "${OUT_DIR}"

    if [ ! -f "${MAGMA_PREFIX}.genes.out" ]; then
        echo "Skip ${TRAIT}: MAGMA file missing"
        continue
    fi

    CMD=(
        python "${SCRIPT_DIR}/main.py"
        --gene_annot_path "${GENE_ANNOT}"
        --feature_mat_prefix "${FEATURE_PREFIX}"
        --num_feature_chunks "${NUM_CHUNKS}"
        --magma_prefix "${MAGMA_PREFIX}"
        --out_prefix "${OUT_PREFIX}"
        --external_graph_path "${EXTERNAL_GRAPH}"
        --model_type "${MODEL_TYPE}"
        --loss_type "${LOSS_TYPE}"
        --target_protocol "${TARGET_PROTOCOL}"
        --feature_selection_mode "${FEATURE_SELECTION_MODE}"
        --feature_selection_max_num "${FEATURE_SELECTION_MAX_NUM}"
        --xgb_n_jobs "${XGB_N_JOBS}"
        --xgb_tree_method "${XGB_TREE_METHOD}"
        --verbose
    )

    if [ "${XGB_AUTO_TUNE}" = "1" ]; then
        CMD+=(
            --xgb_auto_tune
            --xgb_search_iter "${XGB_SEARCH_ITER}"
        )
    fi

    "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/${TRAIT}.log"
done
