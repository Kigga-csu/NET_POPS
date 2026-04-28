#!/bin/bash

source activate net_pops

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAGMA_BASE="/Volumes/WSX19819083255_data/data_genomic/MAGMA_result/pops_hg19_results_ExWAS"
FEATURE_PREFIX="/Users/wangshixian/Documents/project/POPS_data_information/feature_munge/pops_features"
GENE_ANNOT="/Users/wangshixian/Documents/project/POP_NETS/pops/example/data/utils/gene_annot_jun10.txt"
EXTERNAL_GRAPH="/Users/wangshixian/Documents/project/POPS_data_information/string_PPI/graph_build_outputs/run_ensembl_gene_20260414_113748/03_graph_score_ge_700/gene_graph_edges_score_gte_700.tsv"
OUTPUT_BASE="${SCRIPT_DIR}/output/direct_graph_magma"

NUM_CHUNKS=116
MODEL_TYPE="graphsage"
LOSS_TYPE="mse"
TARGET_PROTOCOL="zstat"
FEATURE_SELECTION_MODE="train_marginal"
HIDDEN_DIM=32
NUM_LAYERS=2
DROPOUT=0.2
EDGE_DROPOUT=0.1
HUB_STRATEGY="topk_by_score"
HUB_TOPK=128
EPOCHS=300

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

    python "${SCRIPT_DIR}/main.py" \
        --gene_annot_path "${GENE_ANNOT}" \
        --feature_mat_prefix "${FEATURE_PREFIX}" \
        --num_feature_chunks "${NUM_CHUNKS}" \
        --magma_prefix "${MAGMA_PREFIX}" \
        --out_prefix "${OUT_PREFIX}" \
        --external_graph_path "${EXTERNAL_GRAPH}" \
        --target_protocol "${TARGET_PROTOCOL}" \
        --feature_selection_mode "${FEATURE_SELECTION_MODE}" \
        --model_type "${MODEL_TYPE}" \
        --loss_type "${LOSS_TYPE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --num_layers "${NUM_LAYERS}" \
        --dropout "${DROPOUT}" \
        --edge_dropout "${EDGE_DROPOUT}" \
        --hub_strategy "${HUB_STRATEGY}" \
        --hub_topk "${HUB_TOPK}" \
        --feature_selection_p_cutoff 0.05 \
        --epochs "${EPOCHS}" \
        --verbose \
        2>&1 | tee "${OUT_DIR}/${TRAIT}.log"
done
