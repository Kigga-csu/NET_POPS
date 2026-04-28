#!/bin/bash

source activate net_pops

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAGMA_BASE="/Volumes/WSX19819083255_data/data_genomic/MAGMA_result/pops_hg19_results_ExWAS"
FEATURE_PREFIX="/Users/wangshixian/Documents/project/POPS_data_information/feature_munge/pops_features"
GENE_ANNOT="/Users/wangshixian/Documents/project/POP_NETS/pops/example/data/utils/gene_annot_jun10.txt"
EXTERNAL_GRAPH="/Users/wangshixian/Documents/project/POPS_data_information/string_PPI/graph_build_outputs/run_ensembl_gene_20260414_113748/03_graph_score_ge_700/gene_graph_edges_score_gte_700.tsv"
OUTPUT_BASE="${SCRIPT_DIR}/output/search_runs"
SEARCH_TSV="${OUTPUT_BASE}/search_results.tsv"

NUM_CHUNKS=116
TRAITS=("GCST006979" "GCST90245992" "GCST90019500" "GCST90002412")
MODEL_TYPES=("mlp" "graphsage" "gcn" "appnp" "xgboost")
LOSS_TYPES=("mse" "huber")
HUB_TOPKS=(32 128 -1)

mkdir -p "${OUTPUT_BASE}"
echo -e "gcst\tmodel_type\tloss_type\thub_topk\tbest_monitor_loss\tmeta_path" > "${SEARCH_TSV}"

for TRAIT in "${TRAITS[@]}"; do
    MAGMA_PREFIX="${MAGMA_BASE}/${TRAIT}/${TRAIT}_magma_res"
    if [ ! -f "${MAGMA_PREFIX}.genes.out" ]; then
        echo "Skip ${TRAIT}: MAGMA file missing"
        continue
    fi

    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
            for HUB_TOPK in "${HUB_TOPKS[@]}"; do
                OUT_DIR="${OUTPUT_BASE}/${TRAIT}/${MODEL_TYPE}_${LOSS_TYPE}_topk${HUB_TOPK}"
                OUT_PREFIX="${OUT_DIR}/${TRAIT}"
                mkdir -p "${OUT_DIR}"

                CMD=(
                    python "${SCRIPT_DIR}/main.py"
                    --gene_annot_path "${GENE_ANNOT}"
                    --feature_mat_prefix "${FEATURE_PREFIX}"
                    --num_feature_chunks "${NUM_CHUNKS}"
                    --magma_prefix "${MAGMA_PREFIX}"
                    --out_prefix "${OUT_PREFIX}"
                    --external_graph_path "${EXTERNAL_GRAPH}"
                    --target_protocol zstat
                    --feature_selection_mode train_marginal
                    --model_type "${MODEL_TYPE}"
                    --loss_type "${LOSS_TYPE}"
                    --hidden_dim 32
                    --num_layers 2
                    --dropout 0.2
                    --edge_dropout 0.1
                    --hub_strategy topk_by_score
                    --hub_topk "${HUB_TOPK}"
                    --feature_selection_max_num 256
                    --epochs 80
                    --warmup_epochs 10
                    --patience 8
                    --eval_every 5
                )
                if [ "${MODEL_TYPE}" = "xgboost" ]; then
                    CMD+=(--xgb_auto_tune --xgb_search_iter 20)
                fi
                "${CMD[@]}" >/tmp/string_graph_direct_magma_gnn_search.log 2>&1

                META_PATH="${OUT_PREFIX}.meta.tsv"
                if [ -f "${META_PATH}" ]; then
                    BEST_LOSS=$(python - <<PY
import pandas as pd
meta = pd.read_csv("${META_PATH}", sep="\t")
row = meta[meta["parameter"] == "BEST_MONITOR_LOSS"]
print(row["value"].iloc[0] if len(row) else "NA")
PY
)
                    echo -e "${TRAIT}\t${MODEL_TYPE}\t${LOSS_TYPE}\t${HUB_TOPK}\t${BEST_LOSS}\t${META_PATH}" >> "${SEARCH_TSV}"
                fi
            done
        done
    done
done

echo "Search summary written to ${SEARCH_TSV}"
