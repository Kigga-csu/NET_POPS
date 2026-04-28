import json
import logging
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import torch
from sklearn.linear_model import LinearRegression


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(string_))]


def infer_gcst_id(magma_prefix):
    base = os.path.basename(magma_prefix)
    match = re.search(r"(GCST\d+)", base)
    if match:
        return match.group(1)
    parent = os.path.basename(os.path.dirname(magma_prefix))
    match = re.search(r"(GCST\d+)", parent)
    if match:
        return match.group(1)
    raise ValueError(f"Could not infer GCST id from magma_prefix: {magma_prefix}")


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def read_gene_annot_df(gene_annot_path):
    gene_annot_df = pd.read_csv(gene_annot_path, sep=r"\s+").set_index("ENSGID")
    gene_annot_df["CHR"] = gene_annot_df["CHR"].astype(str)
    return gene_annot_df


def get_hla_genes(gene_annot_df):
    sub_df = gene_annot_df[gene_annot_df.CHR == "6"]
    sub_df = sub_df[(sub_df.TSS >= 20 * (10 ** 6)) & (sub_df.TSS <= 40 * (10 ** 6))]
    return set(sub_df.index.values)


def get_gene_indices_to_use(y_ids, gene_annot_df, use_chrs, remove_hla):
    chr_gene_set = set(gene_annot_df[gene_annot_df.CHR.isin(use_chrs)].index.values)
    if remove_hla:
        hla_set = get_hla_genes(gene_annot_df)
        flags = [(gid in chr_gene_set) and (gid not in hla_set) for gid in y_ids]
    else:
        flags = [gid in chr_gene_set for gid in y_ids]
    return np.asarray(flags, dtype=bool)


def get_indices_in_target_order(ref_list, target_names):
    mapper = {name: idx for idx, name in enumerate(ref_list)}
    return np.asarray([mapper[name] for name in target_names], dtype=int)


def munge_magma_covariance_metadata(magma_raw_path):
    sigmas = []
    gene_metadata = []
    with open(magma_raw_path, "r", encoding="utf-8") as handle:
        lines = list(handle)[2:]
    rows = [np.asarray(line.strip("\n").split(" ")) for line in lines]
    all_chroms = np.asarray([row[1] for row in rows])
    seq_breaks = np.where(all_chroms[:-1] != all_chroms[1:])[0]
    assert len(seq_breaks) == len(set(all_chroms)) - 1, "Chromosomes are not sequentially ordered."

    current_chr = rows[0][1]
    current_idx = 0
    current_chr_gene_count = sum(1 for row in rows if row[1] == current_chr)
    current_sigma = np.zeros((current_chr_gene_count, current_chr_gene_count))
    current_meta = []
    for row in rows:
        if row[1] != current_chr:
            sigmas.append(current_sigma + current_sigma.T + np.eye(current_sigma.shape[0]))
            gene_metadata.append(current_meta)
            current_chr = row[1]
            current_idx = 0
            current_chr_gene_count = sum(1 for item in rows if item[1] == current_chr)
            current_sigma = np.zeros((current_chr_gene_count, current_chr_gene_count))
            current_meta = []
        current_meta.append([row[0], float(row[4]), float(row[5]), float(row[7])])
        if len(row) > 9:
            gene_corrs = np.asarray([float(val) for val in row[9:]])
            current_sigma[current_idx, current_idx - gene_corrs.shape[0] : current_idx] = gene_corrs
        current_idx += 1

    sigmas.append(current_sigma + current_sigma.T + np.eye(current_sigma.shape[0]))
    gene_metadata.append(current_meta)
    gene_metadata = pd.DataFrame(np.vstack(gene_metadata), columns=["GENE", "NSNPS", "NPARAM", "MAC"])
    gene_metadata.NSNPS = gene_metadata.NSNPS.astype(np.float64)
    gene_metadata.NPARAM = gene_metadata.NPARAM.astype(np.float64)
    gene_metadata.MAC = gene_metadata.MAC.astype(np.float64)
    return sigmas, gene_metadata


def build_control_covariates(metadata):
    gene_size = metadata.NPARAM.values
    gene_density = metadata.NPARAM.values / metadata.NSNPS.values
    gene_density[metadata.NSNPS.values == 0] = 0.0
    inverse_mac = 1.0 / metadata.MAC.values
    cov = np.stack(
        (
            gene_size,
            np.log(np.maximum(gene_size, 1e-9)),
            gene_density,
            np.log(np.maximum(gene_density, 1e-9)),
            inverse_mac,
            np.log(np.maximum(inverse_mac, 1e-9)),
        ),
        axis=1,
    )
    cov_df = pd.DataFrame(
        cov,
        columns=[
            "gene_size",
            "log_gene_size",
            "gene_density",
            "log_gene_density",
            "inverse_mac",
            "log_inverse_mac",
        ],
    )
    cov_df["GENE"] = metadata.GENE.values
    cov_df = cov_df.set_index("GENE")
    return cov_df


def read_magma(magma_prefix, need_covariates=False, need_error_cov=False):
    magma_df = pd.read_csv(magma_prefix + ".genes.out", sep=r"\s+")
    y = magma_df.ZSTAT.values.astype(np.float64)
    y_ids = magma_df.GENE.values
    covariates = None
    error_cov = None
    if need_covariates or need_error_cov:
        sigmas, gene_metadata = munge_magma_covariance_metadata(magma_prefix + ".genes.raw")
        if need_covariates:
            cov_df = build_control_covariates(gene_metadata)
            assert (cov_df.index.values == y_ids).all(), "Covariate ids and Y ids do not match."
            covariates = cov_df.values.astype(np.float64)
        if need_error_cov:
            error_cov = scipy.linalg.block_diag(*sigmas).astype(np.float64)
    return y, covariates, error_cov, y_ids


def block_Linv(matrix, block_labels):
    block_labels = np.asarray(block_labels)
    linv = np.zeros(matrix.shape, dtype=np.float64)
    for label in sorted(set(block_labels), key=natural_key):
        mask = block_labels == label
        sub_matrix = matrix[np.ix_(mask, mask)]
        linv[np.ix_(mask, mask)] = np.linalg.inv(np.linalg.cholesky(sub_matrix))
    return linv


def block_AB(matrix_a, block_labels, matrix_b):
    block_labels = np.asarray(block_labels)
    new_b = np.zeros(matrix_b.shape, dtype=np.float64)
    for label in sorted(set(block_labels), key=natural_key):
        mask = block_labels == label
        new_b[mask] = matrix_a[np.ix_(mask, mask)].dot(matrix_b[mask])
    return new_b


def project_out_V(matrix_m, matrix_v):
    gram_inv = np.linalg.inv(matrix_v.T.dot(matrix_v))
    betas = gram_inv.dot(matrix_v.T.dot(matrix_m))
    return matrix_m - matrix_v.dot(betas)


def regularize_error_cov(error_cov, y, y_ids, gene_annot_df):
    y_chr = gene_annot_df.reindex(y_ids).CHR.fillna("0").values
    min_lambda = 0.0
    for chrom in sorted(set(y_chr), key=natural_key):
        mask = y_chr == chrom
        eigvals = np.linalg.eigvalsh(error_cov[np.ix_(mask, mask)])
        min_lambda = min(min_lambda, eigvals.min())
    ridge = abs(min(min_lambda, 0.0)) + 0.05 + 0.9 * max(0.0, np.var(y) - 1.0)
    return error_cov + np.eye(error_cov.shape[0]) * ridge


def project_out_covariates(y, covariates, error_cov, y_ids, gene_annot_df, keep_inds):
    if not np.isclose(covariates.var(axis=0), 0.0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0], 1), dtype=np.float64)))
    x_train = covariates[keep_inds]
    y_train = y[keep_inds]
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(keep_inds, keep_inds)]
        sub_labels = gene_annot_df.reindex(y_ids[keep_inds]).CHR.fillna("0").values
        linv = block_Linv(sub_error_cov, sub_labels)
        x_train = block_AB(linv, sub_labels, x_train)
        y_train = block_AB(linv, sub_labels, y_train)
    reg = LinearRegression(fit_intercept=False).fit(x_train, y_train)
    return y - reg.predict(covariates)


def batch_marginal_ols(y, x):
    old_settings = np.seterr(divide="ignore")
    sum_sq_x = np.sum(np.square(x), axis=0)
    near_const = np.isclose(sum_sq_x, 0.0)
    sum_sq_safe = sum_sq_x.copy()
    sum_sq_safe[near_const] = 1.0
    betas = y.dot(x) / sum_sq_safe
    mse = np.mean(np.square(y.reshape(-1, 1) - x * betas), axis=0)
    se = np.sqrt(mse / sum_sq_safe)
    z_scores = betas / se
    chi2 = np.square(z_scores)
    pvals = scipy.stats.chi2.sf(chi2, 1)
    r2 = 1.0 - (mse / np.var(y))
    betas[near_const] = np.nan
    se[near_const] = np.nan
    pvals[near_const] = np.nan
    r2[near_const] = np.nan
    np.seterr(**old_settings)
    return betas, se, pvals, r2


def compute_marginal_assoc(
    feature_mat_prefix,
    num_feature_chunks,
    y,
    y_ids,
    covariates,
    error_cov,
    gene_annot_df,
    feature_selection_y_gene_inds,
):
    feature_genes = y_ids[feature_selection_y_gene_inds]
    sub_y = y[feature_selection_y_gene_inds]
    if covariates is not None and not np.isclose(covariates.var(axis=0), 0.0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0], 1), dtype=np.float64)))
    elif covariates is None:
        covariates = np.ones((y.shape[0], 1), dtype=np.float64)
    sub_cov = covariates[feature_selection_y_gene_inds]
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(feature_selection_y_gene_inds, feature_selection_y_gene_inds)]
        sub_labels = gene_annot_df.reindex(feature_genes).CHR.fillna("0").values
        linv = block_Linv(sub_error_cov, sub_labels)
        sub_y = block_AB(linv, sub_labels, sub_y)
        sub_cov = block_AB(linv, sub_labels, sub_cov)
    else:
        linv = None
        sub_labels = None
    sub_y = project_out_V(sub_y.reshape(-1, 1), sub_cov).flatten()
    rows = np.loadtxt(feature_mat_prefix + ".rows.txt", dtype=str).flatten()
    x_inds = get_indices_in_target_order(rows, feature_genes)
    assoc_data = []
    all_cols = []
    for chunk_idx in range(num_feature_chunks):
        mat = np.load(feature_mat_prefix + f".mat.{chunk_idx}.npy").astype(np.float64)
        mat = mat[x_inds]
        cols = np.loadtxt(feature_mat_prefix + f".cols.{chunk_idx}.txt", dtype=str).flatten()
        if error_cov is not None:
            mat = block_AB(linv, sub_labels, mat)
        mat = project_out_V(mat, sub_cov)
        assoc_data.append(np.vstack(batch_marginal_ols(sub_y, mat)).T)
        all_cols.append(cols)
    assoc_data = np.vstack(assoc_data)
    all_cols = np.hstack(all_cols)
    return pd.DataFrame(assoc_data, columns=["beta", "se", "pval", "r2"], index=all_cols)


def select_features_from_marginal_assoc_df(
    marginal_assoc_df,
    subset_features_path,
    control_features_path,
    feature_selection_p_cutoff,
    feature_selection_max_num,
):
    if subset_features_path is not None:
        subset_features = np.loadtxt(subset_features_path, dtype=str).flatten()
        marginal_assoc_df = marginal_assoc_df.loc[marginal_assoc_df.index.isin(subset_features)]
    control_df = None
    if control_features_path is not None:
        control_features = np.loadtxt(control_features_path, dtype=str).flatten()
        control_df = marginal_assoc_df[marginal_assoc_df.index.isin(control_features)]
        marginal_assoc_df = marginal_assoc_df[~marginal_assoc_df.index.isin(control_features)]
    if feature_selection_p_cutoff is not None:
        marginal_assoc_df = marginal_assoc_df[marginal_assoc_df.pval < feature_selection_p_cutoff]
    if feature_selection_max_num is not None:
        marginal_assoc_df = marginal_assoc_df.sort_values("pval").iloc[:feature_selection_max_num]
    selected_features = list(marginal_assoc_df.index.values)
    if control_df is not None:
        selected_features.extend(list(control_df.index.values))
    return selected_features


def load_feature_matrix(feature_mat_prefix, num_feature_chunks, selected_features):
    selected_set = set(selected_features) if selected_features is not None else None
    rows = np.loadtxt(feature_mat_prefix + ".rows.txt", dtype=str).flatten()
    mats = []
    all_cols = []
    for chunk_idx in range(num_feature_chunks):
        mat = np.load(feature_mat_prefix + f".mat.{chunk_idx}.npy").astype(np.float64)
        cols = np.loadtxt(feature_mat_prefix + f".cols.{chunk_idx}.txt", dtype=str).flatten()
        if selected_set is not None:
            keep = np.asarray([col in selected_set for col in cols], dtype=bool)
            mat = mat[:, keep]
            cols = cols[keep]
        mats.append(mat)
        all_cols.append(cols)
    return np.hstack(mats), np.hstack(all_cols), rows


def align_targets_to_rows(rows, y_ids, y_values):
    y_map = {gid: value for gid, value in zip(y_ids, y_values)}
    y_full = np.zeros(len(rows), dtype=np.float64)
    matched_mask = np.zeros(len(rows), dtype=bool)
    for idx, gid in enumerate(rows):
        if gid in y_map:
            y_full[idx] = y_map[gid]
            matched_mask[idx] = True
    return y_full, matched_mask


def process_feature_matrix_gls_center(x_all, rows, y_ids, gene_annot_df, error_cov, covariates):
    row_map = {gid: idx for idx, gid in enumerate(rows)}
    y_map = {gid: idx for idx, gid in enumerate(y_ids)}
    valid_genes = [gid for gid in rows if gid in y_map]
    if not valid_genes:
        return x_all.copy()
    valid_rows = np.asarray([row_map[gid] for gid in valid_genes], dtype=int)
    valid_y = np.asarray([y_map[gid] for gid in valid_genes], dtype=int)
    x_sub = x_all[valid_rows].astype(np.float64).copy()
    if error_cov is not None:
        sub_error_cov = error_cov[np.ix_(valid_y, valid_y)]
        sub_labels = gene_annot_df.reindex(valid_genes).CHR.fillna("0").values
        linv = block_Linv(sub_error_cov, sub_labels)
        x_sub = block_AB(linv, sub_labels, x_sub)
    if covariates is not None:
        proj = covariates[valid_y].astype(np.float64)
        if not np.isclose(proj.var(axis=0), 0.0).any():
            proj = np.hstack((proj, np.ones((proj.shape[0], 1), dtype=np.float64)))
        if error_cov is not None:
            proj = block_AB(linv, sub_labels, proj)
    else:
        proj = np.ones((x_sub.shape[0], 1), dtype=np.float64)
        if error_cov is not None:
            proj = block_AB(linv, sub_labels, proj)
    x_sub = project_out_V(x_sub, proj)
    x_processed = x_all.copy().astype(np.float64)
    x_processed[valid_rows] = x_sub
    return x_processed


def build_split_masks(rows, gene_annot_df, matched_mask, training_chromosomes, validation_chromosomes, remove_hla):
    row_chr = gene_annot_df.reindex(rows).CHR.fillna("NA").values
    train_chr_set = set(training_chromosomes)
    if validation_chromosomes is None or len(validation_chromosomes) == 0:
        preferred = "22" if "22" in train_chr_set else sorted(train_chr_set, key=natural_key)[-1]
        validation_chromosomes = [preferred]
    val_chr_set = set(validation_chromosomes)
    hla_genes = get_hla_genes(gene_annot_df) if remove_hla else set()
    train_mask = np.zeros(len(rows), dtype=bool)
    val_mask = np.zeros(len(rows), dtype=bool)
    for idx, gid in enumerate(rows):
        if not matched_mask[idx]:
            continue
        if gid in hla_genes:
            continue
        chrom = row_chr[idx]
        if chrom in val_chr_set:
            val_mask[idx] = True
        elif chrom in train_chr_set:
            train_mask[idx] = True
    return train_mask, val_mask, validation_chromosomes


def resolve_target(target_protocol, y, y_ids, covariates, error_cov, gene_annot_df, training_chromosomes):
    if target_protocol == "zstat":
        return y
    cov_keep_inds = get_gene_indices_to_use(y_ids, gene_annot_df, training_chromosomes, True)
    if covariates is None:
        raise ValueError(f"target_protocol={target_protocol} requires MAGMA covariates, but they were not loaded.")
    if target_protocol == "cov_projected":
        return project_out_covariates(y, covariates, None, y_ids, gene_annot_df, cov_keep_inds)
    if target_protocol == "cov_projected_gls":
        if error_cov is None:
            raise ValueError("target_protocol=cov_projected_gls requires error covariance, but it was not loaded.")
        return project_out_covariates(y, covariates, error_cov, y_ids, gene_annot_df, cov_keep_inds)
    raise ValueError(f"Unsupported target_protocol: {target_protocol}")


def load_trait_mapping(mapping_path):
    df = pd.read_csv(mapping_path, sep="\t")
    gcst_col = None
    for col in ["DataSourceLDSC", "Data source (LDSC)", "Data_source_(LDSC)", "GCST", "GCST_ID", "gcst"]:
        if col in df.columns:
            gcst_col = col
            break
    if gcst_col is None:
        raise ValueError("Could not find a GCST column in trait mapping file.")

    pheno_col = None
    for col in ["trait", "Trait", "Phenotype", "phenotype"]:
        if col in df.columns:
            pheno_col = col
            break
    if pheno_col is None:
        raise ValueError("Could not find a phenotype column in trait mapping file.")

    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row[gcst_col]).strip()] = str(row[pheno_col]).strip().lower()
    return mapping


def build_magma_corr_blocks(magma_raw_path, corr_abs_threshold=0.0):
    with open(magma_raw_path, "r", encoding="utf-8") as handle:
        lines = list(handle)[2:]

    all_gene_ids = []
    all_block_ids = []
    block_sizes = []
    total_edges = 0

    current_chr = None
    chrom_genes = []
    chrom_corr_rows = []

    def finalize_chromosome(chromosome, genes, corr_rows):
        nonlocal total_edges
        if not genes:
            return
        dsu = DisjointSet(len(genes))
        for current_idx, corr_values in enumerate(corr_rows):
            if len(corr_values) == 0:
                continue
            prev_start = current_idx - len(corr_values)
            for offset, corr in enumerate(corr_values):
                if abs(corr) > corr_abs_threshold:
                    dsu.union(current_idx, prev_start + offset)
                    total_edges += 1
        root_to_block = {}
        block_ids_local = []
        for idx in range(len(genes)):
            root = dsu.find(idx)
            if root not in root_to_block:
                root_to_block[root] = len(root_to_block)
            block_ids_local.append(root_to_block[root])
        counts = pd.Series(block_ids_local).value_counts().sort_index()
        base = len(block_sizes)
        all_gene_ids.extend(genes)
        all_block_ids.extend([base + bid for bid in block_ids_local])
        block_sizes.extend(counts.tolist())

    for line in lines:
        row = np.asarray(line.strip("\n").split(" "))
        gene_id = str(row[0])
        chrom = str(row[1])
        corr_values = np.asarray([float(val) for val in row[9:]], dtype=np.float64) if len(row) > 9 else np.asarray([], dtype=np.float64)
        if current_chr is None:
            current_chr = chrom
        if chrom != current_chr:
            finalize_chromosome(current_chr, chrom_genes, chrom_corr_rows)
            current_chr = chrom
            chrom_genes = []
            chrom_corr_rows = []
        chrom_genes.append(gene_id)
        chrom_corr_rows.append(corr_values)
    finalize_chromosome(current_chr, chrom_genes, chrom_corr_rows)

    metadata = {
        "MAGMA_BLOCK_NUM_BLOCKS": int(len(set(all_block_ids))),
        "MAGMA_BLOCK_NUM_GENES": int(len(all_gene_ids)),
        "MAGMA_BLOCK_NUM_CORR_EDGES": int(total_edges),
        "MAGMA_BLOCK_MEAN_SIZE": float(np.mean(block_sizes)) if block_sizes else 0.0,
        "MAGMA_BLOCK_MEDIAN_SIZE": float(np.median(block_sizes)) if block_sizes else 0.0,
        "MAGMA_BLOCK_MAX_SIZE": int(max(block_sizes)) if block_sizes else 0,
        "MAGMA_BLOCK_SINGLETON_BLOCKS": int(sum(size == 1 for size in block_sizes)),
    }
    return np.asarray(all_gene_ids), np.asarray(all_block_ids, dtype=np.int64), metadata


def align_block_ids_to_rows(rows, block_gene_ids, block_ids):
    block_map = {gid: block_id for gid, block_id in zip(block_gene_ids, block_ids)}
    full_block_ids = np.full(len(rows), -1, dtype=np.int64)
    for idx, gid in enumerate(rows):
        if gid in block_map:
            full_block_ids[idx] = block_map[gid]
    return full_block_ids


def build_pairwise_groups_from_block_ids(block_ids):
    groups = []
    valid_block_ids = sorted(set(block_ids[block_ids >= 0].tolist()))
    for block_id in valid_block_ids:
        block_indices = np.where(block_ids == block_id)[0].tolist()
        groups.append({"block_id": int(block_id), "indices": block_indices})
    return groups


def _build_symbol_to_unique_ensg(gene_annot_df):
    symbol_to_ensgs = defaultdict(list)
    for ensg, row in gene_annot_df.reset_index()[["ENSGID", "NAME"]].itertuples(index=False):
        symbol_to_ensgs[str(row)].append(str(ensg))
    unique_map = {}
    ambiguous = set()
    for symbol, ensgs in symbol_to_ensgs.items():
        if len(ensgs) == 1:
            unique_map[symbol] = ensgs[0]
        else:
            ambiguous.add(symbol)
    return unique_map, ambiguous


def build_locus_supervision(
    rows,
    gene_annot_df,
    benchmark_path,
    trait_mapping_path,
    gcst_id,
):
    benchmark_df = pd.read_csv(benchmark_path, sep="\t")
    trait_mapping = load_trait_mapping(trait_mapping_path)
    if gcst_id not in trait_mapping:
        raise ValueError(f"GCST id {gcst_id} not found in trait mapping file.")
    phenotype = trait_mapping[gcst_id]
    bench_sub = benchmark_df[benchmark_df["phenotype"].astype(str).str.lower() == phenotype].copy()
    if bench_sub.empty:
        raise ValueError(f"No benchmark rows found for phenotype '{phenotype}'.")

    unique_map, ambiguous_symbols = _build_symbol_to_unique_ensg(gene_annot_df)
    row_map = {gid: idx for idx, gid in enumerate(rows)}
    tp_labels = np.full(len(rows), -1.0, dtype=np.float32)
    locus_ids = np.full(len(rows), -1, dtype=np.int64)
    locus_name_to_id = {}
    ambiguous_count = 0
    missing_symbol_count = 0
    missing_row_count = 0

    for _, row in bench_sub.iterrows():
        symbol = str(row["symbol"])
        locus_name = str(row["filename"])
        if symbol in ambiguous_symbols:
            ambiguous_count += 1
            continue
        ensg = unique_map.get(symbol)
        if ensg is None:
            missing_symbol_count += 1
            continue
        row_idx = row_map.get(ensg)
        if row_idx is None:
            missing_row_count += 1
            continue
        if locus_name not in locus_name_to_id:
            locus_name_to_id[locus_name] = len(locus_name_to_id)
        locus_ids[row_idx] = locus_name_to_id[locus_name]
        tp_labels[row_idx] = float(row["TP"])

    valid_mask = locus_ids >= 0
    valid_loci = np.unique(locus_ids[valid_mask])
    metadata = {
        "SUPERVISION_GCST": gcst_id,
        "SUPERVISION_PHENOTYPE": phenotype,
        "SUPERVISION_VALID_GENES": int(valid_mask.sum()),
        "SUPERVISION_VALID_LOCI": int(valid_loci.shape[0]),
        "SUPERVISION_AMBIGUOUS_SYMBOL_ROWS_SKIPPED": int(ambiguous_count),
        "SUPERVISION_MISSING_SYMBOL_ROWS_SKIPPED": int(missing_symbol_count),
        "SUPERVISION_GENES_NOT_IN_ROWS_SKIPPED": int(missing_row_count),
    }
    if metadata["SUPERVISION_VALID_LOCI"] == 0:
        logging.warning("No valid loci survived supervision alignment for %s.", gcst_id)
    return tp_labels, locus_ids, metadata


def metadata_to_dataframe(metadata_dict):
    return pd.DataFrame({"parameter": list(metadata_dict.keys()), "value": list(metadata_dict.values())})


def save_run_outputs(out_prefix, rows, scores, marginal_assoc_df, selected_features, metadata, history_rows):
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    preds_df = pd.DataFrame({"ENSGID": rows, "PoPS_Score": scores})
    preds_df.to_csv(out_prefix + ".preds", sep="\t", index=False)
    if marginal_assoc_df is not None:
        marginal_assoc_df = marginal_assoc_df.copy()
        marginal_assoc_df["selected"] = marginal_assoc_df.index.isin(selected_features)
        marginal_assoc_df.to_csv(out_prefix + ".marginals", sep="\t")
    meta_df = metadata_to_dataframe(metadata)
    meta_df.to_csv(out_prefix + ".meta.tsv", sep="\t", index=False)
    meta_df.to_csv(out_prefix + ".coefs", sep="\t", index=False)
    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(out_prefix + ".history.tsv", sep="\t", index=False)
    logging.info("Outputs written to %s.*", out_prefix)
