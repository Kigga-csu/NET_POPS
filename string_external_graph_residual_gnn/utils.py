import logging
import os
import re

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


def read_magma(magma_prefix, use_magma_covariates, use_magma_error_cov):
    magma_df = pd.read_csv(magma_prefix + ".genes.out", sep=r"\s+")
    y = magma_df.ZSTAT.values.astype(np.float64)
    y_ids = magma_df.GENE.values
    covariates = None
    error_cov = None
    if use_magma_covariates or use_magma_error_cov:
        sigmas, gene_metadata = munge_magma_covariance_metadata(magma_prefix + ".genes.raw")
        if use_magma_covariates:
            cov_df = build_control_covariates(gene_metadata)
            assert (cov_df.index.values == y_ids).all(), "Covariate ids and Y ids do not match."
            covariates = cov_df.values.astype(np.float64)
        if use_magma_error_cov:
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


def metadata_to_dataframe(metadata_dict):
    return pd.DataFrame({"parameter": list(metadata_dict.keys()), "value": list(metadata_dict.values())})


def save_run_outputs(out_prefix, rows, scores, marginal_assoc_df, selected_features, metadata):
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    preds_df = pd.DataFrame({"ENSGID": rows, "PoPS_Score": scores})
    preds_df.to_csv(out_prefix + ".preds", sep="\t", index=False)
    if marginal_assoc_df is not None:
        marginal_assoc_df = marginal_assoc_df.copy()
        marginal_assoc_df["selected"] = marginal_assoc_df.index.isin(selected_features)
        marginal_assoc_df.to_csv(out_prefix + ".marginals", sep="\t")
    metadata_to_dataframe(metadata).to_csv(out_prefix + ".coefs", sep="\t", index=False)
    logging.info("Outputs written to %s.*", out_prefix)
