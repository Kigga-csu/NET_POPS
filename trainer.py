import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, loss_value):
        if self.best_loss is None or loss_value < self.best_loss - self.min_delta:
            self.best_loss = loss_value
            self.counter = 0
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


def _masked_regression_loss(pred, target, mask, loss_type, huber_delta):
    if not mask.any():
        raise ValueError("Regression loss requested with empty mask.")
    if loss_type == "mse":
        return nn.functional.mse_loss(pred[mask], target[mask])
    if loss_type == "huber":
        return nn.functional.huber_loss(pred[mask], target[mask], delta=huber_delta)
    raise ValueError(f"Unsupported regression loss_type: {loss_type}")


def _sample_pairs(pos_idx, neg_idx, max_pairs, mode, device):
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None, None
    if mode == "all":
        pos_repeat = np.repeat(pos_idx, len(neg_idx))
        neg_tile = np.tile(neg_idx, len(pos_idx))
        if max_pairs is not None and len(pos_repeat) > max_pairs:
            choice = np.linspace(0, len(pos_repeat) - 1, num=max_pairs, dtype=int)
            pos_repeat = pos_repeat[choice]
            neg_tile = neg_tile[choice]
        return (
            torch.as_tensor(pos_repeat, device=device, dtype=torch.long),
            torch.as_tensor(neg_tile, device=device, dtype=torch.long),
        )

    max_pairs = max_pairs or min(len(pos_idx) * len(neg_idx), max(len(pos_idx), len(neg_idx)))
    rng = np.random.default_rng(42)
    pos_choice = rng.choice(pos_idx, size=max_pairs, replace=len(pos_idx) < max_pairs)
    neg_choice = rng.choice(neg_idx, size=max_pairs, replace=len(neg_idx) < max_pairs)
    return (
        torch.as_tensor(pos_choice, device=device, dtype=torch.long),
        torch.as_tensor(neg_choice, device=device, dtype=torch.long),
    )


def compute_pairwise_rank_loss(
    pred,
    target,
    pairwise_groups,
    mask_tensor,
    margin=0.2,
    min_target_diff=0.0,
    sample_mode="balanced",
    max_pairs_per_locus=256,
):
    losses = []
    valid_groups = 0
    device = pred.device
    for group_info in pairwise_groups:
        group_idx = torch.as_tensor(group_info["indices"], device=device, dtype=torch.long)
        group_idx = group_idx[mask_tensor[group_idx]]
        if group_idx.numel() < 2:
            continue
        group_target = target[group_idx]
        target_diff = group_target.unsqueeze(1) - group_target.unsqueeze(0)
        pos_local, neg_local = torch.where(target_diff > min_target_diff)
        if pos_local.numel() == 0:
            continue
        if max_pairs_per_locus is not None and pos_local.numel() > max_pairs_per_locus:
            if sample_mode == "all":
                keep = torch.linspace(0, pos_local.numel() - 1, steps=max_pairs_per_locus, device=device).long()
            else:
                keep = torch.randperm(pos_local.numel(), device=device)[:max_pairs_per_locus]
            pos_local = pos_local[keep]
            neg_local = neg_local[keep]
        pos_tensor = group_idx[pos_local]
        neg_tensor = group_idx[neg_local]
        score_diff = pred[pos_tensor] - pred[neg_tensor]
        if margin is not None:
            loss = torch.log1p(torch.exp(-(score_diff - margin))).mean()
        else:
            loss = torch.log1p(torch.exp(-score_diff)).mean()
        losses.append(loss)
        valid_groups += 1
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), valid_groups


def compute_objective(
    pred,
    target,
    mask_tensor,
    pairwise_loci,
    loss_type,
    huber_delta,
    margin,
    min_target_diff,
    sample_mode,
    max_pairs_per_locus,
    reg_loss_weight,
    rank_loss_weight,
    allow_pairwise_fallback=False,
):
    metrics = {}
    reg_mse = nn.functional.mse_loss(pred[mask_tensor], target[mask_tensor]) if mask_tensor.any() else None
    metrics["mse"] = reg_mse.item() if reg_mse is not None else np.nan

    if loss_type == "mse":
        total = _masked_regression_loss(pred, target, mask_tensor, "mse", huber_delta)
        metrics["monitor_name"] = "mse"
        metrics["total"] = total.item()
        return total, metrics

    if loss_type == "huber":
        total = _masked_regression_loss(pred, target, mask_tensor, "huber", huber_delta)
        metrics["monitor_name"] = "huber"
        metrics["total"] = total.item()
        return total, metrics

    pair_loss, valid_loci = compute_pairwise_rank_loss(
        pred,
        target,
        pairwise_loci,
        mask_tensor,
        margin=margin,
        min_target_diff=min_target_diff,
        sample_mode=sample_mode,
        max_pairs_per_locus=max_pairs_per_locus,
    )
    metrics["valid_pairwise_loci"] = valid_loci
    metrics["pairwise"] = pair_loss.item() if pair_loss is not None else np.nan

    if loss_type == "pairwise_rank":
        if pair_loss is None:
            if allow_pairwise_fallback:
                total = _masked_regression_loss(pred, target, mask_tensor, "mse", huber_delta)
                metrics["monitor_name"] = "pairwise_rank_fallback_mse"
                metrics["total"] = total.item()
                return total, metrics
            raise ValueError("pairwise_rank requested but no valid loci with both positive and negative genes were found.")
        metrics["monitor_name"] = "pairwise_rank"
        metrics["total"] = pair_loss.item()
        return pair_loss, metrics

    if loss_type == "hybrid_mse_rank":
        if pair_loss is None:
            logging.warning("No valid pairwise loci for current split; hybrid loss falls back to regression term only.")
            pair_term = 0.0
        else:
            pair_term = rank_loss_weight * pair_loss
        reg_term = reg_loss_weight * _masked_regression_loss(pred, target, mask_tensor, "mse", huber_delta)
        total = reg_term + pair_term
        metrics["monitor_name"] = "hybrid_mse_rank"
        metrics["total"] = total.item()
        metrics["reg_component"] = reg_term.item()
        metrics["rank_component"] = float(pair_term.item()) if isinstance(pair_term, torch.Tensor) else float(pair_term)
        return total, metrics

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def evaluate_model(model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config):
    model.eval()
    with torch.no_grad():
        pred = model(batch)
        train_loss, train_metrics = compute_objective(
            pred,
            target_tensor,
            train_mask_tensor,
            batch["pairwise_groups"],
            **loss_config,
        )
        if val_mask_tensor.any():
            val_loss, val_metrics = compute_objective(
                pred,
                target_tensor,
                val_mask_tensor,
                batch["pairwise_groups"],
                allow_pairwise_fallback=True,
                **loss_config,
            )
        else:
            val_loss, val_metrics = train_loss, train_metrics
    return pred, train_loss.item(), val_loss.item(), train_metrics, val_metrics


def evaluate_fusion_model(model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config, fusion_mode):
    model.eval()
    with torch.no_grad():
        base_pred = model.predict_base(batch)
        branch_pred = model.predict_branch(batch)
        final_pred = base_pred + branch_pred

        baseline_train_metrics = baseline_val_metrics = None
        if val_mask_tensor.any():
            baseline_train_loss, baseline_train_metrics = compute_objective(
                base_pred,
                target_tensor,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
                allow_pairwise_fallback=True,
            )
            baseline_val_loss, baseline_val_metrics = compute_objective(
                base_pred,
                target_tensor,
                val_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
                allow_pairwise_fallback=True,
            )
        else:
            baseline_train_loss, baseline_train_metrics = compute_objective(
                base_pred,
                target_tensor,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
                allow_pairwise_fallback=True,
            )
            baseline_val_loss, baseline_val_metrics = baseline_train_loss, baseline_train_metrics

        if fusion_mode == "residual":
            residual_target = target_tensor - base_pred
            train_loss, train_metrics = compute_objective(
                branch_pred,
                residual_target,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
                allow_pairwise_fallback=True,
            )
            if val_mask_tensor.any():
                val_loss, val_metrics = compute_objective(
                    branch_pred,
                    residual_target,
                    val_mask_tensor,
                    batch["pairwise_groups"],
                    **loss_config,
                    allow_pairwise_fallback=True,
                )
            else:
                val_loss, val_metrics = train_loss, train_metrics
        else:
            train_loss, train_metrics = compute_objective(
                final_pred,
                target_tensor,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
                allow_pairwise_fallback=True,
            )
            if val_mask_tensor.any():
                val_loss, val_metrics = compute_objective(
                    final_pred,
                    target_tensor,
                    val_mask_tensor,
                    batch["pairwise_groups"],
                    **loss_config,
                    allow_pairwise_fallback=True,
                )
            else:
                val_loss, val_metrics = train_loss, train_metrics

    extras = {
        "baseline_train_loss": baseline_train_loss.item(),
        "baseline_val_loss": baseline_val_loss.item(),
        "baseline_train_mse": baseline_train_metrics["mse"],
        "baseline_val_mse": baseline_val_metrics["mse"],
        "final_val_mse": nn.functional.mse_loss(final_pred[val_mask_tensor], target_tensor[val_mask_tensor]).item()
        if val_mask_tensor.any()
        else nn.functional.mse_loss(final_pred[train_mask_tensor], target_tensor[train_mask_tensor]).item(),
        "final_train_mse": nn.functional.mse_loss(final_pred[train_mask_tensor], target_tensor[train_mask_tensor]).item(),
    }
    return final_pred, train_loss.item(), val_loss.item(), train_metrics, val_metrics, extras


def train_model(
    model,
    batch,
    targets,
    train_mask,
    val_mask,
    lr,
    weight_decay,
    epochs,
    loss_type,
    huber_delta=1.0,
    pairwise_margin=0.2,
    pairwise_min_target_diff=0.0,
    pairwise_sample_mode="balanced",
    pairwise_max_pairs_per_locus=256,
    reg_loss_weight=1.0,
    rank_loss_weight=0.2,
    warmup_epochs=20,
    patience=15,
    eval_every=5,
    fusion_mode="pure",
):
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    train_mask_tensor = torch.from_numpy(train_mask).to(batch["x_all"].device)
    val_mask_tensor = torch.from_numpy(val_mask).to(batch["x_all"].device)
    target_tensor = torch.from_numpy(targets.astype(np.float32)).to(batch["x_all"].device)
    loss_config = {
        "loss_type": loss_type,
        "huber_delta": huber_delta,
        "margin": pairwise_margin,
        "min_target_diff": pairwise_min_target_diff,
        "sample_mode": pairwise_sample_mode,
        "max_pairs_per_locus": pairwise_max_pairs_per_locus,
        "reg_loss_weight": reg_loss_weight,
        "rank_loss_weight": rank_loss_weight,
    }

    if hasattr(model, "fit_warm_start") and fusion_mode in {"add", "residual"}:
        model.fit_warm_start(batch, train_mask, targets[train_mask])

    if fusion_mode == "pure":
        baseline_pred, baseline_train_loss, baseline_val_loss, baseline_train_metrics, baseline_val_metrics = evaluate_model(
            model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config
        )
        logging.info(
            "Baseline train_loss=%.6f monitor_loss=%.6f train_mse=%.6f val_mse=%.6f",
            baseline_train_loss,
            baseline_val_loss,
            baseline_train_metrics["mse"],
            baseline_val_metrics["mse"],
        )
    else:
        (
            baseline_pred,
            baseline_train_loss,
            baseline_val_loss,
            baseline_train_metrics,
            baseline_val_metrics,
            baseline_extras,
        ) = evaluate_fusion_model(
            model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config, fusion_mode
        )
        logging.info(
            "Base-only train_loss=%.6f monitor_loss=%.6f train_mse=%.6f val_mse=%.6f",
            baseline_extras["baseline_train_loss"],
            baseline_extras["baseline_val_loss"],
            baseline_extras["baseline_train_mse"],
            baseline_extras["baseline_val_mse"],
        )
        logging.info(
            "Fusion(%s) baseline train_loss=%.6f monitor_loss=%.6f final_train_mse=%.6f final_val_mse=%.6f",
            fusion_mode,
            baseline_train_loss,
            baseline_val_loss,
            baseline_extras["final_train_mse"],
            baseline_extras["final_val_mse"],
        )

    early_stopping = EarlyStopping(patience=patience)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_loss = baseline_val_loss
    history = [
        {
            "epoch": -1,
            "train_loss": baseline_train_loss,
            "monitor_loss": baseline_val_loss,
            "train_mse": baseline_train_metrics["mse"],
            "val_mse": baseline_val_metrics["mse"],
            "monitor_name": baseline_val_metrics["monitor_name"],
        }
    ]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if fusion_mode == "residual":
            base_pred = model.predict_base(batch).detach()
            branch_pred = model.predict_branch(batch)
            residual_target = target_tensor - base_pred
            loss, _ = compute_objective(
                branch_pred,
                residual_target,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
            )
        else:
            pred = model(batch)
            loss, _ = compute_objective(
                pred,
                target_tensor,
                train_mask_tensor,
                batch["pairwise_groups"],
                **loss_config,
            )
        loss.backward()
        optimizer.step()

        should_eval = (epoch == 0) or ((epoch + 1) % eval_every == 0)
        if not should_eval:
            continue

        if fusion_mode == "pure":
            _, train_loss, monitor_loss, train_metrics, val_metrics = evaluate_model(
                model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config
            )
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "monitor_name": val_metrics["monitor_name"],
                "valid_pairwise_loci_train": train_metrics.get("valid_pairwise_loci", np.nan),
                "valid_pairwise_loci_val": val_metrics.get("valid_pairwise_loci", np.nan),
            }
        else:
            _, train_loss, monitor_loss, train_metrics, val_metrics, fusion_extras = evaluate_fusion_model(
                model, batch, target_tensor, train_mask_tensor, val_mask_tensor, loss_config, fusion_mode
            )
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "monitor_name": val_metrics["monitor_name"],
                "base_train_loss": fusion_extras["baseline_train_loss"],
                "base_val_loss": fusion_extras["baseline_val_loss"],
                "base_train_mse": fusion_extras["baseline_train_mse"],
                "base_val_mse": fusion_extras["baseline_val_mse"],
                "final_train_mse": fusion_extras["final_train_mse"],
                "final_val_mse": fusion_extras["final_val_mse"],
                "valid_pairwise_loci_train": train_metrics.get("valid_pairwise_loci", np.nan),
                "valid_pairwise_loci_val": val_metrics.get("valid_pairwise_loci", np.nan),
            }
        history.append(
            history_row
        )

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if fusion_mode == "pure":
            logging.info(
                "Epoch %d train_loss=%.6f monitor_loss=%.6f train_mse=%.6f val_mse=%.6f",
                epoch,
                train_loss,
                monitor_loss,
                train_metrics["mse"],
                val_metrics["mse"],
            )
        else:
            logging.info(
                "Epoch %d fusion=%s train_loss=%.6f monitor_loss=%.6f train_mse=%.6f val_mse=%.6f base_val_mse=%.6f final_val_mse=%.6f",
                epoch,
                fusion_mode,
                train_loss,
                monitor_loss,
                train_metrics["mse"],
                val_metrics["mse"],
                fusion_extras["baseline_val_mse"],
                fusion_extras["final_val_mse"],
            )

        if epoch + 1 > warmup_epochs:
            early_stopping.step(monitor_loss)
            if early_stopping.should_stop:
                logging.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return model, best_loss, best_epoch, history


def predict_scores(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu().numpy()
