import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
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


def evaluate_losses(model, batch, target_tensor, train_mask_tensor, val_mask_tensor):
    model.eval()
    with torch.no_grad():
        pred = model(batch)
        criterion = nn.MSELoss()
        train_loss = criterion(pred[train_mask_tensor], target_tensor[train_mask_tensor]).item()
        if val_mask_tensor.any():
            monitor_loss = criterion(pred[val_mask_tensor], target_tensor[val_mask_tensor]).item()
        else:
            monitor_loss = train_loss
    return train_loss, monitor_loss


def evaluate_residual_losses(model, batch, residual_target_tensor, train_mask_tensor, val_mask_tensor):
    model.eval()
    with torch.no_grad():
        pred_residual = model.predict_residual(batch)
        criterion = nn.MSELoss()
        train_loss = criterion(pred_residual[train_mask_tensor], residual_target_tensor[train_mask_tensor]).item()
        if val_mask_tensor.any():
            monitor_loss = criterion(
                pred_residual[val_mask_tensor], residual_target_tensor[val_mask_tensor]
            ).item()
        else:
            monitor_loss = train_loss
    return train_loss, monitor_loss


def train_model(
    model,
    batch,
    targets,
    train_mask,
    val_mask,
    lr,
    weight_decay,
    epochs,
    warmup_epochs=25,
    patience=60,
    eval_every=5,
):
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_mask_tensor = torch.from_numpy(train_mask).to(batch["x_all"].device)
    val_mask_tensor = torch.from_numpy(val_mask).to(batch["x_all"].device)
    target_tensor = torch.from_numpy(targets.astype(np.float32)).to(batch["x_all"].device)

    if hasattr(model, "fit_warm_start"):
        model.fit_warm_start(batch, train_mask, targets[train_mask])

    with torch.no_grad():
        base_score = model.predict_base(batch)

    baseline_train_loss, baseline_monitor_loss = evaluate_losses(
        model, batch, target_tensor, train_mask_tensor, val_mask_tensor
    )
    logging.info(
        "Baseline-only train_loss=%.6f monitor_loss=%.6f",
        baseline_train_loss,
        baseline_monitor_loss,
    )

    residual_target_tensor = target_tensor - base_score
    baseline_residual_train_loss, baseline_residual_monitor_loss = evaluate_residual_losses(
        model, batch, residual_target_tensor, train_mask_tensor, val_mask_tensor
    )
    logging.info(
        "Residual-target baseline train_loss=%.6f monitor_loss=%.6f",
        baseline_residual_train_loss,
        baseline_residual_monitor_loss,
    )

    early_stopping = EarlyStopping(patience=patience)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = baseline_monitor_loss
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_residual = model.predict_residual(batch)
        train_loss = criterion(pred_residual[train_mask_tensor], residual_target_tensor[train_mask_tensor])
        train_loss.backward()
        optimizer.step()

        should_eval = (epoch == 0) or ((epoch + 1) % eval_every == 0)
        if not should_eval:
            continue

        eval_train_loss, monitor_loss = evaluate_residual_losses(
            model, batch, residual_target_tensor, train_mask_tensor, val_mask_tensor
        )

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = copy.deepcopy(model.state_dict())

        logging.info(
            "Epoch %d residual_train_loss=%.6f residual_monitor_loss=%.6f",
            epoch,
            eval_train_loss,
            monitor_loss,
        )

        if epoch + 1 > warmup_epochs:
            early_stopping.step(monitor_loss)
            if early_stopping.should_stop:
                logging.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return model, best_loss


def predict_scores(model, batch):
    model.eval()
    with torch.no_grad():
        scores = model(batch).detach().cpu().numpy()
    return scores
