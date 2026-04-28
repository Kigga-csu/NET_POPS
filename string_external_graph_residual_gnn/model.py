import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV


class MeanGraphConv(nn.Module):
    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return out / deg


class ResidualGraphBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, edge_dropout, residual_alpha):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = MeanGraphConv()
        self.conv2 = MeanGraphConv()
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.residual_alpha = residual_alpha
        # Start from a clean residual path so epoch 0 is close to the warm-started baseline.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _drop_edges(self, edge_index):
        if (not self.training) or self.edge_dropout <= 0.0:
            return edge_index
        keep_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.edge_dropout
        self_loop_mask = edge_index[0] == edge_index[1]
        keep_mask = keep_mask | self_loop_mask
        return edge_index[:, keep_mask]

    def forward(self, x, edge_index, graph_mask):
        edge_index = self._drop_edges(edge_index)
        hidden = F.relu(self.in_proj(x))
        hidden = hidden + self.conv1(hidden, edge_index)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = hidden + self.conv2(hidden, edge_index)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        residual = self.out_proj(hidden).squeeze(-1)
        return self.residual_alpha * residual * graph_mask


class ExternalGraphResidualModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=32,
        dropout=0.2,
        edge_dropout=0.0,
        residual_alpha=0.1,
        freeze_base=True,
    ):
        super().__init__()
        self.base = nn.Linear(in_dim, 1, bias=False)
        self.graph_branch = ResidualGraphBranch(in_dim, hidden_dim, dropout, edge_dropout, residual_alpha)
        self.freeze_base = freeze_base

    def fit_warm_start(self, batch, train_mask, y_train):
        x_all = batch["x_all"].detach().cpu().numpy().astype(np.float64)
        ridge = RidgeCV(alphas=np.logspace(-2, 10, num=25), fit_intercept=False)
        ridge.fit(x_all[train_mask], y_train.astype(np.float64))
        coef = ridge.coef_.astype(np.float32).reshape(1, -1)
        with torch.no_grad():
            self.base.weight.copy_(torch.from_numpy(coef).to(self.base.weight.device))
        if self.freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, batch):
        x_all = batch["x_all"]
        edge_index = batch["edge_index"]
        graph_mask = batch["graph_mask"]
        base_score = self.base(x_all).squeeze(-1)
        residual_score = self.graph_branch(x_all, edge_index, graph_mask)
        return base_score + residual_score

    def predict_base(self, batch):
        x_all = batch["x_all"]
        return self.base(x_all).squeeze(-1)

    def predict_residual(self, batch):
        x_all = batch["x_all"]
        edge_index = batch["edge_index"]
        graph_mask = batch["graph_mask"]
        return self.graph_branch(x_all, edge_index, graph_mask)
