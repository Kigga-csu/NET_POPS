import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV


def drop_edges(edge_index, edge_dropout, training):
    if (not training) or edge_dropout <= 0.0:
        return edge_index
    keep_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > edge_dropout
    self_loop_mask = edge_index[0] == edge_index[1]
    keep_mask = keep_mask | self_loop_mask
    return edge_index[:, keep_mask]


def aggregate_mean(x, edge_index):
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    deg = deg.clamp(min=1.0).unsqueeze(-1)
    return out / deg


def aggregate_gcn(x, edge_index):
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    ones = torch.ones_like(src, dtype=x.dtype)
    deg.index_add_(0, src, ones)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1.0)
    norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] * norm)
    return out


def segment_softmax(scores, dst, num_nodes):
    max_per_dst = torch.full((num_nodes,), -float("inf"), device=scores.device, dtype=scores.dtype)
    max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    stabilized = scores - max_per_dst[dst]
    exp_scores = torch.exp(stabilized)
    denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, dst, exp_scores)
    return exp_scores / denom[dst].clamp(min=1e-12)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, num_layers=2):
        super().__init__()
        layers = []
        current_dim = in_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x):
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, num_layers=2):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dim, dropout, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        h = self.encoder(batch["x_all"])
        return self.out_proj(h).squeeze(-1)


class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h, edge_index):
        neigh = aggregate_mean(h, edge_index)
        return self.lin(torch.cat([h, neigh], dim=-1))


class GCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, edge_index):
        return self.lin(aggregate_gcn(h, edge_index))


class GATLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.uniform_(self.attn_src, -0.1, 0.1)
        nn.init.uniform_(self.attn_dst, -0.1, 0.1)

    def forward(self, h, edge_index):
        z = self.lin(h)
        src, dst = edge_index[0], edge_index[1]
        logits = (z[src] * self.attn_src).sum(-1) + (z[dst] * self.attn_dst).sum(-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)
        alpha = segment_softmax(logits, dst, z.size(0)).unsqueeze(-1)
        out = torch.zeros_like(z)
        out.index_add_(0, dst, z[src] * alpha)
        return out


class APPNPPropagator(nn.Module):
    def __init__(self, num_steps=10, alpha=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.alpha = alpha

    def forward(self, h, edge_index, edge_dropout, training):
        h0 = h
        current = h
        for _ in range(self.num_steps):
            current_edges = drop_edges(edge_index, edge_dropout, training)
            propagated = aggregate_gcn(current, current_edges)
            current = (1.0 - self.alpha) * propagated + self.alpha * h0
        return current


class GraphRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        dropout,
        edge_dropout,
        num_layers,
        model_type,
        appnp_steps=10,
        appnp_alpha=0.1,
    ):
        super().__init__()
        self.model_type = model_type
        self.edge_dropout = edge_dropout
        self.dropout = dropout
        self.encoder = MLPEncoder(in_dim, hidden_dim, dropout, num_layers=1)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.num_layers = num_layers

        if model_type == "graphsage":
            self.layers = nn.ModuleList([GraphSAGELayer(hidden_dim) for _ in range(num_layers)])
        elif model_type == "gcn":
            self.layers = nn.ModuleList([GCNLayer(hidden_dim) for _ in range(num_layers)])
        elif model_type == "gat":
            self.layers = nn.ModuleList([GATLayer(hidden_dim) for _ in range(num_layers)])
        elif model_type == "appnp":
            self.layers = nn.ModuleList([])
            self.propagator = APPNPPropagator(num_steps=appnp_steps, alpha=appnp_alpha)
        else:
            raise ValueError(f"Unsupported graph model_type: {model_type}")

    def forward(self, batch):
        x_all = batch["x_all"]
        edge_index = batch["edge_index"]
        h = F.relu(self.input_proj(self.encoder(x_all)))
        if self.model_type == "appnp":
            h = self.propagator(h, edge_index, self.edge_dropout, self.training)
            h = F.dropout(h, p=self.dropout, training=self.training)
            return self.out_proj(h).squeeze(-1)

        current_edges = drop_edges(edge_index, self.edge_dropout, self.training)
        for layer in self.layers:
            h = layer(h, current_edges)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_proj(h).squeeze(-1)

class FusionRegressor(nn.Module):
    def __init__(self, branch_model, in_dim, freeze_base=True):
        super().__init__()
        self.branch_model = branch_model
        self.base = nn.Linear(in_dim, 1, bias=False)
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

    def predict_base(self, batch):
        return self.base(batch["x_all"]).squeeze(-1)

    def predict_branch(self, batch):
        return self.branch_model(batch)

    def forward(self, batch):
        return self.predict_base(batch) + self.predict_branch(batch)


def build_core_model(
    model_type,
    in_dim,
    hidden_dim,
    dropout,
    edge_dropout,
    num_layers=2,
    appnp_steps=10,
    appnp_alpha=0.1,
):
    if model_type == "mlp":
        return MLPRegressor(in_dim, hidden_dim, dropout, num_layers=max(2, num_layers))
    return GraphRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        edge_dropout=edge_dropout,
        num_layers=num_layers,
        model_type=model_type,
        appnp_steps=appnp_steps,
        appnp_alpha=appnp_alpha,
    )


def build_model(
    model_type,
    in_dim,
    hidden_dim,
    dropout,
    edge_dropout,
    num_layers=2,
    appnp_steps=10,
    appnp_alpha=0.1,
    fusion_mode="pure",
    freeze_base=True,
):
    core_model = build_core_model(
        model_type=model_type,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        edge_dropout=edge_dropout,
        num_layers=num_layers,
        appnp_steps=appnp_steps,
        appnp_alpha=appnp_alpha,
    )
    if fusion_mode == "pure":
        return core_model
    if fusion_mode in {"add", "residual"}:
        return FusionRegressor(core_model, in_dim=in_dim, freeze_base=freeze_base)
    raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
