#!/usr/bin/env python
"""Rigorous GNN baselines for TCD-JEPA comparison.

Implements properly tuned GNN baselines with:
- GCN (3-layer, residual, layer norm)
- GAT (3-layer, multi-head additive attention)
- GraphSAGE (3-layer, mean aggregation)
- R-GCN (relation-aware, for heterogeneous edges)
- All with early stopping, multiple seeds, gradient clipping,
  cosine LR schedule, and documented hyperparameters.

Usage:
    python scripts/gnn_baselines.py --data-dir ./data/semiconductor
    python scripts/gnn_baselines.py --data-dir ./data/gdelt --seeds 5
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("gnn_baselines")


# ---------------------------------------------------------------------------
# GNN Layers
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class GATLayer(nn.Module):
    """Multi-head GAT with additive attention (memory-efficient)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, concat: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads if concat else out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        Wh = self.W(x).view(N, self.num_heads, self.head_dim)
        e_src = (Wh * self.attn_src.unsqueeze(0)).sum(-1)
        e_dst = (Wh * self.attn_dst.unsqueeze(0)).sum(-1)
        e = F.leaky_relu(e_src.unsqueeze(1) + e_dst.unsqueeze(0), 0.2)
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=1).masked_fill(mask, 0.0)
        out = torch.einsum("nmh,nhd->nhd", alpha, Wh)
        return out.reshape(N, -1) if self.concat else out.mean(dim=1)


class SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        return self.linear(torch.cat([x, (adj @ x) / degree], dim=-1))


class RGCNLayer(nn.Module):
    """Relational GCN — separate weights per edge weight bin."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 4) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(num_relations)
        ])
        self.self_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        for w in list(self.weights) + [self.self_weight]:
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = x @ self.self_weight
        edges_exist = adj > 0
        for r in range(self.num_relations):
            lo = r / self.num_relations
            hi = (r + 1) / self.num_relations + (1e-6 if r == self.num_relations - 1 else 0)
            mask = edges_exist & (adj >= lo) & (adj < hi)
            if mask.any():
                adj_r = mask.float()
                deg = adj_r.sum(-1, keepdim=True).clamp(min=1)
                out = out + ((adj_r @ x) / deg) @ self.weights[r]
        return out


# ---------------------------------------------------------------------------
# 3-layer GNN with residuals
# ---------------------------------------------------------------------------

class GNNModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 gnn_type: str = "gcn", dropout: float = 0.3) -> None:
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        Layer = {"gcn": GCNLayer, "gat": lambda i, o: GATLayer(i, o, 4),
                 "sage": SAGELayer, "rgcn": RGCNLayer}[gnn_type]
        self.l1 = Layer(in_dim, hidden)
        self.l2 = Layer(hidden, hidden)
        self.l3 = (GATLayer(hidden, out_dim, 1, concat=False) if gnn_type == "gat"
                   else Layer(hidden, out_dim) if gnn_type != "rgcn"
                   else RGCNLayer(hidden, out_dim))
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.res = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.l1(x, adj)))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.n2(self.l2(h, adj))) + h  # residual
        h = F.dropout(h, self.dropout, self.training)
        return self.l3(h, adj)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([zi, zj], -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    d = adj.sum(1)
    d_inv = torch.where(d > 0, 1.0 / d.sqrt(), torch.zeros_like(d))
    D = torch.diag(d_inv)
    return D @ adj @ D


def evaluate_link_prediction(emb: torch.Tensor, adj: torch.Tensor) -> dict:
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    pos = sim[adj > 0]
    neg_mask = (adj == 0) & ~torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)
    neg = sim[neg_mask]
    if len(pos) == 0 or len(neg) == 0:
        return {"link_auc": 0.5, "link_sim_gap": 0.0, "link_pos_sim": 0.0, "link_neg_sim": 0.0}
    n = min(10000, len(pos) * len(neg))
    auc = (pos[torch.randint(len(pos), (n,))] > neg[torch.randint(len(neg), (n,))]).float().mean().item()
    return {"link_auc": auc, "link_pos_sim": pos.mean().item(),
            "link_neg_sim": neg.mean().item(), "link_sim_gap": pos.mean().item() - neg.mean().item()}


def evaluate_knn(emb: torch.Tensor, labels: torch.Tensor) -> dict:
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    sim.fill_diagonal_(-1)
    results = {}
    for k in [1, 5, 20]:
        topk = sim.topk(k, dim=-1).indices
        tl = labels[topk]
        preds = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
        for i in range(len(labels)):
            v, c = tl[i].unique(return_counts=True)
            preds[i] = v[c.argmax()]
        results[f"knn_k{k}"] = (preds == labels).float().mean().item()
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnn(features, adj, labels, gnn_type, hidden=256, out_dim=128,
              epochs=500, lr=0.005, patience=50, seed=42, device="cuda"):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    features, adj, labels = features.to(dev), adj.to(dev), labels.to(dev)

    adj_in = normalize_adjacency(adj) if gnn_type == "gcn" else adj + torch.eye(adj.shape[0], device=dev)
    model = GNNModel(features.shape[1], hidden, out_dim, gnn_type).to(dev)
    lp = LinkPredictor(out_dim).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(lp.parameters()), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    pos_e = torch.where(adj > 0)
    n_pos = len(pos_e[0])
    neg_e = torch.where((adj == 0) & ~torch.eye(adj.shape[0], dtype=torch.bool, device=dev))

    best, wait = float("inf"), 0
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        emb = model(features, adj_in)
        ns = min(n_pos, 5000)
        pi = torch.randperm(n_pos, device=dev)[:ns]
        ni = torch.randperm(len(neg_e[0]), device=dev)[:ns]
        loss = (F.binary_cross_entropy_with_logits(lp(emb[pos_e[0][pi]], emb[pos_e[1][pi]]), torch.ones(ns, device=dev))
                + F.binary_cross_entropy_with_logits(lp(emb[neg_e[0][ni]], emb[neg_e[1][ni]]), torch.zeros(ns, device=dev)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best - 1e-4:
            best, wait = loss.item(), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    with torch.no_grad():
        emb = model(features, adj_in)
        return {**evaluate_link_prediction(emb, adj), **evaluate_knn(emb, labels),
                "gnn_type": gnn_type, "epochs_trained": ep + 1}


def main():
    parser = argparse.ArgumentParser(description="Rigorous GNN baselines")
    parser.add_argument("--data-dir", default="./data/semiconductor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    features = torch.load(data_dir / "fingerprints.pt", weights_only=True)
    adjacency = torch.load(data_dir / "adjacency.pt", weights_only=True)
    labels = torch.load(data_dir / "labels.pt", weights_only=True)

    logger.info(f"Data: {features.shape[0]} entities, {int((adjacency > 0).sum())} edges, "
                f"{labels.max().item() + 1} classes")
    logger.info(f"Config: hidden={args.hidden}, out={args.out_dim}, epochs={args.epochs}, "
                f"seeds={args.seeds}, lr=0.005, wd=5e-4, dropout=0.3, patience=50")

    all_results = {}
    for gnn_type in ["gcn", "gat", "sage", "rgcn"]:
        logger.info(f"\n{'=' * 50}\n  {gnn_type.upper()} (3-layer, {args.seeds} seeds)\n{'=' * 50}")
        seed_results = []
        for s in range(args.seeds):
            try:
                r = train_gnn(features, adjacency, labels, gnn_type, args.hidden, args.out_dim,
                              args.epochs, seed=42 + s, device=args.device)
                seed_results.append(r)
                logger.info(f"  Seed {s}: auc={r['link_auc']:.4f} knn1={r.get('knn_k1', 0):.4f}")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  OOM — skipping {gnn_type.upper()}")
                break
            except Exception as e:
                logger.warning(f"  Failed: {e}")

        if seed_results:
            avg = {}
            for key in seed_results[0]:
                if isinstance(seed_results[0][key], (int, float)):
                    vals = [r[key] for r in seed_results]
                    avg[key] = sum(vals) / len(vals)
                    avg[f"{key}_std"] = (sum((v - avg[key])**2 for v in vals) / max(len(vals) - 1, 1))**0.5
            avg["gnn_type"] = gnn_type
            avg["num_seeds"] = len(seed_results)
            all_results[gnn_type] = avg
            logger.info(f"  Mean: auc={avg['link_auc']:.4f}±{avg.get('link_auc_std', 0):.4f}")

    logger.info(f"\n{'=' * 80}\nCOMPARISON (mean±std, {args.seeds} seeds)\n{'=' * 80}")
    logger.info(f"{'Method':8s} {'N':>4s} {'Link AUC':>14s} {'Sim Gap':>14s} {'kNN-1':>14s}")
    logger.info("-" * 60)
    for name, r in all_results.items():
        logger.info(f"{name.upper():8s} {r.get('num_seeds', 0):4d} "
                     f"{r['link_auc']:.4f}±{r.get('link_auc_std', 0):.3f}  "
                     f"{r['link_sim_gap']:.4f}±{r.get('link_sim_gap_std', 0):.3f}  "
                     f"{r.get('knn_k1', 0):.4f}±{r.get('knn_k1_std', 0):.3f}")


if __name__ == "__main__":
    main()
