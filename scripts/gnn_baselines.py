#!/usr/bin/env python
"""GNN baselines for semiconductor supply chain link prediction.

Runs GCN, GAT, and GraphSAGE on the same link prediction task as TCD-JEPA
for direct comparison. Uses simple 2-layer GNN implementations to avoid
PyG dependency.

Usage:
    python scripts/gnn_baselines.py --data-dir ./data/semiconductor
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
# Simple GNN layers (no PyG dependency)
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Graph Convolutional Network layer: H' = sigma(D^{-1/2} A D^{-1/2} H W)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(adj_norm @ x))


class GATLayer(nn.Module):
    """Graph Attention Network layer with single head."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(x)  # [N, out_dim]
        N = Wh.shape[0]
        # Pairwise attention
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)
        e = self.leaky_relu(self.attn(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        # Mask non-edges
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask, 0.0)
        return F.elu(alpha @ Wh)


class SAGELayer(nn.Module):
    """GraphSAGE mean-aggregation layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Mean aggregation of neighbors
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        neighbor_agg = (adj @ x) / degree
        combined = torch.cat([x, neighbor_agg], dim=-1)
        return F.relu(self.linear(combined))


class GNNModel(nn.Module):
    """2-layer GNN for node embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, gnn_type: str = "gcn") -> None:
        super().__init__()
        self.gnn_type = gnn_type
        if gnn_type == "gcn":
            self.layer1 = GCNLayer(in_dim, hidden_dim)
            self.layer2 = GCNLayer(hidden_dim, out_dim)
        elif gnn_type == "gat":
            self.layer1 = GATLayer(in_dim, hidden_dim)
            self.layer2 = GATLayer(hidden_dim, out_dim)
        elif gnn_type == "sage":
            self.layer1 = SAGELayer(in_dim, hidden_dim)
            self.layer2 = SAGELayer(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.layer2(h, adj)
        return h


class LinkPredictor(nn.Module):
    """MLP link predictor: takes pair of node embeddings, predicts link."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_i, z_j], dim=-1)).squeeze(-1)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    adj = adj + torch.eye(adj.shape[0], device=adj.device)  # self-loops
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))
    D = torch.diag(d_inv_sqrt)
    return D @ adj @ D


def evaluate_link_prediction(embeddings: torch.Tensor, adj: torch.Tensor) -> dict:
    """Evaluate link prediction using cosine similarity."""
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.t()

    pos_sims = sim[adj > 0]
    neg_mask = (adj == 0) & ~torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)
    neg_sims = sim[neg_mask]

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        return {"link_auc": 0.5, "link_sim_gap": 0.0}

    pos_mean = pos_sims.mean().item()
    neg_mean = neg_sims.mean().item()

    # Approximate AUC
    threshold_range = torch.linspace(0, 1, 100, device=sim.device)
    best_f1 = 0.0
    for t in threshold_range:
        tp = (pos_sims > t).float().sum()
        fp = (neg_sims > t).float().sum()
        fn = (pos_sims <= t).float().sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1 = max(best_f1, f1.item())

    # Simple AUC approximation: fraction of pos > neg pairs
    n_samples = min(10000, len(pos_sims) * len(neg_sims))
    pos_sample = pos_sims[torch.randint(len(pos_sims), (n_samples,))]
    neg_sample = neg_sims[torch.randint(len(neg_sims), (n_samples,))]
    auc = (pos_sample > neg_sample).float().mean().item()

    return {
        "link_auc": auc,
        "link_pos_sim": pos_mean,
        "link_neg_sim": neg_mean,
        "link_sim_gap": pos_mean - neg_mean,
    }


def evaluate_node_classification(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    """k-NN classification accuracy."""
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.t()
    sim.fill_diagonal_(-1)

    results = {}
    for k in [1, 5, 20]:
        topk_idx = sim.topk(k, dim=-1).indices
        topk_labels = labels[topk_idx]
        # Majority vote
        preds = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
        for i in range(len(labels)):
            vals, counts = topk_labels[i].unique(return_counts=True)
            preds[i] = vals[counts.argmax()]
        acc = (preds == labels).float().mean().item()
        results[f"knn_k{k}"] = acc

    return results


def train_gnn_link_prediction(
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    gnn_type: str,
    hidden_dim: int = 128,
    out_dim: int = 64,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cuda",
) -> dict:
    """Train a GNN and evaluate on link prediction + node classification."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    adj_norm = normalize_adjacency(adj)
    adj_input = adj_norm if gnn_type == "gcn" else (adj + torch.eye(adj.shape[0], device=device))

    in_dim = features.shape[1]
    model = GNNModel(in_dim, hidden_dim, out_dim, gnn_type).to(device)
    link_pred = LinkPredictor(out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_pred.parameters()), lr=lr, weight_decay=5e-4,
    )

    # Prepare training edges
    pos_edges = torch.where(adj > 0)
    num_pos = len(pos_edges[0])
    neg_mask = (adj == 0) & ~torch.eye(adj.shape[0], dtype=torch.bool, device=device)
    neg_edges = torch.where(neg_mask)

    logger.info(f"  Training {gnn_type.upper()}: {in_dim}→{hidden_dim}→{out_dim}, "
                f"{num_pos} pos edges, {epochs} epochs")

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        embeddings = model(features, adj_input)

        # Sample balanced positive/negative edges
        n_sample = min(num_pos, 5000)
        pos_idx = torch.randperm(num_pos, device=device)[:n_sample]
        neg_idx = torch.randperm(len(neg_edges[0]), device=device)[:n_sample]

        pos_scores = link_pred(embeddings[pos_edges[0][pos_idx]], embeddings[pos_edges[1][pos_idx]])
        neg_scores = link_pred(embeddings[neg_edges[0][neg_idx]], embeddings[neg_edges[1][neg_idx]])

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adj_input)
        link_results = evaluate_link_prediction(embeddings, adj)
        knn_results = evaluate_node_classification(embeddings, labels)

    return {**link_results, **knn_results, "gnn_type": gnn_type}


def main():
    parser = argparse.ArgumentParser(description="GNN baselines for supply chain link prediction")
    parser.add_argument("--data-dir", default="./data/semiconductor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    fingerprints = torch.load(data_dir / "fingerprints.pt", weights_only=True)
    adjacency = torch.load(data_dir / "adjacency.pt", weights_only=True)
    labels = torch.load(data_dir / "labels.pt", weights_only=True)

    logger.info(f"Data: {fingerprints.shape[0]} entities, {int((adjacency > 0).sum())} edges, "
                f"{labels.max().item() + 1} classes")

    # Run all GNN baselines
    all_results = {}
    for gnn_type in ["gcn", "gat", "sage"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"  {gnn_type.upper()}")
        logger.info(f"{'='*50}")

        results = train_gnn_link_prediction(
            fingerprints, adjacency, labels,
            gnn_type=gnn_type,
            epochs=args.epochs,
            device=args.device,
        )
        all_results[gnn_type] = results

        for k, v in sorted(results.items()):
            if isinstance(v, float):
                logger.info(f"  {k:25s}: {v:.4f}")

    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':20s} {'Link AUC':>10s} {'Sim Gap':>10s} {'kNN-1':>8s} {'kNN-5':>8s}")
    logger.info("-" * 60)
    for name, r in all_results.items():
        logger.info(f"{name.upper():20s} {r['link_auc']:10.4f} {r['link_sim_gap']:10.4f} "
                     f"{r.get('knn_k1', 0):8.4f} {r.get('knn_k5', 0):8.4f}")

    # Print TCD-JEPA results for reference
    logger.info(f"\n{'Reference (from earlier runs)':20s}")
    logger.info(f"{'JEPA Baseline':20s} {'0.7370':>10s} {'0.4763':>10s} {'0.9901':>8s} {'0.9781':>8s}")
    logger.info(f"{'TCD-JEPA':20s} {'0.8270':>10s} {'0.4185':>10s} {'0.9779':>8s} {'0.9604':>8s}")


if __name__ == "__main__":
    main()
