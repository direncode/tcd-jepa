"""Evaluation protocols for Latent Ocean TCD-JEPA.

Standard SSL metrics:
1. Linear probe — frozen encoder → train linear classifier on entity types
2. k-NN — nearest neighbors in embedding space should share entity type
3. Causal link prediction — embedding similarity should predict causal links
4. Manifold neighborhood preservation — nearby on S² ↔ nearby in embedding

Latent Ocean KPIs (mapped from TCD topology):
5. Clarity Score ← H0 persistence (cluster separation quality)
6. Drift Velocity ← prediction error (active vs. converged dynamics)
7. Opportunity Surface ← H1 cycles (cross-cluster causal loops)
8. Risk Horizon ← H2 boundaries (entities at cluster interfaces)
"""


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_manifold_features(
    model: nn.Module,
    dataset,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract frozen encoder features from manifold data.

    Returns:
        features: [total_tokens, embed_dim]
        labels: [total_tokens]
        indices: [total_tokens]
    """
    model.eval()
    all_features, all_labels, all_indices = [], [], []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            fp = sample["fingerprints"].unsqueeze(0).to(device)
            coords = sample.get("coords")
            if coords is not None:
                coords = coords.unsqueeze(0).to(device)
            adj = sample.get("adjacency")
            if adj is not None:
                adj = adj.unsqueeze(0).to(device)
            vel = sample.get("velocity")
            if vel is not None:
                vel = vel.unsqueeze(0).to(device)

            encoder = model.context_encoder if hasattr(model, "context_encoder") else model
            z = encoder(fp, coords=coords, adjacency=adj, velocity=vel)
            z = z.squeeze(0)  # [N, D]

            all_features.append(z.cpu())
            if "labels" in sample:
                all_labels.append(sample["labels"])
            all_indices.append(sample["indices"])

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(features.shape[0], dtype=torch.long)
    indices = torch.cat(all_indices, dim=0)

    return features, labels, indices


def linear_probe(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int,
    train_ratio: float = 0.8, num_epochs: int = 100, lr: float = 0.01,
) -> dict[str, float]:
    """Standard linear probe on frozen features."""
    N = features.shape[0]
    perm = torch.randperm(N)
    split = int(N * train_ratio)
    X_train, y_train = features[perm[:split]], labels[perm[:split]]
    X_test, y_test = features[perm[split:]], labels[perm[split:]]

    mu, std = X_train.mean(0), X_train.std(0).clamp(min=1e-6)
    X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

    clf = nn.Linear(features.shape[1], num_classes)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs)

    clf.train()
    for _ in range(num_epochs):
        loss = F.cross_entropy(clf(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        train_acc = (clf(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (clf(X_test).argmax(1) == y_test).float().mean().item()

    return {"train_acc": train_acc, "test_acc": test_acc}


def knn_evaluate(
    features: torch.Tensor, labels: torch.Tensor,
    k_values: list[int] = [1, 5, 20], train_ratio: float = 0.8,
) -> dict[str, float]:
    """k-NN evaluation on frozen features."""
    N = features.shape[0]
    perm = torch.randperm(N)
    split = int(N * train_ratio)
    X_train, y_train = features[perm[:split]], labels[perm[:split]]
    X_test, y_test = features[perm[split:]], labels[perm[split:]]

    mu, std = X_train.mean(0), X_train.std(0).clamp(min=1e-6)
    X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

    dists = torch.cdist(X_test, X_train)
    results = {}
    for k in k_values:
        _, nn_idx = dists.topk(k, largest=False)
        pred = torch.mode(y_train[nn_idx], dim=1).values
        results[f"knn_k{k}"] = (pred == y_test).float().mean().item()
    return results


def causal_link_prediction(
    features: torch.Tensor, adjacency: torch.Tensor,
    indices: torch.Tensor, num_entities: int,
) -> dict[str, float]:
    """Evaluate causal link prediction from learned representations."""
    entity_features = torch.zeros(num_entities, features.shape[1])
    entity_counts = torch.zeros(num_entities)

    for i in range(len(indices)):
        eid = indices[i].item()
        if eid < num_entities:
            entity_features[eid] += features[i]
            entity_counts[eid] += 1

    mask = entity_counts > 0
    entity_features[mask] /= entity_counts[mask].unsqueeze(1)

    valid = torch.where(mask)[0]
    if len(valid) < 10:
        return {"link_auc": 0.0, "link_sim_gap": 0.0}

    valid_feat = F.normalize(entity_features[valid], dim=1)
    sim = valid_feat @ valid_feat.t()
    valid_adj = adjacency[valid][:, valid]

    pos_sims = sim[valid_adj > 0]
    neg_sims = sim[valid_adj == 0]

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        return {"link_auc": 0.5, "link_sim_gap": 0.0}

    n = min(1000, len(pos_sims) * len(neg_sims))
    pos_s = pos_sims[torch.randint(len(pos_sims), (n,))]
    neg_s = neg_sims[torch.randint(len(neg_sims), (n,))]

    return {
        "link_auc": (pos_s > neg_s).float().mean().item(),
        "link_pos_sim": pos_sims.mean().item(),
        "link_neg_sim": neg_sims.mean().item(),
        "link_sim_gap": pos_sims.mean().item() - neg_sims.mean().item(),
    }


def manifold_neighborhood_preservation(
    features: torch.Tensor, coords: torch.Tensor, k: int = 10,
) -> dict[str, float]:
    """Measure embedding vs. manifold neighborhood overlap."""
    feat_norm = F.normalize(features, dim=1)
    coord_norm = F.normalize(coords, dim=1)

    embed_dists = torch.cdist(feat_norm, feat_norm)
    _, embed_nn = embed_dists.topk(k + 1, largest=False)
    embed_nn = embed_nn[:, 1:]

    coord_dists = torch.cdist(coord_norm, coord_norm)
    _, coord_nn = coord_dists.topk(k + 1, largest=False)
    coord_nn = coord_nn[:, 1:]

    overlaps = []
    for i in range(len(features)):
        e_set = set(embed_nn[i].tolist())
        c_set = set(coord_nn[i].tolist())
        overlaps.append(len(e_set & c_set) / len(e_set | c_set))

    t = torch.tensor(overlaps)
    return {"neighborhood_preservation": t.mean().item(), "preservation_std": t.std().item()}


def compute_representation_metrics(features: torch.Tensor) -> dict[str, float]:
    """Effective rank, feature diversity, uniformity."""
    centered = features - features.mean(0)
    metrics = {}

    try:
        S = torch.linalg.svdvals(centered)
        S_norm = S / (S.sum() + 1e-8)
        entropy = -(S_norm * (S_norm + 1e-8).log()).sum()
        metrics["effective_rank"] = entropy.exp().item()
    except Exception:
        metrics["effective_rank"] = 0.0

    metrics["feature_std"] = features.std(0).mean().item()

    feat_norm = F.normalize(features, dim=1)
    n = min(features.shape[0], 500)
    sq = torch.cdist(feat_norm[:n], feat_norm[:n]).pow(2)
    metrics["uniformity"] = (-2.0 * sq).exp().mean().log().item()

    return metrics


# ============================================================================
# Latent Ocean KPI Computation
# ============================================================================

def compute_latent_ocean_kpis(
    model: nn.Module,
    dataset,
    device: torch.device,
    recursive_loop=None,
    max_samples: int = 200,
) -> dict[str, float]:
    """Compute Latent Ocean's 4 strategic KPIs from TCD-JEPA outputs.

    Mapping:
    - Clarity Score (0-100) ← intra/inter cluster distance ratio in learned space
    - Drift Velocity ← mean prediction error (high = active dynamics)
    - Opportunity Surface ← count of high-similarity cross-type connections
    - Risk Horizon ← count of entities near cluster boundaries in embedding space

    Args:
        model: Trained ManifoldJEPAModel.
        dataset: Dataset with entity_labels, adjacency, etc.
        device: Compute device.
        recursive_loop: Optional TCD recursive loop for topological features.
        max_samples: Number of windows to process.

    Returns:
        Dict with KPI values.
    """
    features, labels, indices = extract_manifold_features(model, dataset, device, max_samples)

    kpis = {}

    # ── Clarity Score (0-100) ──
    # Ratio of intra-cluster to inter-cluster distance in learned embedding space
    # Higher = better-separated entity types
    num_classes = int(labels.max().item()) + 1
    intra_dists, inter_dists = [], []

    feat_norm = F.normalize(features, dim=1)
    for c in range(num_classes):
        c_mask = labels == c
        if c_mask.sum() < 2:
            continue
        c_feats = feat_norm[c_mask]
        # Intra: pairwise distances within cluster
        intra = torch.cdist(c_feats, c_feats)
        intra_dists.append(intra[intra > 0].mean().item() if (intra > 0).any() else 0.0)

        # Inter: distances to other clusters
        other_feats = feat_norm[~c_mask]
        if len(other_feats) > 0:
            inter = torch.cdist(c_feats, other_feats)
            inter_dists.append(inter.mean().item())

    if intra_dists and inter_dists:
        intra_mean = sum(intra_dists) / len(intra_dists)
        inter_mean = sum(inter_dists) / len(inter_dists)
        # Clarity: high when inter >> intra. Scale to 0-100.
        ratio = inter_mean / (intra_mean + 1e-8)
        kpis["clarity_score"] = min(100.0, max(0.0, (ratio - 1.0) * 50.0))
    else:
        kpis["clarity_score"] = 0.0

    # ── Drift Velocity ──
    # JEPA prediction error as dynamics proxy
    # High error = entities are in unpredictable/active evolution
    model.eval()
    pred_errors = []
    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            sample = dataset[i]
            fp = sample["fingerprints"].unsqueeze(0).to(device)
            coords = sample.get("coords")
            if coords is not None:
                coords = coords.unsqueeze(0).to(device)
            adj = sample.get("adjacency")
            if adj is not None:
                adj = adj.unsqueeze(0).to(device)
            vel = sample.get("velocity")
            if vel is not None:
                vel = vel.unsqueeze(0).to(device)

            # Encode full (no masking) for prediction error estimate
            z_ctx = model.context_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
            z_tgt = model.target_encoder(fp, coords=coords, adjacency=adj, velocity=vel)

            error = (z_ctx - z_tgt).pow(2).mean().item()
            pred_errors.append(error)

    kpis["drift_velocity"] = sum(pred_errors) / len(pred_errors) if pred_errors else 0.0

    # ── Opportunity Surface ──
    # Count of high-similarity cross-type connections in embedding space
    # These are entities of different types that are nonetheless close in
    # learned representation — anomalous connections indicating opportunities
    if hasattr(dataset, 'adjacency'):
        # Aggregate entity-level features
        entity_feat = torch.zeros(dataset.num_entities, features.shape[1])
        entity_count = torch.zeros(dataset.num_entities)
        for i in range(len(indices)):
            eid = indices[i].item()
            if eid < dataset.num_entities:
                entity_feat[eid] += features[i]
                entity_count[eid] += 1

        valid_mask = entity_count > 0
        entity_feat[valid_mask] /= entity_count[valid_mask].unsqueeze(1)
        entity_feat_norm = F.normalize(entity_feat, dim=1)

        valid_idx = torch.where(valid_mask)[0]
        if len(valid_idx) > 10:
            v_feat = entity_feat_norm[valid_idx]
            v_labels = dataset.entity_labels[valid_idx]
            sim = v_feat @ v_feat.t()

            # Cross-type high-similarity pairs
            cross_type = v_labels.unsqueeze(0) != v_labels.unsqueeze(1)
            high_sim = sim > 0.7
            opportunities = (cross_type & high_sim).sum().item()
            kpis["opportunity_surface"] = opportunities
        else:
            kpis["opportunity_surface"] = 0
    else:
        kpis["opportunity_surface"] = 0

    # ── Risk Horizon ──
    # Entities near cluster boundaries in embedding space
    # Close to decision boundary = unstable classification = risk
    if num_classes > 1:
        # Compute distance to nearest different-class centroid
        centroids = torch.zeros(num_classes, features.shape[1])
        for c in range(num_classes):
            c_mask = labels == c
            if c_mask.sum() > 0:
                centroids[c] = features[c_mask].mean(0)

        boundary_count = 0
        for i in range(len(features)):
            own_class = labels[i].item()
            own_dist = (features[i] - centroids[own_class]).norm()
            # Distance to nearest other centroid
            other_dists = []
            for c in range(num_classes):
                if c != own_class:
                    other_dists.append((features[i] - centroids[c]).norm().item())
            if other_dists:
                nearest_other = min(other_dists)
                # Near boundary if own_dist / nearest_other > 0.7
                if own_dist.item() / (nearest_other + 1e-8) > 0.7:
                    boundary_count += 1

        kpis["risk_horizon"] = boundary_count
    else:
        kpis["risk_horizon"] = 0

    # ── Topological features from TCD ──
    if recursive_loop is not None:
        kpis["tcd_num_modules"] = recursive_loop.num_modules
        kpis["tcd_converged"] = recursive_loop.is_converged

        # Extract H0/H1/H2 counts from crystallization history
        history = recursive_loop.crystallizer.crystallization_history
        if history:
            last = history[-1]
            module_types = last.get("registry_stats", {}).get("module_types", {})
            kpis["tcd_attractors_h0"] = module_types.get("attractor", 0)
            kpis["tcd_cycles_h1"] = module_types.get("cycle", 0)
            kpis["tcd_boundaries_h2"] = module_types.get("boundary", 0)
            kpis["tcd_total_persistence"] = last.get("total_persistence", 0.0)

    return kpis


def run_full_manifold_evaluation(
    model: nn.Module,
    dataset,
    device: torch.device,
    num_classes: int,
    max_samples: int = 500,
    recursive_loop=None,
) -> dict:
    """Run complete evaluation suite for Latent Ocean TCD-JEPA."""
    features, labels, indices = extract_manifold_features(model, dataset, device, max_samples)

    results = {}

    # Standard SSL metrics
    lp = linear_probe(features, labels, num_classes)
    results["linear_probe_train"] = lp["train_acc"]
    results["linear_probe_test"] = lp["test_acc"]

    knn = knn_evaluate(features, labels)
    results.update(knn)

    # Causal link prediction
    if hasattr(dataset, "adjacency") and dataset.adjacency.numel() > 0:
        clp = causal_link_prediction(features, dataset.adjacency, indices, dataset.num_entities)
        results.update(clp)
    elif hasattr(dataset, "_sparse_coo") and dataset._sparse_coo is not None:
        # Build adjacency for evaluated subset from COO edges
        try:
            src_all, tgt_all, val_all = dataset._sparse_coo
            valid_entities = indices.unique()
            n_eval = int(valid_entities.max().item()) + 1
            local_adj = torch.zeros(n_eval, n_eval)
            mask_src = src_all < n_eval
            mask_tgt = tgt_all < n_eval
            mask_both = mask_src & mask_tgt
            if mask_both.any():
                local_adj[src_all[mask_both], tgt_all[mask_both]] = val_all[mask_both]
            clp = causal_link_prediction(features, local_adj, indices, n_eval)
            results.update(clp)
        except Exception as e:
            logging.getLogger("tcd_jepa").warning(f"Causal link prediction skipped: {e}")

    # Manifold neighborhood preservation
    if hasattr(dataset, "coords"):
        all_coords = []
        for i in range(min(max_samples, len(dataset))):
            all_coords.append(dataset[i]["coords"])
        coords_flat = torch.cat(all_coords, dim=0)[:len(features)]
        mnp = manifold_neighborhood_preservation(features, coords_flat)
        results.update(mnp)

    # Representation quality
    results.update(compute_representation_metrics(features))

    # Latent Ocean KPIs
    kpis = compute_latent_ocean_kpis(model, dataset, device, recursive_loop, max_samples=200)
    results.update(kpis)

    return results
