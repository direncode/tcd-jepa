#!/usr/bin/env python
"""Analyze TCD crystallized modules for interpretability.

Maps each crystallized module back to real semiconductor entities to show
what topological structure TCD discovered in the supply chain.

Usage:
    python scripts/analyze_modules.py \
        --checkpoint logs/manifold/checkpoints/manifold_checkpoint_0099.pt \
        --data-dir ./data/semiconductor \
        --entities-json ./data/semiconductor/entities.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("module_analysis")


def analyze_modules(
    model,
    dataset,
    entities: list[dict],
    device: torch.device,
    num_samples: int = 200,
) -> list[dict]:
    """Analyze what each crystallized module responds to.

    For each module, finds the entities that activate it most strongly,
    revealing what topological feature the module represents.
    """
    model.eval()

    if not hasattr(model, "predictor") or not hasattr(model.predictor, "registry"):
        logger.warning("Model has no dynamic predictor / registry")
        return []

    modules = model.predictor.registry.get_all_modules()
    if not modules:
        logger.warning("No crystallized modules found")
        return []

    logger.info(f"Analyzing {len(modules)} crystallized modules...")

    # Collect entity representations
    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            fingerprints = sample["fingerprints"].unsqueeze(0).to(device)
            coords = sample.get("coords")
            if coords is not None:
                coords = coords.unsqueeze(0).to(device)
            adjacency = sample.get("adjacency")
            if adjacency is not None:
                adjacency = adjacency.unsqueeze(0).to(device)
            velocity = sample.get("velocity")
            if velocity is not None:
                velocity = velocity.unsqueeze(0).to(device)

            z = model.context_encoder(
                fingerprints, coords=coords, adjacency=adjacency, velocity=velocity,
            )
            all_embeddings.append(z.squeeze(0))
            all_indices.append(sample["indices"])

    embeddings = torch.cat(all_embeddings, dim=0)  # [total_tokens, D]
    indices = torch.cat(all_indices, dim=0)  # [total_tokens]
    logger.info(f"Collected {embeddings.shape[0]} token embeddings")

    # Flatten for module analysis
    z_flat = embeddings.reshape(-1, embeddings.shape[-1])

    # Analyze each module
    module_analyses = []

    for module_id, module in modules:
        module = module.to(device)
        with torch.no_grad():
            # Get module output magnitude as activation score
            output = module(z_flat)
            activation = output.norm(dim=-1)  # [N]

        # Top-k activated tokens
        topk_vals, topk_idx = activation.topk(min(20, len(activation)))
        topk_global_indices = indices[topk_idx].tolist()

        # Map to entity names
        top_entities = []
        entity_types = {}
        entity_stages = {}

        for global_idx in topk_global_indices:
            if global_idx < len(entities):
                e = entities[global_idx]
                top_entities.append(e["name"])
                etype = e.get("type", "unknown")
                stage = e.get("stage", "unknown")
                entity_types[etype] = entity_types.get(etype, 0) + 1
                entity_stages[stage] = entity_stages.get(stage, 0) + 1

        # Determine dominant type and stage
        dominant_type = max(entity_types, key=entity_types.get) if entity_types else "unknown"
        dominant_stage = max(entity_stages, key=entity_stages.get) if entity_stages else "unknown"

        # Activation statistics
        act_mean = activation.mean().item()
        act_std = activation.std().item()
        act_max = activation.max().item()

        analysis = {
            "module_id": module_id,
            "dominant_type": dominant_type,
            "dominant_stage": dominant_stage,
            "type_distribution": entity_types,
            "stage_distribution": entity_stages,
            "top_entities": top_entities[:10],
            "activation_mean": act_mean,
            "activation_std": act_std,
            "activation_max": act_max,
        }
        module_analyses.append(analysis)

        # Generate human-readable label
        if dominant_type == "organization" and dominant_stage == "":
            label = f"Provider cluster ({dominant_type})"
        elif dominant_stage:
            label = f"{dominant_stage} — {dominant_type}"
        else:
            label = f"{dominant_type} cluster"

        logger.info(f"\n  Module {module_id}:")
        logger.info(f"    Label: {label}")
        logger.info(f"    Dominant type: {dominant_type} ({entity_types.get(dominant_type, 0)}/20)")
        logger.info(f"    Stage: {dominant_stage}")
        logger.info(f"    Top entities: {', '.join(top_entities[:5])}")
        logger.info(f"    Activation: mean={act_mean:.4f}, max={act_max:.4f}")

    return module_analyses


def find_hidden_connections(
    model,
    dataset,
    entities: list[dict],
    adjacency: torch.Tensor,
    device: torch.device,
    num_samples: int = 200,
    similarity_threshold: float = 0.8,
) -> list[dict]:
    """Find high-similarity entity pairs that have no direct supply chain link.

    These are potential hidden dependencies or undocumented relationships
    that TCD's representations suggest should exist.
    """
    model.eval()

    # Collect per-entity embeddings
    entity_embeddings = torch.zeros(len(entities), model.context_encoder.embed_dim)
    entity_counts = torch.zeros(len(entities))

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            fingerprints = sample["fingerprints"].unsqueeze(0).to(device)
            coords = sample.get("coords")
            if coords is not None:
                coords = coords.unsqueeze(0).to(device)
            adjacency_batch = sample.get("adjacency")
            if adjacency_batch is not None:
                adjacency_batch = adjacency_batch.unsqueeze(0).to(device)
            velocity = sample.get("velocity")
            if velocity is not None:
                velocity = velocity.unsqueeze(0).to(device)

            z = model.context_encoder(
                fingerprints, coords=coords, adjacency=adjacency_batch, velocity=velocity,
            )
            z = z.squeeze(0).cpu()

            for j, idx in enumerate(sample["indices"]):
                eid = idx.item()
                if eid < len(entities):
                    entity_embeddings[eid] += z[j] if j < z.shape[0] else z[0]
                    entity_counts[eid] += 1

    # Average embeddings
    mask = entity_counts > 0
    entity_embeddings[mask] /= entity_counts[mask].unsqueeze(1)

    # Normalize
    valid = torch.where(mask)[0]
    valid_emb = F.normalize(entity_embeddings[valid], dim=1)

    # Compute similarity
    sim = valid_emb @ valid_emb.t()

    # Find high-similarity pairs with no direct link
    hidden_connections = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ei, ej = valid[i].item(), valid[j].item()
            if sim[i, j] > similarity_threshold and adjacency[ei, ej] == 0 and adjacency[ej, ei] == 0:
                # Different entity types = more interesting
                e1, e2 = entities[ei], entities[ej]
                if e1.get("type") != e2.get("type") or e1.get("stage") != e2.get("stage"):
                    hidden_connections.append({
                        "entity_1": e1["name"],
                        "entity_2": e2["name"],
                        "type_1": e1.get("type", "?"),
                        "type_2": e2.get("type", "?"),
                        "stage_1": e1.get("stage", "?"),
                        "stage_2": e2.get("stage", "?"),
                        "similarity": sim[i, j].item(),
                    })

    # Sort by similarity
    hidden_connections.sort(key=lambda x: -x["similarity"])

    logger.info(f"\n{'='*60}")
    logger.info(f"HIDDEN CONNECTIONS (similarity > {similarity_threshold}, no direct link)")
    logger.info(f"{'='*60}")
    for idx, conn in enumerate(hidden_connections[:20]):
        logger.info(
            f"  {idx+1}. {conn['entity_1']} ({conn['type_1']}) <-> "
            f"{conn['entity_2']} ({conn['type_2']}) "
            f"sim={conn['similarity']:.4f}"
        )

    return hidden_connections


def main():
    parser = argparse.ArgumentParser(description="Analyze TCD crystallized modules")
    parser.add_argument("--checkpoint", required=True, help="TCD model checkpoint")
    parser.add_argument("--data-dir", default="./data/semiconductor")
    parser.add_argument("--entities-json", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load entities metadata
    entities_path = args.entities_json or str(data_dir / "entities.json")
    with open(entities_path) as f:
        entities = json.load(f)
    logger.info(f"Loaded {len(entities)} entity records")

    # Load data
    adjacency = torch.load(data_dir / "adjacency.pt", weights_only=True)

    # Build model (must match training config)
    from tcd_jepa.manifold.dataset import LatentOceanDataset
    from tcd_jepa.manifold.model import build_manifold_jepa

    model = build_manifold_jepa(
        fingerprint_dim=384, num_tokens=64, coord_dim=3,
        embed_dim=192, depth=6, num_heads=3,
        predictor_embed_dim=96, predictor_depth=4, predictor_num_heads=3,
        use_dynamic_predictor=True,
        use_causal_encoding=True, use_velocity_encoding=True, sphere_radius=4.5,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.context_encoder.load_state_dict(ckpt["encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset
    dataset = LatentOceanDataset(
        str(data_dir), num_tokens=64, num_samples=200,
    )

    # Analyze modules
    logger.info(f"\n{'='*60}")
    logger.info("MODULE INTERPRETABILITY ANALYSIS")
    logger.info(f"{'='*60}")
    module_analyses = analyze_modules(model, dataset, entities, device)

    # Find hidden connections
    hidden = find_hidden_connections(
        model, dataset, entities, adjacency, device,
        similarity_threshold=0.7,
    )

    # Save results
    output = {
        "modules": module_analyses,
        "hidden_connections": hidden[:50],
    }
    output_path = data_dir / "tcd_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()
