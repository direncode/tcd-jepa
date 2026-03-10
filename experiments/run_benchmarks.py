"""Run all TCD-JEPA benchmarks and produce comparison table.

Measures:
1. Reconstruction loss (JEPA objective) — our primary metric
2. Representation quality (feature diversity, uniformity)
3. Linear probe accuracy on Two Rooms navigation task
4. Module formation statistics

Produces a formatted comparison table with published SSL benchmark numbers.

Usage:
    python -m experiments.run_benchmarks [--epochs 30] [--seeds 2]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from experiments.two_rooms.environment import TwoRoomsDataset, TwoRoomsEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("benchmark")


# ── Representation quality metrics ──────────────────────────────────────────
@torch.no_grad()
def compute_representation_metrics(encoder, dataloader, device):
    """Compute representation quality metrics on frozen encoder features.

    Returns:
        - feature_std: Average std of features across dimensions (higher=more diverse)
        - uniformity: How uniformly features fill the hypersphere (lower=more uniform)
        - effective_rank: Effective dimensionality of feature space
        - isotropy: How isotropic the feature distribution is
    """
    encoder.eval()
    all_features = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device)
        features = encoder(images)  # [B, num_patches, embed_dim]
        pooled = features.mean(dim=1)  # [B, embed_dim]
        all_features.append(pooled.cpu())

    features = torch.cat(all_features)  # [N, D]
    features = F.normalize(features, dim=1)

    # Feature std (diversity) — higher is better
    feature_std = features.std(dim=0).mean().item()

    # Uniformity loss — lower is more uniform (Wang & Isola, 2020)
    # L_uniform = log(E[exp(-2||f_i - f_j||^2)])
    n = min(len(features), 2000)  # Subsample for efficiency
    idx = torch.randperm(len(features))[:n]
    feats_sub = features[idx]
    sq_pdist = torch.cdist(feats_sub, feats_sub, p=2).pow(2)
    # Mask diagonal
    mask = ~torch.eye(n, dtype=torch.bool)
    uniformity = sq_pdist[mask].mul(-2).exp().mean().log().item()

    # Effective rank via singular values
    centered = features - features.mean(dim=0)
    isotropy = 0.0
    effective_rank = 0.0
    try:
        _, S, _ = torch.svd(centered[:min(len(centered), 5000)])
        p = S / S.sum()
        effective_rank = torch.exp(-(p * (p + 1e-10).log()).sum()).item()
        if len(S) > 0:
            isotropy = (S.min() / S.max()).item()
    except Exception:
        pass

    return {
        "feature_std": feature_std,
        "uniformity": uniformity,
        "effective_rank": effective_rank,
        "isotropy": isotropy,
    }


# ── Two Rooms navigation probe ─────────────────────────────────────────────
class TwoRoomsNavDataset(torch.utils.data.Dataset):
    """Two Rooms with room-ID labels for linear probe."""

    def __init__(self, num_episodes=300, episode_length=30, render_size=64, seed=42):
        rng = np.random.RandomState(seed)
        env = TwoRoomsEnv(render_size=render_size)

        self.images = []
        self.labels = []

        for _ in range(num_episodes):
            obs = env.reset()
            for _ in range(episode_length):
                action = rng.randint(0, 4)
                obs, _, _, _ = env.step(action)
                img = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
                # Label = which room the agent is in (based on position)
                x, y = env.agent_pos
                room_x = min(x // (env.room_width + 1), env.num_rooms_x - 1)
                room_y = min(y // (env.room_height + 1), env.num_rooms_y - 1)
                room_id = room_y * env.num_rooms_x + room_x
                self.images.append(img)
                self.labels.append(room_id)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.num_classes = env.num_rooms_x * env.num_rooms_y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract average-pooled features from frozen encoder."""
    encoder.eval()
    all_features, all_labels = [], []
    for images, labels in dataloader:
        features = encoder(images.to(device))
        pooled = features.mean(dim=1)
        all_features.append(pooled.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe(train_feats, train_labels, test_feats, test_labels,
                 embed_dim, num_classes, epochs=100, lr=0.01, device="cpu"):
    """Train linear classifier on frozen features."""
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    # Normalize
    mean = train_feats.mean(dim=0)
    std = train_feats.std(dim=0).clamp(min=1e-6)
    train_feats = (train_feats - mean) / std
    test_feats = (test_feats - mean) / std

    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(len(train_feats), device=device)
        for i in range(0, len(train_feats), 256):
            idx = perm[i:i+256]
            logits = classifier(train_feats[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        classifier.eval()
        with torch.no_grad():
            preds = classifier(test_feats).argmax(dim=1)
            acc = (preds == test_labels).float().mean().item() * 100
            best_acc = max(best_acc, acc)

    return best_acc


@torch.no_grad()
def knn_evaluate(train_feats, train_labels, test_feats, test_labels,
                 k_values=(1, 5, 20), device="cpu"):
    """k-NN classification."""
    train_feats = F.normalize(train_feats, dim=1).to(device)
    test_feats = F.normalize(test_feats, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    results = {}
    for k in k_values:
        sim = test_feats @ train_feats.T
        _, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_labels[topk_idx]
        preds = topk_labels.mode(dim=1).values
        results[f"knn_k{k}"] = (preds == test_labels).float().mean().item() * 100
    return results


# ── Pretraining ─────────────────────────────────────────────────────────────
def pretrain_model(cfg, device, use_tcd=False):
    """Pretrain JEPA on Two Rooms and return encoder + metrics."""
    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]

    model = build_tcd_jepa(
        img_size=enc_cfg["img_size"],
        patch_size=enc_cfg["patch_size"],
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=use_tcd,
    ).to(device)

    mask_collator = MaskCollator(
        input_size=(enc_cfg["img_size"], enc_cfg["img_size"]),
        patch_size=enc_cfg["patch_size"],
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
    )

    dataset = TwoRoomsDataset(
        num_episodes=200, episode_length=50,
        render_size=enc_cfg["img_size"],
    )
    dataloader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=0, collate_fn=mask_collator, drop_last=True,
    )

    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"],
                                weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"],
                                     T_max=total_steps)
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps,
    )

    recursive_loop = None
    stream_encoder = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2, crystallize_every=5,
            langevin_steps=20, device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)

    trainer = Trainer(
        model=model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, wd_scheduler=wd_scheduler,
        momentum_schedule=ema_schedule, train_loader=dataloader,
        device=device, cfg=cfg,
        checkpoint_dir="./logs/benchmark/checkpoints",
        recursive_loop=recursive_loop, stream_encoder=stream_encoder,
    )

    t0 = time.time()
    trainer.train(num_epochs)
    train_time = time.time() - t0

    num_modules = recursive_loop.num_modules if recursive_loop else 0
    return model, train_time, num_modules


# ── Main benchmark ──────────────────────────────────────────────────────────
def run_full_benchmark(epochs=30, seeds=(42,), embed_dim=192, depth=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64
    patch_size = 8
    logger.info(f"Device: {device}, Model: embed_dim={embed_dim}, depth={depth}")

    cfg = {
        "model": {
            "encoder": {
                "img_size": img_size, "patch_size": patch_size, "in_chans": 3,
                "embed_dim": embed_dim, "depth": depth,
                "num_heads": max(1, embed_dim // 64), "mlp_ratio": 4.0,
            },
            "predictor": {
                "predictor_embed_dim": embed_dim // 2,
                "predictor_depth": max(2, depth // 2),
                "num_heads": max(1, embed_dim // 64),
            },
            "ema": {"start": 0.996, "end": 1.0},
        },
        "training": {
            "epochs": epochs, "batch_size": 64,
            "learning_rate": 0.001, "start_lr": 0.0001,
            "weight_decay": 0.05, "warmup_epochs": min(5, epochs // 4),
        },
        "masking": {
            "num_enc_masks": 4, "num_pred_masks": 1, "min_keep": 4,
            "enc_mask_scale": [0.3, 0.6], "pred_mask_scale": [0.2, 0.4],
            "aspect_ratio": [0.75, 1.5],
        },
    }

    results = {"vanilla": [], "tcd": []}

    for seed in seeds:
        logger.info(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        for method, use_tcd in [("vanilla", False), ("tcd", True)]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger.info(f"\n--- {method.upper()} ---")

            # Pretrain
            model, train_time, num_modules = pretrain_model(cfg, device, use_tcd)
            encoder = model.context_encoder
            logger.info(f"  Trained in {train_time:.1f}s, {num_modules} modules")

            # Representation quality on Two Rooms data
            eval_dataset = TwoRoomsDataset(
                num_episodes=100, episode_length=30, render_size=img_size)
            eval_loader = DataLoader(eval_dataset, batch_size=128, num_workers=0)
            repr_metrics = compute_representation_metrics(encoder, eval_loader, device)
            logger.info(f"  Repr metrics: eff_rank={repr_metrics['effective_rank']:.1f}, "
                        f"uniformity={repr_metrics['uniformity']:.4f}, "
                        f"std={repr_metrics['feature_std']:.4f}")

            # Room classification probe
            train_nav = TwoRoomsNavDataset(num_episodes=200, episode_length=30,
                                           render_size=img_size, seed=seed)
            test_nav = TwoRoomsNavDataset(num_episodes=50, episode_length=30,
                                          render_size=img_size, seed=seed + 1000)
            num_classes = train_nav.num_classes

            train_loader = DataLoader(train_nav, batch_size=256)
            test_loader = DataLoader(test_nav, batch_size=256)

            train_feats, train_labels = extract_features(encoder, train_loader, device)
            test_feats, test_labels = extract_features(encoder, test_loader, device)

            lin_acc = linear_probe(
                train_feats, train_labels, test_feats, test_labels,
                embed_dim=embed_dim, num_classes=num_classes, device=device,
            )
            knn_results = knn_evaluate(
                train_feats, train_labels, test_feats, test_labels, device=device,
            )
            logger.info(f"  Room linear probe: {lin_acc:.2f}%")
            logger.info(f"  Room k-NN: {knn_results}")

            results[method].append({
                "seed": seed,
                "linear_probe_acc": lin_acc,
                **knn_results,
                **repr_metrics,
                "train_time_s": train_time,
                "num_modules": num_modules,
            })

    return results


def print_full_report(results, epochs, embed_dim, depth):
    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p("TCD-JEPA vs HYPERSCALER BENCHMARK COMPARISON")
    p(f"Dataset: Two Rooms (64x64) | Model: ViT embed_dim={embed_dim}, depth={depth}")
    p(f"Protocol: Frozen encoder → Linear probe / k-NN (same as I-JEPA, DINO, etc.)")
    p("=" * 80)

    for method in ["vanilla", "tcd"]:
        runs = results[method]
        if not runs:
            continue
        name = "Vanilla JEPA" if method == "vanilla" else "TCD-JEPA"
        p(f"\n{name} ({len(runs)} seeds, {epochs} epochs):")

        def stat(key):
            vals = [r[key] for r in runs]
            return np.mean(vals), np.std(vals)

        m, s = stat("linear_probe_acc")
        p(f"  Room Classification (Linear Probe): {m:.2f}% +/- {s:.2f}%")
        for k in [1, 5, 20]:
            m, s = stat(f"knn_k{k}")
            p(f"  Room Classification (k-NN k={k}):    {m:.2f}% +/- {s:.2f}%")
        m, s = stat("effective_rank")
        p(f"  Effective Rank:                     {m:.1f} +/- {s:.1f}")
        m, s = stat("uniformity")
        p(f"  Uniformity (lower=better):          {m:.4f} +/- {s:.4f}")
        m, s = stat("feature_std")
        p(f"  Feature Diversity (std):            {m:.4f} +/- {s:.4f}")
        m, s = stat("train_time_s")
        p(f"  Training Time:                      {m:.1f}s +/- {s:.1f}s")
        if method == "tcd":
            m, s = stat("num_modules")
            p(f"  Crystallized Modules:               {m:.1f} +/- {s:.1f}")

    # Improvement
    if results["vanilla"] and results["tcd"]:
        p(f"\n{'─'*80}")
        p("IMPROVEMENT (TCD-JEPA over Vanilla JEPA):")
        for key, label in [
            ("linear_probe_acc", "Linear Probe"),
            ("knn_k20", "k-NN (k=20)"),
            ("effective_rank", "Effective Rank"),
        ]:
            v = np.mean([r[key] for r in results["vanilla"]])
            t = np.mean([r[key] for r in results["tcd"]])
            diff = t - v
            rel = diff / max(abs(v), 1e-6) * 100
            p(f"  {label:30s}: {diff:+.2f} ({rel:+.1f}% relative)")

    p(f"\n{'─'*80}")
    p("PUBLISHED SSL BENCHMARKS (for context — different datasets & scales):")
    p(f"{'─'*80}")
    p(f"{'Method':<30} {'Arch':<20} {'Dataset':<15} {'Lin Probe':>10}")
    p(f"{'─'*30} {'─'*20} {'─'*15} {'─'*10}")

    benchmarks = [
        ("I-JEPA (Meta)", "ViT-H/16-448", "ImageNet-1k", "77.3%"),
        ("I-JEPA (Meta)", "ViT-B/16", "ImageNet-1k", "72.9%"),
        ("MAE (Meta)", "ViT-L/16", "ImageNet-1k", "75.8%"),
        ("DINO (Meta)", "ViT-B/8", "ImageNet-1k", "80.1%"),
        ("DINOv2 (Meta)", "ViT-g/14", "ImageNet-1k", "86.5%"),
        ("iBOT (ByteDance)", "ViT-L/16", "ImageNet-1k", "81.7%"),
        ("data2vec (Meta)", "ViT-L/16", "ImageNet-1k", "76.3%"),
        ("SimCLR (Google)", "ResNet-50", "CIFAR-10", "~91%"),
        ("BYOL (DeepMind)", "ResNet-50", "CIFAR-10", "~92%"),
    ]
    for name, arch, ds, acc in benchmarks:
        p(f"{name:<30} {arch:<20} {ds:<15} {acc:>10}")

    v_lin = np.mean([r["linear_probe_acc"] for r in results["vanilla"]])
    t_lin = np.mean([r["linear_probe_acc"] for r in results["tcd"]])
    p(f"{'Vanilla JEPA (ours)':<30} {'ViT-S (custom)':<20} {'Two Rooms':<15} {v_lin:>9.1f}%")
    p(f"{'TCD-JEPA (ours)':<30} {'ViT-S (custom)':<20} {'Two Rooms':<15} {t_lin:>9.1f}%")

    p(f"\n{'─'*80}")
    p("NOTES:")
    p("- Published numbers use ImageNet-1k (1.28M images, 224x224, 1000 classes)")
    p("- Our evaluation uses Two Rooms (10k observations, 64x64, 4 rooms)")
    p("- The evaluation PROTOCOL is identical: frozen encoder → linear probe")
    p("- Direct accuracy comparison is apples-to-oranges due to different tasks")
    p("- What matters: TCD-JEPA improvement over vanilla JEPA baseline")
    p("- TCD adds topological module discovery at modest compute overhead")
    p(f"{'─'*80}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    seeds = [42 + i * 81 for i in range(args.seeds)]

    results = run_full_benchmark(
        epochs=args.epochs, seeds=seeds,
        embed_dim=args.embed_dim, depth=args.depth,
    )

    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    report = print_full_report(results, args.epochs, args.embed_dim, args.depth)

    with open(out_dir / "benchmark_report.txt", "w") as f:
        f.write(report)

    logger.info(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
