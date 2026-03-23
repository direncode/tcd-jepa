#!/usr/bin/env python3
"""TCD-JEPA H100 RunPod Benchmark — 12-Hour Maximum Value Run.

Runs multi-scale experiments (ViT-S/B/L) with multiple seeds to establish
statistical significance of TCD-JEPA vs vanilla JEPA.

Optimizations:
  - Flash Attention 2 via F.scaled_dot_product_attention
  - BF16 mixed precision (H100 tensor cores)
  - torch.compile with reduce-overhead mode
  - TF32 matmul, cuDNN benchmark mode
  - Optimized data pipeline (pin_memory, persistent_workers, prefetch)
  - Gradient clipping for stable large-batch training

Usage:
    python run_h100_benchmark.py                    # Full 12h run
    python run_h100_benchmark.py --phase small      # Only ViT-S experiments
    python run_h100_benchmark.py --phase base       # Only ViT-B experiments
    python run_h100_benchmark.py --quick-test       # 5-epoch sanity check
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Subset

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("h100_benchmark.log"),
    ],
)
logger = logging.getLogger("h100_bench")


# ============================================================================
# Hardware Optimization
# ============================================================================
def setup_h100_optimizations():
    """Configure CUDA for maximum H100 throughput."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU (will be very slow)")
        return torch.device("cpu")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    logger.info(f"GPU: {props.name} ({props.total_mem / 1e9:.1f} GB)")

    # TF32 for matmul — free 3x speedup on Ampere+ with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark — finds fastest conv algorithms for fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Enable Flash Attention backend preference
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    logger.info("H100 optimizations: TF32=ON, cuDNN.benchmark=ON, Flash SDP=ON")
    return torch.device("cuda")


# ============================================================================
# Data Pipeline
# ============================================================================
class SSLWrapper(torch.utils.data.Dataset):
    """Drop labels for SSL pretraining."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx][0]


def get_cifar10(data_dir="./data"):
    """Download and prepare CIFAR-10."""
    logger.info("Preparing CIFAR-10 dataset...")
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([T.ToTensor(), normalize])
    ssl_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(32, scale=(0.6, 1.0)),
        T.ToTensor(),
        normalize,
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)
    ssl_ds = SSLWrapper(torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=ssl_transform))

    logger.info(f"CIFAR-10: {len(train_ds)} train, {len(test_ds)} test")
    return train_ds, test_ds, ssl_ds


# ============================================================================
# Feature Extraction & Evaluation
# ============================================================================
@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract average-pooled features from frozen encoder."""
    encoder.eval()
    all_features, all_labels = [], []
    for images, labels in dataloader:
        features = encoder(images.to(device, non_blocking=True))
        pooled = features.mean(dim=1)
        all_features.append(pooled.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe(train_feats, train_labels, test_feats, test_labels,
                 embed_dim, num_classes=10, epochs=100, lr=0.01, device="cpu"):
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
        for i in range(0, len(train_feats), 512):
            idx = perm[i:i + 512]
            logits = classifier(train_feats[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad(set_to_none=True)
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
                 k_values=(1, 5, 10, 20), device="cpu"):
    """k-NN classification on frozen features."""
    train_feats = F.normalize(train_feats, dim=1).to(device)
    test_feats = F.normalize(test_feats, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    results = {}
    chunk_size = 2000
    for k in k_values:
        correct = 0
        for start in range(0, len(test_feats), chunk_size):
            end = min(start + chunk_size, len(test_feats))
            sim = test_feats[start:end] @ train_feats.T
            _, topk_idx = sim.topk(k, dim=1)
            topk_labels = train_labels[topk_idx]
            preds = topk_labels.mode(dim=1).values
            correct += (preds == test_labels[start:end]).sum().item()
        results[f"knn_k{k}"] = correct / len(test_feats) * 100
    return results


@torch.no_grad()
def compute_representation_quality(encoder, dataloader, device):
    """Compute representation quality metrics."""
    encoder.eval()
    all_features = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        features = encoder(images.to(device, non_blocking=True))
        all_features.append(features.mean(dim=1).cpu())

    features = torch.cat(all_features)
    features = F.normalize(features, dim=1)

    feature_std = features.std(dim=0).mean().item()

    # Uniformity
    n = min(len(features), 3000)
    idx = torch.randperm(len(features))[:n]
    feats_sub = features[idx]
    sq_pdist = torch.cdist(feats_sub, feats_sub, p=2).pow(2)
    mask = ~torch.eye(n, dtype=torch.bool)
    uniformity = sq_pdist[mask].mul(-2).exp().mean().log().item()

    # Effective rank
    centered = features - features.mean(dim=0)
    effective_rank = 0.0
    try:
        _, S, _ = torch.svd(centered[:min(len(centered), 5000)])
        p = S / S.sum()
        effective_rank = torch.exp(-(p * (p + 1e-10).log()).sum()).item()
    except Exception:
        pass

    return {
        "feature_std": feature_std,
        "uniformity": uniformity,
        "effective_rank": effective_rank,
    }


# ============================================================================
# Pretraining
# ============================================================================
def pretrain(cfg, ssl_ds, device, use_tcd=False, seed=42, compile_model=True):
    """Pretrain JEPA/TCD-JEPA and return encoder + metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {num_params:,}")

    # torch.compile for 10-30% speedup
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("  torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"  torch.compile failed: {e}")

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

    data_cfg = cfg.get("data", {})
    dataloader = DataLoader(
        ssl_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        collate_fn=mask_collator,
        drop_last=True,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True) and data_cfg.get("num_workers", 8) > 0,
        prefetch_factor=data_cfg.get("prefetch_factor", 4) if data_cfg.get("num_workers", 8) > 0 else None,
    )

    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"],
                                weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg.get("warmup_epochs", 10) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
        final_lr=train_cfg.get("final_lr", 0.0),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=train_cfg["weight_decay"],
        T_max=total_steps,
        final_wd=train_cfg.get("final_weight_decay", train_cfg["weight_decay"]),
    )
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps,
    )

    recursive_loop = None
    stream_encoder = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2,
            crystallize_every=5,
            langevin_steps=20,
            device=device,
        )
        # Access the underlying model if compiled
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        stream_encoder = StreamEncoder(raw_model.context_encoder, raw_model.target_encoder)
        raw_model.set_module_registry(recursive_loop.crystallizer.registry)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule=ema_schedule,
        train_loader=dataloader,
        device=device,
        cfg=cfg,
        checkpoint_dir=f"./logs/h100_run/checkpoints/{'tcd' if use_tcd else 'vanilla'}_s{seed}",
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
    )

    t0 = time.time()
    trainer.train(num_epochs)
    train_time = time.time() - t0

    num_modules = recursive_loop.num_modules if recursive_loop else 0

    # Get the raw model for evaluation
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    encoder = raw_model.context_encoder

    logger.info(f"  Trained in {train_time:.1f}s ({train_time / 60:.1f}min), "
                f"{num_modules} modules, "
                f"{total_steps / train_time:.1f} steps/sec")

    return encoder, train_time, num_modules


# ============================================================================
# Single Experiment
# ============================================================================
def run_single_experiment(cfg, train_ds, test_ds, ssl_ds, device,
                          use_tcd, seed, compile_model=True):
    """Run one pretrain + eval experiment."""
    method = "TCD-JEPA" if use_tcd else "Vanilla JEPA"
    embed_dim = cfg["model"]["encoder"]["embed_dim"]
    depth = cfg["model"]["encoder"]["depth"]
    logger.info(f"  [{method}] seed={seed}, dim={embed_dim}, depth={depth}")

    encoder, train_time, num_modules = pretrain(
        cfg, ssl_ds, device, use_tcd=use_tcd, seed=seed,
        compile_model=compile_model,
    )

    # Evaluate
    eval_batch = 512
    train_loader = DataLoader(train_ds, batch_size=eval_batch, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=eval_batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)

    lin_acc = linear_probe(
        train_feats, train_labels, test_feats, test_labels,
        embed_dim=embed_dim, device=device,
    )
    knn_results = knn_evaluate(
        train_feats, train_labels, test_feats, test_labels, device=device,
    )
    repr_metrics = compute_representation_quality(
        encoder, test_loader, device,
    )

    logger.info(f"    Linear: {lin_acc:.2f}%, kNN-20: {knn_results['knn_k20']:.2f}%, "
                f"eff_rank: {repr_metrics['effective_rank']:.1f}")

    result = {
        "method": "tcd" if use_tcd else "vanilla",
        "seed": seed,
        "embed_dim": embed_dim,
        "depth": depth,
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "linear_probe_acc": lin_acc,
        **knn_results,
        **repr_metrics,
        "train_time_s": train_time,
        "num_modules": num_modules,
        "num_params": sum(p.numel() for p in encoder.parameters()),
    }

    # Free memory
    del encoder, train_feats, test_feats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# Phase Runner
# ============================================================================
def run_phase(phase_name, phase_cfg, train_ds, test_ds, ssl_ds, device,
              compile_model=True, time_limit_s=None):
    """Run all experiments for one phase (scale)."""
    seeds = phase_cfg.get("seeds", [42])
    results = {"vanilla": [], "tcd": []}
    start = time.time()

    for seed in seeds:
        for method, use_tcd in [("vanilla", False), ("tcd", True)]:
            if time_limit_s and (time.time() - start) > time_limit_s:
                logger.warning(f"  Phase {phase_name} time limit reached, skipping remaining")
                return results

            result = run_single_experiment(
                phase_cfg, train_ds, test_ds, ssl_ds, device,
                use_tcd=use_tcd, seed=seed, compile_model=compile_model,
            )
            results[method].append(result)

            # Save incremental results
            out_dir = Path(f"results/h100_run/{phase_name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "results_incremental.json", "w") as f:
                json.dump(results, f, indent=2)

    return results


# ============================================================================
# Report Generation
# ============================================================================
def generate_report(all_results, output_path):
    """Generate comprehensive comparison report."""
    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 90)
    p("TCD-JEPA H100 BENCHMARK REPORT")
    p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Dataset: CIFAR-10 (50k train / 10k test, 32x32)")
    p(f"Protocol: SSL pretrain → Frozen encoder → Linear probe / k-NN")
    p("=" * 90)

    for phase_name, results in all_results.items():
        if not results.get("vanilla") and not results.get("tcd"):
            continue

        sample = (results.get("vanilla") or results.get("tcd", [{}]))[0]
        p(f"\n{'─' * 90}")
        p(f"PHASE: {phase_name.upper()} (ViT dim={sample.get('embed_dim')}, "
          f"depth={sample.get('depth')}, {sample.get('epochs')} epochs, "
          f"batch={sample.get('batch_size')})")
        p(f"{'─' * 90}")

        for method in ["vanilla", "tcd"]:
            runs = results.get(method, [])
            if not runs:
                continue

            name = "Vanilla JEPA" if method == "vanilla" else "TCD-JEPA"
            p(f"\n  {name} ({len(runs)} seeds):")

            def stat(key):
                vals = [r[key] for r in runs if key in r]
                if not vals:
                    return 0.0, 0.0
                return np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0

            m, s = stat("linear_probe_acc")
            p(f"    Linear Probe:       {m:6.2f}% ± {s:.2f}%")
            for k in [1, 5, 10, 20]:
                m, s = stat(f"knn_k{k}")
                p(f"    k-NN (k={k:2d}):       {m:6.2f}% ± {s:.2f}%")
            m, s = stat("effective_rank")
            p(f"    Effective Rank:     {m:6.1f} ± {s:.1f}")
            m, s = stat("uniformity")
            p(f"    Uniformity:         {m:7.4f} ± {s:.4f}")
            m, s = stat("feature_std")
            p(f"    Feature Diversity:  {m:7.4f} ± {s:.4f}")
            m, s = stat("train_time_s")
            p(f"    Train Time:         {m:6.1f}s ± {s:.1f}s")
            if method == "tcd":
                m, s = stat("num_modules")
                p(f"    Modules:            {m:6.1f} ± {s:.1f}")

        # Improvement
        if results.get("vanilla") and results.get("tcd"):
            p(f"\n  IMPROVEMENT (TCD-JEPA over Vanilla):")
            for key, label in [
                ("linear_probe_acc", "Linear Probe"),
                ("knn_k20", "k-NN (k=20)"),
                ("effective_rank", "Effective Rank"),
            ]:
                v = np.mean([r[key] for r in results["vanilla"]])
                t = np.mean([r[key] for r in results["tcd"]])
                diff = t - v
                rel = diff / max(abs(v), 1e-6) * 100
                p(f"    {label:25s}: {diff:+.2f} ({rel:+.1f}% relative)")

    # Published baselines
    p(f"\n{'=' * 90}")
    p("PUBLISHED SSL BENCHMARKS (for reference)")
    p(f"{'=' * 90}")
    p(f"{'Method':<30} {'Arch':<22} {'Dataset':<15} {'Lin Probe':>10}")
    p(f"{'─' * 30} {'─' * 22} {'─' * 15} {'─' * 10}")

    baselines = [
        ("I-JEPA (Meta)", "ViT-H/16-448", "ImageNet-1k", "77.3%"),
        ("I-JEPA (Meta)", "ViT-B/16", "ImageNet-1k", "72.9%"),
        ("MAE (Meta)", "ViT-L/16", "ImageNet-1k", "75.8%"),
        ("DINO (Meta)", "ViT-B/8", "ImageNet-1k", "80.1%"),
        ("DINOv2 (Meta)", "ViT-g/14", "ImageNet-1k", "86.5%"),
        ("SimCLR (Google)", "ResNet-50", "CIFAR-10", "~93.6%"),
        ("BYOL (DeepMind)", "ResNet-50", "CIFAR-10", "~94.2%"),
        ("VICReg (Meta)", "ResNet-50", "CIFAR-10", "~92.1%"),
    ]
    for name, arch, ds, acc in baselines:
        p(f"{name:<30} {arch:<22} {ds:<15} {acc:>10}")

    p(f"\n{'─' * 90}")
    p("NOTES:")
    p("- Published ImageNet numbers use much larger models and data (1.28M images, 224x224)")
    p("- CIFAR-10 SSL linear probe state-of-art is ~94-96% (with large pretrained models)")
    p("- Direct comparison is apples-to-oranges — what matters is TCD vs vanilla delta")
    p("- Statistical significance: compare across seeds using t-test or Wilcoxon test")
    p(f"{'─' * 90}")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    return report


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="TCD-JEPA H100 Benchmark")
    parser.add_argument("--config", default="configs/h100_runpod.yaml")
    parser.add_argument("--phase", choices=["small", "base", "large", "all"],
                        default="all", help="Which phase(s) to run")
    parser.add_argument("--quick-test", action="store_true",
                        help="5-epoch sanity check")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--time-limit-hours", type=float, default=12.0,
                        help="Total time limit in hours")
    args = parser.parse_args()

    # Setup
    device = setup_h100_optimizations()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Data
    train_ds, test_ds, ssl_ds = get_cifar10(args.data_dir)

    # Quick test override
    if args.quick_test:
        logger.info("QUICK TEST MODE — 5 epochs, 1 seed")
        for phase in config["experiments"].values():
            phase["training"]["epochs"] = 5
            phase["seeds"] = [42]

    start_time = time.time()
    time_limit_s = args.time_limit_hours * 3600
    all_results = {}

    phases_to_run = (
        ["small", "base", "large"] if args.phase == "all"
        else [args.phase]
    )

    # Time budget allocation
    phase_time_fractions = {"small": 0.33, "base": 0.42, "large": 0.25}

    for phase_name in phases_to_run:
        if phase_name not in config["experiments"]:
            continue

        elapsed = time.time() - start_time
        remaining = time_limit_s - elapsed
        if remaining < 60:
            logger.warning(f"Time limit reached, skipping phase {phase_name}")
            break

        phase_time = remaining * phase_time_fractions.get(phase_name, 0.5)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PHASE: {phase_name.upper()} "
                     f"(budget: {phase_time / 3600:.1f}h, "
                     f"elapsed: {elapsed / 3600:.1f}h)")
        logger.info(f"{'=' * 60}")

        results = run_phase(
            phase_name,
            config["experiments"][phase_name],
            train_ds, test_ds, ssl_ds, device,
            compile_model=not args.no_compile,
            time_limit_s=phase_time,
        )
        all_results[phase_name] = results

    # Generate report
    out_dir = Path("results/h100_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    generate_report(all_results, out_dir / "final_report.txt")

    total_time = time.time() - start_time
    logger.info(f"\nTotal time: {total_time / 3600:.2f} hours")
    logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
