#!/usr/bin/env python3
"""Standard benchmark evaluation for TCD-JEPA on CIFAR-10, STL-10, and ImageNet-100.

Pretrains Vanilla JEPA and TCD-JEPA with ViT-S/16, then evaluates frozen
representations using linear probe and k-NN — the standard SSL evaluation protocol
used by I-JEPA, DINO, DINOv2, MAE, etc.

Usage:
    # CIFAR-10 (fastest, ~2h on A100)
    python -m benchmarks.run_standard_benchmarks --dataset cifar10

    # STL-10 (medium, ~3-4h on A100)
    python -m benchmarks.run_standard_benchmarks --dataset stl10

    # ImageNet-100 (longest, ~8-12h on A100)
    python -m benchmarks.run_standard_benchmarks --dataset imagenet100

    # Run all three sequentially
    python -m benchmarks.run_standard_benchmarks --dataset all

    # Quick test run (5 epochs, sanity check)
    python -m benchmarks.run_standard_benchmarks --dataset cifar10 --epochs 5 --quick
"""

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.logging import MetricLogger
from tcd_jepa.utils.config import load_config_with_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark")


# ── Published benchmarks for context ──────────────────────────────────────────
PUBLISHED_BENCHMARKS = {
    "I-JEPA ViT-S/16 (300ep)": {"imagenet1k_linear": 72.9, "cifar10_linear": "~92"},
    "I-JEPA ViT-B/16 (300ep)": {"imagenet1k_linear": 73.3},
    "DINO ViT-S/16 (300ep)": {"imagenet1k_linear": 77.0},
    "MAE ViT-B/16 (1600ep)": {"imagenet1k_linear": 68.0},
    "DINOv2 ViT-S/14": {"imagenet1k_linear": 81.1},
}


# ── Dataset helpers ───────────────────────────────────────────────────────────

class SSLWrapper(Dataset):
    """Drop labels from a dataset for self-supervised pretraining."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset from a folder structure.

    Expected layout:
        data_dir/train/class_0001/img_001.JPEG
        data_dir/train/class_0002/img_001.JPEG
        ...
        data_dir/val/class_0001/img_001.JPEG
        ...
    """
    def __init__(self, root, split="train", transform=None):
        from torchvision.datasets import ImageFolder
        self.dataset = ImageFolder(os.path.join(root, split), transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_ssl_transforms(dataset_name, img_size):
    """Get SSL pretraining transforms (no labels needed)."""
    if dataset_name == "cifar10":
        return T.Compose([
            T.Resize(img_size),
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif dataset_name == "stl10":
        return T.Compose([
            T.Resize(img_size),
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
    elif dataset_name == "imagenet100":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_eval_transforms(dataset_name, img_size):
    """Get evaluation transforms (deterministic, for feature extraction)."""
    if dataset_name == "cifar10":
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif dataset_name == "stl10":
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
    elif dataset_name == "imagenet100":
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_ssl_dataset(dataset_name, data_dir, img_size):
    """Get unlabeled dataset for SSL pretraining."""
    transform = get_ssl_transforms(dataset_name, img_size)

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        return SSLWrapper(ds)

    elif dataset_name == "stl10":
        # STL-10 has a dedicated 'unlabeled' split with 100k images
        ds = torchvision.datasets.STL10(
            root=data_dir, split="train+unlabeled", download=True, transform=transform)
        return SSLWrapper(ds)

    elif dataset_name == "imagenet100":
        ds = ImageNet100Dataset(data_dir, split="train", transform=transform)
        return SSLWrapper(ds)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_eval_datasets(dataset_name, data_dir, img_size):
    """Get labeled train/test datasets for evaluation."""
    transform = get_eval_transforms(dataset_name, img_size)
    num_classes = 10

    if dataset_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "stl10":
        train_ds = torchvision.datasets.STL10(
            root=data_dir, split="train", download=True, transform=transform)
        test_ds = torchvision.datasets.STL10(
            root=data_dir, split="test", download=True, transform=transform)

    elif dataset_name == "imagenet100":
        train_ds = ImageNet100Dataset(data_dir, split="train", transform=transform)
        test_ds = ImageNet100Dataset(data_dir, split="val", transform=transform)
        num_classes = 100

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, test_ds, num_classes


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract features from frozen encoder. Average pool over patch tokens."""
    encoder.eval()
    all_features, all_labels = [], []

    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1]
        features = encoder(images)          # [B, num_patches, embed_dim]
        features = features.mean(dim=1)     # [B, embed_dim]
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe(train_features, train_labels, test_features, test_labels,
                 embed_dim, num_classes=10, epochs=100, lr=0.3, device="cpu"):
    """Train a linear classifier on frozen features (DINO/I-JEPA protocol)."""
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # Normalize features (standard practice)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp(min=1e-6)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    batch_size = 1024
    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(len(train_features), device=device)
        for i in range(0, len(train_features), batch_size):
            idx = perm[i:i + batch_size]
            logits = classifier(train_features[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        classifier.eval()
        with torch.no_grad():
            logits = classifier(test_features)
            acc = (logits.argmax(dim=1) == test_labels).float().mean().item() * 100
            best_acc = max(best_acc, acc)

    return best_acc


# ── k-NN evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def knn_evaluate(train_features, train_labels, test_features, test_labels,
                 k_values=(1, 5, 20), device="cpu"):
    """k-NN classification on frozen features."""
    train_features = F.normalize(train_features, dim=1).to(device)
    test_features = F.normalize(test_features, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    results = {}
    chunk_size = 256

    for k in k_values:
        correct = 0
        for start in range(0, len(test_features), chunk_size):
            end = min(start + chunk_size, len(test_features))
            sim = test_features[start:end] @ train_features.T
            _, topk_idx = sim.topk(k, dim=1)
            topk_labels = train_labels[topk_idx]
            preds = topk_labels.mode(dim=1).values
            correct += (preds == test_labels[start:end]).sum().item()
        results[f"knn_k{k}"] = correct / len(test_features) * 100

    return results


# ── Representation quality metrics ────────────────────────────────────────────

@torch.no_grad()
def representation_quality(features):
    """Compute representation quality metrics."""
    features = F.normalize(features, dim=1)

    # Effective rank (via singular values)
    try:
        _, s, _ = torch.svd(features[:min(2048, len(features))])
        p = s / s.sum()
        eff_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
    except Exception:
        eff_rank = -1.0

    # Uniformity (how evenly distributed on hypersphere)
    n = min(2048, len(features))
    subset = features[:n]
    dists = torch.cdist(subset, subset)
    uniformity = torch.exp(-2 * dists.pow(2)).mean().item()

    # Feature std (collapse detection)
    feat_std = features.std(dim=0).mean().item()

    return {
        "effective_rank": eff_rank,
        "uniformity": uniformity,
        "feature_std": feat_std,
    }


# ── SSL pretraining ──────────────────────────────────────────────────────────

def pretrain(cfg, dataset_name, data_dir, device, use_tcd=False):
    """Pretrain a JEPA model and return the trained encoder."""
    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]

    # Build model
    model = build_tcd_jepa(
        img_size=img_size,
        patch_size=enc_cfg["patch_size"],
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=use_tcd,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    method = "TCD-JEPA" if use_tcd else "Vanilla JEPA"
    logger.info(f"[{method}] {param_count:,} parameters")

    # Build mask collator
    mask_collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=enc_cfg["patch_size"],
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
    )

    # Build dataset
    ssl_dataset = get_ssl_dataset(dataset_name, data_dir, img_size)
    logger.info(f"[{method}] SSL dataset: {len(ssl_dataset)} samples")

    num_workers = train_cfg.get("num_workers", cfg.get("data", {}).get("num_workers", 4))
    dataloader = DataLoader(
        ssl_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mask_collator,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    logger.info(f"[{method}] {num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    optimizer = build_optimizer(
        model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg.get("warmup_epochs", 40) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_sched = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps)

    # Optional TCD loop
    recursive_loop = None
    stream_encoder = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2,
            crystallize_every=5,
            langevin_steps=30,
            device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)
        logger.info(f"[{method}] TCD recursive loop enabled")

    # Logging
    log_dir = cfg.get("logging", {}).get("log_dir", f"./logs/benchmark_{dataset_name}")
    suffix = "_tcd" if use_tcd else "_vanilla"
    run_log_dir = f"{log_dir}{suffix}"
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)

    metric_logger = MetricLogger(
        log_dir=run_log_dir,
        use_wandb=use_wandb,
        wandb_project=cfg.get("logging", {}).get("wandb_project", "tcd-jepa-benchmarks"),
        wandb_config={**cfg, "method": method, "dataset": dataset_name},
    )

    # Mixed precision — bfloat16 doesn't need GradScaler (same dynamic range as fp32)
    use_amp = train_cfg.get("use_bfloat16", False) and device.type == "cuda"
    scaler = None  # GradScaler only needed for float16, not bfloat16

    # Trainer
    checkpoint_dir = f"{run_log_dir}/checkpoints"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule=ema_sched,
        train_loader=dataloader,
        device=device,
        cfg=cfg,
        metric_logger=metric_logger,
        checkpoint_dir=checkpoint_dir,
        scaler=scaler,
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
        use_amp=use_amp,
        amp_dtype=torch.bfloat16,
    )

    logger.info(f"[{method}] Starting pretraining...")
    t0 = time.time()
    trainer.train(num_epochs)
    train_time = time.time() - t0
    logger.info(f"[{method}] Pretraining done in {train_time / 3600:.2f} hours")

    metric_logger.close()
    num_modules = recursive_loop.num_modules if recursive_loop else 0

    # Clean up to free GPU memory before evaluation
    encoder = model.context_encoder
    del trainer, dataloader, optimizer, lr_scheduler, wd_scheduler
    del recursive_loop, stream_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return encoder, train_time, num_modules


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_single_benchmark(dataset_name, cfg, seeds=(42,)):
    """Run benchmark on a single dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = cfg.get("data", {}).get("data_dir", "./data")
    enc_cfg = cfg["model"]["encoder"]
    embed_dim = enc_cfg["embed_dim"]
    img_size = enc_cfg["img_size"]

    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARK: {dataset_name.upper()} | ViT-S/16 | {img_size}px | embed_dim={embed_dim}")
    logger.info(f"Device: {device}")
    logger.info(f"{'='*70}")

    # Get eval datasets
    train_ds, test_ds, num_classes = get_eval_datasets(dataset_name, data_dir, img_size)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    logger.info(f"Eval data: {len(train_ds)} train, {len(test_ds)} test, {num_classes} classes")

    all_results = {"dataset": dataset_name, "num_classes": num_classes,
                   "img_size": img_size, "embed_dim": embed_dim, "vanilla": [], "tcd": []}

    for seed in seeds:
        logger.info(f"\n{'─'*50} SEED {seed} {'─'*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for method_name, use_tcd in [("vanilla", False), ("tcd", True)]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            label = "TCD-JEPA" if use_tcd else "Vanilla JEPA"
            logger.info(f"\n>>> {label} (seed={seed})")

            try:
                # Pretrain
                encoder, train_time, num_modules = pretrain(
                    cfg, dataset_name, data_dir, device, use_tcd=use_tcd)

                # Extract features
                logger.info(f"[{label}] Extracting features...")
                train_feats, train_labels = extract_features(encoder, train_loader, device)
                test_feats, test_labels = extract_features(encoder, test_loader, device)
                logger.info(f"[{label}] Features: train={train_feats.shape}, test={test_feats.shape}")

                # Linear probe
                logger.info(f"[{label}] Linear probe ({num_classes} classes)...")
                lin_acc = linear_probe(
                    train_feats, train_labels, test_feats, test_labels,
                    embed_dim=embed_dim, num_classes=num_classes, device=device)
                logger.info(f"[{label}] Linear probe: {lin_acc:.2f}%")

                # k-NN
                logger.info(f"[{label}] k-NN evaluation...")
                knn_results = knn_evaluate(
                    train_feats, train_labels, test_feats, test_labels, device=device)
                logger.info(f"[{label}] k-NN: {knn_results}")

                # Representation quality
                quality = representation_quality(test_feats)
                logger.info(f"[{label}] Quality: {quality}")

                result = {
                    "seed": seed,
                    "method": method_name,
                    "linear_probe_acc": lin_acc,
                    **knn_results,
                    **quality,
                    "train_time_s": train_time,
                    "train_time_h": train_time / 3600,
                    "num_modules": num_modules,
                    "epochs": cfg["training"]["epochs"],
                }
                all_results[method_name].append(result)

                # Save incremental results
                save_results(all_results, dataset_name)

            except Exception as e:
                logger.error(f"[{label}] FAILED (seed={seed}): {e}", exc_info=True)
            finally:
                # Free GPU memory between runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

    return all_results


def save_results(results, dataset_name):
    """Save results to disk."""
    out_dir = Path(f"results/benchmark_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable report
    report = format_report(results)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)
    print(report)


def format_report(results):
    """Format results into a readable report."""
    lines = []
    ds = results["dataset"].upper()
    lines.append(f"\n{'='*80}")
    lines.append(f"TCD-JEPA BENCHMARK — {ds}")
    lines.append(f"ViT-S/16 | {results['img_size']}px | {results['num_classes']} classes | embed_dim={results['embed_dim']}")
    lines.append(f"{'='*80}")

    for method in ["vanilla", "tcd"]:
        runs = results[method]
        if not runs:
            continue

        name = "Vanilla JEPA" if method == "vanilla" else "TCD-JEPA"
        lin = [r["linear_probe_acc"] for r in runs]
        k1 = [r["knn_k1"] for r in runs]
        k5 = [r["knn_k5"] for r in runs]
        k20 = [r["knn_k20"] for r in runs]
        times = [r["train_time_h"] for r in runs]
        ranks = [r["effective_rank"] for r in runs]
        stds = [r["feature_std"] for r in runs]

        lines.append(f"\n{name} ({runs[0]['epochs']} epochs):")
        lines.append(f"  Linear Probe:    {np.mean(lin):.2f}% +/- {np.std(lin):.2f}%")
        lines.append(f"  k-NN (k=1):      {np.mean(k1):.2f}% +/- {np.std(k1):.2f}%")
        lines.append(f"  k-NN (k=5):      {np.mean(k5):.2f}% +/- {np.std(k5):.2f}%")
        lines.append(f"  k-NN (k=20):     {np.mean(k20):.2f}% +/- {np.std(k20):.2f}%")
        lines.append(f"  Eff. rank:       {np.mean(ranks):.1f}")
        lines.append(f"  Feature std:     {np.mean(stds):.4f}")
        lines.append(f"  Train time:      {np.mean(times):.2f}h")
        if method == "tcd":
            mods = [r["num_modules"] for r in runs]
            lines.append(f"  Modules:         {np.mean(mods):.1f}")

    if results["vanilla"] and results["tcd"]:
        v_lin = np.mean([r["linear_probe_acc"] for r in results["vanilla"]])
        t_lin = np.mean([r["linear_probe_acc"] for r in results["tcd"]])
        v_knn = np.mean([r["knn_k20"] for r in results["vanilla"]])
        t_knn = np.mean([r["knn_k20"] for r in results["tcd"]])
        lines.append(f"\nTCD-JEPA IMPROVEMENT:")
        lines.append(f"  Linear Probe: {t_lin - v_lin:+.2f}% ({(t_lin-v_lin)/max(v_lin,1)*100:+.1f}% relative)")
        lines.append(f"  k-NN (k=20):  {t_knn - v_knn:+.2f}% ({(t_knn-v_knn)/max(v_knn,1)*100:+.1f}% relative)")

    lines.append(f"\n{'─'*80}")
    lines.append("REFERENCE (ImageNet-1k linear probe, ViT-S/16, 300ep):")
    for name, metrics in PUBLISHED_BENCHMARKS.items():
        if "imagenet1k_linear" in metrics:
            lines.append(f"  {name:<35} {metrics['imagenet1k_linear']:.1f}%")
    lines.append(f"{'─'*80}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TCD-JEPA Standard Benchmarks")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "stl10", "imagenet100", "all"],
                        help="Dataset to benchmark on")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config file (auto-detected if not provided)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (5 epochs, 1 seed)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()

    datasets = ["cifar10", "stl10", "imagenet100"] if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        # Load config
        config_path = args.config or f"configs/benchmark_{ds_name}.yaml"
        if not Path(config_path).exists():
            logger.error(f"Config not found: {config_path}")
            continue

        cfg = load_config_with_overrides(config_path, [])

        # Apply overrides
        if args.epochs:
            cfg["training"]["epochs"] = args.epochs
            cfg["training"]["warmup_epochs"] = min(cfg["training"].get("warmup_epochs", 40),
                                                    args.epochs // 4)
        if args.batch_size:
            cfg["training"]["batch_size"] = args.batch_size
        if args.no_wandb:
            cfg["logging"]["use_wandb"] = False
        if args.data_dir:
            cfg["data"]["data_dir"] = args.data_dir

        if args.quick:
            cfg["training"]["epochs"] = 5
            cfg["training"]["warmup_epochs"] = 1
            cfg["training"]["checkpoint_freq"] = 5
            cfg["logging"]["use_wandb"] = False
            seeds = [42]
        else:
            seeds = [42 + i * 81 for i in range(args.seeds)]

        results = run_single_benchmark(ds_name, cfg, seeds=seeds)

    logger.info("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()
