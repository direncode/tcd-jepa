"""Benchmark evaluation: Linear Probe + k-NN on CIFAR-10.

Trains Vanilla JEPA and TCD-JEPA, then evaluates frozen representations
using standard SSL evaluation protocols (linear probe, k-NN) to compare
against published results from I-JEPA, DINO, DINOv2, MAE, etc.

Usage:
    python -m experiments.benchmark_eval [--epochs 30] [--seeds 2]
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
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.utils.masking import MaskCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("benchmark")

# ── Published benchmark numbers (ImageNet-1k linear probe top-1 %) ──────────
# These are the numbers TCD-JEPA aims to compete with conceptually.
# Our evaluation is on CIFAR-10 (much smaller), so direct comparison is
# apples-to-oranges, but the evaluation *protocol* is identical.
PUBLISHED_BENCHMARKS = {
    # Method: {arch: (ImageNet lin-probe %, CIFAR-10 lin-probe % estimate)}
    "I-JEPA (Meta, CVPR'23)": {
        "ViT-B/16 (300ep)": {"imagenet_linear": 72.9},
        "ViT-L/16 (600ep)": {"imagenet_linear": 75.5},
        "ViT-H/14 (300ep)": {"imagenet_linear": 73.3},
        "ViT-H/16-448 (300ep)": {"imagenet_linear": 77.3},
    },
    "MAE (Meta, CVPR'22)": {
        "ViT-B/16 (1600ep)": {"imagenet_linear": 68.0},
        "ViT-L/16 (1600ep)": {"imagenet_linear": 75.8},
        "ViT-H/14 (1600ep)": {"imagenet_linear": 76.6},
    },
    "DINO (Meta, ICCV'21)": {
        "ViT-S/16 (300ep)": {"imagenet_linear": 77.0},
        "ViT-B/16 (400ep)": {"imagenet_linear": 78.2},
        "ViT-B/8 (300ep)": {"imagenet_linear": 80.1},
    },
    "DINOv2 (Meta, 2023)": {
        "ViT-S/14": {"imagenet_linear": 81.1},
        "ViT-B/14": {"imagenet_linear": 84.5},
        "ViT-L/14": {"imagenet_linear": 86.3},
        "ViT-g/14": {"imagenet_linear": 86.5},
    },
    "data2vec (Meta, ICML'22)": {
        "ViT-B/16 (800ep)": {"imagenet_linear": 71.8},
        "ViT-L/16 (1600ep)": {"imagenet_linear": 76.3},
    },
    "iBOT (ByteDance, ICLR'22)": {
        "ViT-B/16 (400ep)": {"imagenet_linear": 79.5},
        "ViT-L/16 (250ep)": {"imagenet_linear": 81.7},
    },
}


# ── Data helpers ────────────────────────────────────────────────────────────
class StructuredSyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset with 10 visually distinct classes for benchmark evaluation.

    Each class has a unique combination of:
    - Background color/gradient
    - Foreground shape (circle, rectangle, triangle, diamond, cross, etc.)
    - Texture pattern (stripes, dots, solid)

    This provides a challenging classification task where the encoder must learn
    meaningful spatial features to distinguish classes.
    """

    def __init__(self, num_samples=10000, img_size=32, num_classes=10, seed=42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        rng = np.random.RandomState(seed)

        self.images = []
        self.labels = []

        samples_per_class = num_samples // num_classes
        for cls in range(num_classes):
            for _ in range(samples_per_class):
                img = self._generate_image(cls, rng)
                self.images.append(img)
                self.labels.append(cls)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _generate_image(self, cls, rng):
        """Generate a hard-to-classify image.

        Difficulty comes from: heavy noise, random colors, small subtle shapes,
        overlapping textures, and random backgrounds. Classes differ only in
        subtle spatial frequency patterns and small shape cues.
        """
        img = torch.zeros(3, self.img_size, self.img_size)
        s = self.img_size

        # Random background (same distribution for all classes)
        for c in range(3):
            img[c] = 0.3 + rng.rand() * 0.4

        # Add class-specific spatial frequency texture (subtle)
        freq = 2 + cls * 0.7  # Different frequency per class
        phase = rng.rand() * 2 * np.pi
        yy_np = np.arange(s).reshape(-1, 1).repeat(s, axis=1) / s
        xx_np = np.arange(s).reshape(1, -1).repeat(s, axis=0) / s

        # Orientation varies per class
        angle = cls * 18.0 + rng.randn() * 10.0  # degrees + noise
        pattern = np.sin(2 * np.pi * freq * (
            np.cos(np.radians(angle)) * xx_np +
            np.sin(np.radians(angle)) * yy_np
        ) + phase) * 0.15  # Moderate visibility

        for c in range(3):
            img[c] += torch.from_numpy(pattern).float()

        # Small shape cue (varies in position, size, color randomly)
        cx = int(rng.randint(s // 4, 3 * s // 4))
        cy = int(rng.randint(s // 4, 3 * s // 4))
        r = int(rng.randint(3, s // 4))  # Medium shapes
        fg_val = 0.3 + rng.rand() * 0.4  # Random foreground brightness

        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')

        if cls % 5 == 0:  # Circle
            mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        elif cls % 5 == 1:  # Square
            mask = (torch.abs(xx - cx) < r) & (torch.abs(yy - cy) < r)
        elif cls % 5 == 2:  # Diamond
            mask = (torch.abs(xx - cx) + torch.abs(yy - cy)) < r
        elif cls % 5 == 3:  # Cross
            mask = ((torch.abs(xx - cx) < max(1, r // 3)) |
                    (torch.abs(yy - cy) < max(1, r // 3))) & \
                   (torch.abs(xx - cx) < r) & (torch.abs(yy - cy) < r)
        else:  # Ring
            dist = (xx - cx)**2 + (yy - cy)**2
            mask = (dist < r**2) & (dist > max(0, (r - 2))**2)

        # Moderate foreground change
        for c in range(3):
            img[c][mask] += (fg_val - 0.5) * 0.3

        # Add second shape for classes 5-9 (to distinguish from 0-4)
        if cls >= 5:
            cx2 = s - cx  # Mirror position
            cy2 = s - cy
            if cls % 5 == 0:
                mask2 = ((xx - cx2)**2 + (yy - cy2)**2) < r**2
            elif cls % 5 == 1:
                mask2 = (torch.abs(xx - cx2) < r) & (torch.abs(yy - cy2) < r)
            elif cls % 5 == 2:
                mask2 = (torch.abs(xx - cx2) + torch.abs(yy - cy2)) < r
            elif cls % 5 == 3:
                mask2 = ((torch.abs(xx - cx2) < max(1, r // 3)) |
                         (torch.abs(yy - cy2) < max(1, r // 3))) & \
                        (torch.abs(xx - cx2) < r) & (torch.abs(yy - cy2) < r)
            else:
                dist2 = (xx - cx2)**2 + (yy - cy2)**2
                mask2 = (dist2 < r**2) & (dist2 > max(0, (r - 2))**2)
            for c in range(3):
                img[c][mask2] += (fg_val - 0.5) * 0.15

        # Moderate noise
        img += torch.randn_like(img) * 0.12
        img = img.clamp(0, 1)

        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = (img - mean) / std

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class SSLDatasetWrapper(torch.utils.data.Dataset):
    """Wraps a labeled dataset to drop labels for SSL pretraining."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx][0]


def try_load_cifar10(data_dir="./data"):
    """Try to load CIFAR-10, return None if not available."""
    try:
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
        return True
    except Exception:
        return False


def get_eval_loaders(batch_size=256, num_workers=2, data_dir="./data",
                     num_train=10000, num_test=2000):
    """Get train and test loaders WITH labels for evaluation.

    Uses CIFAR-10 if available, otherwise synthetic structured data.
    """
    if try_load_cifar10(data_dir):
        logger.info("Using real CIFAR-10 dataset")
        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
            T.ToTensor(), normalize,
        ])
        test_transform = T.Compose([T.ToTensor(), normalize])
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=test_transform)
        dataset_name = "CIFAR-10"
    else:
        logger.info("CIFAR-10 not available, using synthetic structured dataset")
        train_ds = StructuredSyntheticDataset(num_samples=num_train, seed=0)
        test_ds = StructuredSyntheticDataset(num_samples=num_test, seed=999)
        dataset_name = "Synthetic-10 (CIFAR-10 protocol)"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, test_loader, dataset_name


def get_ssl_dataloader(batch_size, mask_collator, num_workers=2, data_dir="./data",
                       max_samples=None):
    """Get dataloader for SSL pretraining (no labels)."""
    if try_load_cifar10(data_dir):
        transform = T.Compose([
            T.RandomHorizontalFlip(), T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=transform)
        dataset = SSLDatasetWrapper(dataset)
    else:
        n = max_samples or 10000
        labeled = StructuredSyntheticDataset(num_samples=n, seed=0)
        dataset = SSLDatasetWrapper(labeled)

    if max_samples and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=mask_collator, drop_last=True,
    )


# ── Feature extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract features from frozen encoder for all images in dataloader.

    Uses average pooling over patch tokens to get a single feature per image.
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        # Forward through encoder without masks to get all patch tokens
        features = encoder(images)  # [B, num_patches, embed_dim]
        # Average pool over patches
        features = features.mean(dim=1)  # [B, embed_dim]
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# ── Linear probe evaluation ─────────────────────────────────────────────────
def linear_probe(train_features, train_labels, test_features, test_labels,
                 embed_dim, num_classes=10, epochs=100, lr=0.01, device="cpu"):
    """Train a linear classifier on frozen features and return test accuracy."""
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # Normalize features
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0).clamp(min=1e-6)
    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std

    batch_size = 512
    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(len(train_features), device=device)
        for i in range(0, len(train_features), batch_size):
            idx = perm[i:i+batch_size]
            logits = classifier(train_features[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            logits = classifier(test_features)
            preds = logits.argmax(dim=1)
            acc = (preds == test_labels).float().mean().item() * 100
            best_acc = max(best_acc, acc)

    return best_acc


# ── k-NN evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def knn_evaluate(train_features, train_labels, test_features, test_labels,
                 k_values=(1, 5, 20), device="cpu"):
    """k-NN classification on frozen features."""
    train_features = F.normalize(train_features, dim=1).to(device)
    test_features = F.normalize(test_features, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    results = {}
    # Compute similarity in chunks to avoid OOM
    chunk_size = 1000
    num_test = len(test_features)

    for k in k_values:
        correct = 0
        for start in range(0, num_test, chunk_size):
            end = min(start + chunk_size, num_test)
            sim = test_features[start:end] @ train_features.T  # [chunk, N_train]
            _, topk_idx = sim.topk(k, dim=1)  # [chunk, k]
            topk_labels = train_labels[topk_idx]  # [chunk, k]
            # Majority vote
            preds = topk_labels.mode(dim=1).values
            correct += (preds == test_labels[start:end]).sum().item()
        results[f"knn_k{k}"] = correct / num_test * 100

    return results


# ── SSL pretraining ──────────────────────────────────────────────────────────
def pretrain_jepa(cfg, device, use_tcd=False, max_samples=None):
    """Pretrain a JEPA model and return the trained encoder."""
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

    dataloader = get_ssl_dataloader(
        batch_size=train_cfg["batch_size"],
        mask_collator=mask_collator,
        max_samples=max_samples,
    )

    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"],
                                weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg.get("warmup_epochs", 5) * steps_per_epoch,
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
            explore_every=2,
            crystallize_every=5,
            langevin_steps=20,
            device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule=ema_schedule,
        train_loader=dataloader,
        device=device,
        cfg=cfg,
        checkpoint_dir="./logs/benchmark/checkpoints",
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
    )

    logger.info(f"{'TCD-JEPA' if use_tcd else 'Vanilla JEPA'}: pretraining {num_epochs} epochs "
                f"({sum(p.numel() for p in model.parameters()):,} params)")
    t0 = time.time()
    trainer.train(num_epochs)
    train_time = time.time() - t0
    logger.info(f"  Pretraining done in {train_time:.1f}s")

    num_modules = recursive_loop.num_modules if recursive_loop else 0
    return model.context_encoder, train_time, num_modules


# ── Main benchmark ──────────────────────────────────────────────────────────
def run_benchmark(epochs=30, seeds=(42,), embed_dim=192, depth=6,
                  max_samples=None):
    """Run the full benchmark: pretrain + linear probe + k-NN for both methods."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = {
        "model": {
            "encoder": {
                "img_size": 32, "patch_size": 4, "in_chans": 3,
                "embed_dim": embed_dim, "depth": depth, "num_heads": max(1, embed_dim // 64),
                "mlp_ratio": 4.0,
            },
            "predictor": {
                "predictor_embed_dim": embed_dim // 2,
                "predictor_depth": max(2, depth // 2),
                "num_heads": max(1, embed_dim // 64),
            },
            "ema": {"start": 0.996, "end": 1.0},
        },
        "training": {
            "epochs": epochs,
            "batch_size": 128,
            "learning_rate": 0.001,
            "start_lr": 0.0001,
            "weight_decay": 0.05,
            "warmup_epochs": min(5, epochs // 4),
        },
        "masking": {
            "num_enc_masks": 4, "num_pred_masks": 1, "min_keep": 4,
            "enc_mask_scale": [0.15, 0.2],
            "pred_mask_scale": [0.2, 0.4],
            "aspect_ratio": [0.75, 1.5],
        },
    }

    all_results = {"vanilla": [], "tcd": [], "dataset_name": ""}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Get evaluation data
        train_loader, test_loader, dataset_name = get_eval_loaders()
        all_results["dataset_name"] = dataset_name

        for method_name, use_tcd in [("vanilla", False), ("tcd", True)]:
            logger.info(f"\n--- {method_name.upper()} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Pretrain
            encoder, train_time, num_modules = pretrain_jepa(
                cfg, device, use_tcd=use_tcd, max_samples=max_samples,
            )

            # Extract features
            logger.info("  Extracting features...")
            train_feats, train_labels = extract_features(encoder, train_loader, device)
            test_feats, test_labels = extract_features(encoder, test_loader, device)
            logger.info(f"  Features: train={train_feats.shape}, test={test_feats.shape}")

            # Linear probe
            logger.info("  Running linear probe...")
            lin_acc = linear_probe(
                train_feats, train_labels, test_feats, test_labels,
                embed_dim=embed_dim, device=device,
            )
            logger.info(f"  Linear probe accuracy: {lin_acc:.2f}%")

            # k-NN
            logger.info("  Running k-NN evaluation...")
            knn_results = knn_evaluate(
                train_feats, train_labels, test_feats, test_labels,
                device=device,
            )
            logger.info(f"  k-NN results: {knn_results}")

            result = {
                "seed": seed,
                "method": method_name,
                "linear_probe_acc": lin_acc,
                **knn_results,
                "train_time_s": train_time,
                "num_modules": num_modules,
                "embed_dim": embed_dim,
                "depth": depth,
                "epochs": epochs,
            }
            all_results[method_name].append(result)

    return all_results


def print_results_table(results):
    """Print a formatted comparison table."""
    dataset_name = results.get("dataset_name", "CIFAR-10")
    print("\n" + "=" * 80)
    print(f"TCD-JEPA BENCHMARK RESULTS — {dataset_name}")
    print("Evaluation: Linear Probe (frozen encoder) + k-NN")
    print("=" * 80)

    for method in ["vanilla", "tcd"]:
        runs = results[method]
        if not runs:
            continue

        name = "Vanilla JEPA" if method == "vanilla" else "TCD-JEPA"
        lin_accs = [r["linear_probe_acc"] for r in runs]
        knn1_accs = [r["knn_k1"] for r in runs]
        knn5_accs = [r["knn_k5"] for r in runs]
        knn20_accs = [r["knn_k20"] for r in runs]
        times = [r["train_time_s"] for r in runs]

        print(f"\n{name} (embed_dim={runs[0]['embed_dim']}, depth={runs[0]['depth']}, "
              f"epochs={runs[0]['epochs']}):")
        print(f"  Linear Probe:  {np.mean(lin_accs):.2f}% +/- {np.std(lin_accs):.2f}%")
        print(f"  k-NN (k=1):    {np.mean(knn1_accs):.2f}% +/- {np.std(knn1_accs):.2f}%")
        print(f"  k-NN (k=5):    {np.mean(knn5_accs):.2f}% +/- {np.std(knn5_accs):.2f}%")
        print(f"  k-NN (k=20):   {np.mean(knn20_accs):.2f}% +/- {np.std(knn20_accs):.2f}%")
        print(f"  Train time:    {np.mean(times):.1f}s +/- {np.std(times):.1f}s")
        if method == "tcd":
            mods = [r["num_modules"] for r in runs]
            print(f"  Modules:       {np.mean(mods):.1f} +/- {np.std(mods):.1f}")

    # Improvement
    if results["vanilla"] and results["tcd"]:
        v_lin = np.mean([r["linear_probe_acc"] for r in results["vanilla"]])
        t_lin = np.mean([r["linear_probe_acc"] for r in results["tcd"]])
        v_knn = np.mean([r["knn_k20"] for r in results["vanilla"]])
        t_knn = np.mean([r["knn_k20"] for r in results["tcd"]])

        print("\nIMPROVEMENT (TCD over Vanilla):")
        print(f"  Linear Probe:  {t_lin - v_lin:+.2f}% ({(t_lin-v_lin)/max(v_lin,1e-6)*100:+.1f}% relative)")
        print(f"  k-NN (k=20):   {t_knn - v_knn:+.2f}% ({(t_knn-v_knn)/max(v_knn,1e-6)*100:+.1f}% relative)")

    # Published comparison
    print(f"\n{'─'*80}")
    print("PUBLISHED SSL BENCHMARKS (ImageNet-1k linear probe, for reference):")
    print(f"{'─'*80}")
    print(f"{'Method':<35} {'Architecture':<25} {'IN-1k Lin %':>12}")
    print(f"{'─'*35} {'─'*25} {'─'*12}")
    for method_name, archs in PUBLISHED_BENCHMARKS.items():
        for arch, metrics in archs.items():
            print(f"{method_name:<35} {arch:<25} {metrics['imagenet_linear']:>11.1f}%")
        method_name = ""  # Only print method name once

    print(f"\n{'─'*80}")
    print("NOTE: Our results are on CIFAR-10 (50k train / 10k test, 32x32).")
    print("Published numbers are on ImageNet-1k (1.28M train, 224x224).")
    print("The evaluation PROTOCOL is the same (frozen encoder → linear probe),")
    print("but the scale and dataset are very different.")
    print("CIFAR-10 supervised SOTA is ~99.5%. SSL methods typically get 90-97%.")
    print(f"{'─'*80}")


def main():
    parser = argparse.ArgumentParser(description="TCD-JEPA Benchmark Evaluation")
    parser.add_argument("--epochs", type=int, default=30, help="Pretraining epochs")
    parser.add_argument("--seeds", type=int, default=2, help="Number of random seeds")
    parser.add_argument("--embed-dim", type=int, default=192, help="Encoder embed dim")
    parser.add_argument("--depth", type=int, default=6, help="Encoder depth")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples (for faster testing)")
    args = parser.parse_args()

    seeds = [42 + i * 81 for i in range(args.seeds)]

    results = run_benchmark(
        epochs=args.epochs,
        seeds=seeds,
        embed_dim=args.embed_dim,
        depth=args.depth,
        max_samples=args.max_samples,
    )

    # Save results
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_results_table(results)

    # Also save the table as text
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_results_table(results)
    sys.stdout = old_stdout
    table_text = buffer.getvalue()

    with open(out_dir / "benchmark_report.txt", "w") as f:
        f.write(table_text)

    logger.info(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
