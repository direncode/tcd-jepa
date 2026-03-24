"""Comprehensive experiment runner for TCD-JEPA evaluation.

Runs multi-seed experiments with proper statistics across:
1. CIFAR-10 (image prediction)
2. Two Rooms (structured environment)

Reports mean +/- std, convergence speed, module formation dynamics,
and per-epoch loss comparisons.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.utils.masking import MaskCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive")


class SyntheticImageDataset(torch.utils.data.Dataset):
    """Synthetic structured image dataset for testing without CIFAR-10.

    Generates images with geometric patterns (circles, rectangles, gradients)
    that have meaningful spatial structure for JEPA to learn.
    """

    def __init__(self, num_samples: int = 5000, img_size: int = 32, num_classes: int = 10, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        rng = np.random.RandomState(seed)

        self.images = torch.zeros(num_samples, 3, img_size, img_size)
        for i in range(num_samples):
            cls = i % num_classes
            img = self._generate_structured_image(cls, img_size, rng)
            self.images[i] = img

    def _generate_structured_image(self, cls, size, rng):
        """Generate a structured image based on class."""
        img = torch.zeros(3, size, size)
        # Background gradient based on class
        bg_color = torch.tensor([
            (cls * 37 % 256) / 255.0,
            (cls * 73 % 256) / 255.0,
            (cls * 113 % 256) / 255.0,
        ])
        for c in range(3):
            img[c] = bg_color[c] * 0.3

        # Add geometric shapes
        cx, cy = rng.randint(size // 4, 3 * size // 4, size=2)
        radius = rng.randint(size // 8, size // 4)
        color = torch.tensor([rng.random(), rng.random(), rng.random()])

        y_coords, x_coords = torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing="ij"
        )

        if cls % 3 == 0:
            # Circle
            mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < radius ** 2
        elif cls % 3 == 1:
            # Rectangle
            mask = (x_coords >= cx - radius) & (x_coords <= cx + radius) & \
                   (y_coords >= cy - radius) & (y_coords <= cy + radius)
        else:
            # Triangle-ish (diamond)
            mask = (torch.abs(x_coords - cx) + torch.abs(y_coords - cy)) < radius

        for c in range(3):
            img[c][mask] = color[c]

        # Add noise
        img += torch.randn_like(img) * 0.05
        img = img.clamp(0, 1)

        # Normalize like CIFAR
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = (img - mean) / std
        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx]


def _build_cifar10_dataloader(cfg):
    """Build image dataloader (synthetic if CIFAR-10 not available)."""
    enc_cfg = cfg["model"]["encoder"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    # Try CIFAR-10 first, fall back to synthetic
    try:
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize(img_size) if img_size != 32 else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=cfg.get("data", {}).get("data_dir", "./data"),
            train=True, download=True, transform=transform,
        )

        class DropLabel(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                return self.ds[idx][0]

        dataset = DropLabel(dataset)
        logger.info("Using CIFAR-10 dataset")
    except Exception:
        dataset = SyntheticImageDataset(num_samples=500, img_size=img_size)
        logger.info("CIFAR-10 unavailable, using synthetic image dataset (500 samples)")

    mask_collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=patch_size,
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=mask_collator,
        drop_last=True,
    )
    return dataloader


def _build_two_rooms_dataloader(cfg):
    """Build Two Rooms dataloader."""
    from experiments.two_rooms.environment import TwoRoomsDataset

    enc_cfg = cfg["model"]["encoder"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    dataset = TwoRoomsDataset(
        num_episodes=30,
        episode_length=20,
        render_size=img_size,
    )

    mask_collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=patch_size,
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=mask_collator,
        drop_last=True,
    )
    return dataloader


def _train_vanilla(model, dataloader, train_cfg, device, cfg) -> dict:
    """Train vanilla JEPA, return detailed metrics."""
    num_epochs = train_cfg["epochs"]
    total_steps = num_epochs * len(dataloader)
    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_sched = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * len(dataloader),
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_sched = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_sched = momentum_schedule(
        cfg["model"]["ema"]["start"],
        cfg["model"]["ema"]["end"],
        total_steps,
    )

    model.train()
    epoch_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        total, count = 0.0, 0
        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            loss = model(images, masks_enc, masks_pred)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()
            wd_sched.step()
            model.update_target_encoder(next(ema_sched))
            total += loss.item()
            count += 1

        avg = total / max(count, 1)
        epoch_losses.append(avg)
        epoch_times.append(time.time() - t0)
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            logger.info(f"    [vanilla] Epoch {epoch}: loss={avg:.6f} ({time.time()-t0:.1f}s)")

    return {
        "losses": epoch_losses,
        "final_loss": epoch_losses[-1],
        "best_loss": min(epoch_losses),
        "best_epoch": int(np.argmin(epoch_losses)),
        "avg_epoch_time": float(np.mean(epoch_times)),
    }


def _train_tcd(model, dataloader, train_cfg, device, cfg, recursive_loop) -> dict:
    """Train TCD-JEPA, return detailed metrics."""
    num_epochs = train_cfg["epochs"]
    total_steps = num_epochs * len(dataloader)
    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_sched = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * len(dataloader),
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_sched = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_sched = momentum_schedule(
        cfg["model"]["ema"]["start"],
        cfg["model"]["ema"]["end"],
        total_steps,
    )

    stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)

    model.train()
    epoch_losses = []
    epoch_times = []
    module_counts = []
    convergence_scores = []
    known_module_ids: set[str] = set()

    for epoch in range(num_epochs):
        t0 = time.time()
        total, count = 0.0, 0
        last_images = None
        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            loss = model(images, masks_enc, masks_pred)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()
            wd_sched.step()
            model.update_target_encoder(next(ema_sched))
            total += loss.item()
            count += 1
            last_images = images

        avg = total / max(count, 1)
        epoch_losses.append(avg)
        epoch_times.append(time.time() - t0)

        # Recursive loop
        if last_images is not None:
            with torch.no_grad():
                z = model.context_encoder(last_images)
                t = model.target_encoder(last_images)
            energy_fn = stream_encoder.make_energy_fn(t)
            loop_result = recursive_loop.step(z, energy_fn, epoch=epoch)

            if loop_result.get("crystallized"):
                registry = recursive_loop.crystallizer.registry
                for mid, mod in registry.get_all_modules():
                    if mid not in known_module_ids:
                        known_module_ids.add(mid)
                        params = list(mod.parameters())
                        if params:
                            mod.to(device)
                            optimizer.add_param_group({
                                "params": params,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_decay": 0.0,
                            })

            module_counts.append(recursive_loop.num_modules)
            if "convergence" in loop_result:
                convergence_scores.append(loop_result["convergence"]["convergence_score"])

        if epoch % 3 == 0 or epoch == num_epochs - 1:
            mods = recursive_loop.num_modules
            logger.info(f"    [tcd] Epoch {epoch}: loss={avg:.6f} modules={mods} ({time.time()-t0:.1f}s)")

    # Compute convergence speed: epoch where TCD loss first drops below vanilla's final
    return {
        "losses": epoch_losses,
        "final_loss": epoch_losses[-1],
        "best_loss": min(epoch_losses),
        "best_epoch": int(np.argmin(epoch_losses)),
        "avg_epoch_time": float(np.mean(epoch_times)),
        "module_counts": module_counts,
        "final_modules": module_counts[-1] if module_counts else 0,
        "convergence_scores": convergence_scores,
        "converged": recursive_loop.is_converged,
    }


def run_single_experiment(cfg, dataset_type, seed, device) -> dict:
    """Run a single experiment (vanilla + TCD) with one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]

    if dataset_type == "cifar10":
        dataloader = _build_cifar10_dataloader(cfg)
    else:
        dataloader = _build_two_rooms_dataloader(cfg)

    def build_model(use_dynamic=False):
        return build_tcd_jepa(
            img_size=enc_cfg["img_size"],
            patch_size=enc_cfg["patch_size"],
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            predictor_embed_dim=pred_cfg["predictor_embed_dim"],
            predictor_depth=pred_cfg["predictor_depth"],
            predictor_num_heads=pred_cfg["num_heads"],
            use_dynamic_predictor=use_dynamic,
        ).to(device)

    # Vanilla
    logger.info(f"  [seed={seed}] Training Vanilla JEPA...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    vanilla_model = build_model(use_dynamic=False)
    vanilla_results = _train_vanilla(vanilla_model, dataloader, train_cfg, device, cfg)

    # TCD-JEPA
    logger.info(f"  [seed={seed}] Training TCD-JEPA...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tcd_model = build_model(use_dynamic=True)
    recursive_loop = RecursiveLoop(
        embed_dim=enc_cfg["embed_dim"],
        explore_every=2,
        crystallize_every=5,
        langevin_steps=20,
        device=device,
    )
    tcd_model.set_module_registry(recursive_loop.crystallizer.registry)
    tcd_results = _train_tcd(tcd_model, dataloader, train_cfg, device, cfg, recursive_loop)

    # Compute relative improvement
    v_final = vanilla_results["final_loss"]
    t_final = tcd_results["final_loss"]
    improvement = (v_final - t_final) / v_final * 100 if v_final > 0 else 0.0

    # Best-loss improvement
    v_best = vanilla_results["best_loss"]
    t_best = tcd_results["best_loss"]
    best_improvement = (v_best - t_best) / v_best * 100 if v_best > 0 else 0.0

    # Convergence speed: first epoch where TCD < vanilla final
    convergence_epoch = None
    for i, loss in enumerate(tcd_results["losses"]):
        if loss < v_final:
            convergence_epoch = i
            break

    return {
        "seed": seed,
        "vanilla": vanilla_results,
        "tcd": tcd_results,
        "final_loss_improvement_pct": improvement,
        "best_loss_improvement_pct": best_improvement,
        "convergence_epoch": convergence_epoch,
    }


def run_multi_seed(
    config_path: str,
    dataset_type: str,
    seeds: list[int],
    num_epochs: int,
    overrides: list[str] = None,
) -> dict:
    """Run multi-seed experiment and compute statistics."""
    cfg = load_config_with_overrides(config_path, overrides)
    cfg["training"]["epochs"] = num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running {dataset_type} experiment: {len(seeds)} seeds, {num_epochs} epochs on {device}")

    all_results = []
    for seed in seeds:
        result = run_single_experiment(cfg, dataset_type, seed, device)
        all_results.append(result)
        logger.info(
            f"  [seed={seed}] Done: vanilla={result['vanilla']['final_loss']:.6f} "
            f"tcd={result['tcd']['final_loss']:.6f} "
            f"improvement={result['final_loss_improvement_pct']:+.1f}%"
        )

    # Aggregate statistics
    improvements = [r["final_loss_improvement_pct"] for r in all_results]
    best_improvements = [r["best_loss_improvement_pct"] for r in all_results]
    vanilla_finals = [r["vanilla"]["final_loss"] for r in all_results]
    tcd_finals = [r["tcd"]["final_loss"] for r in all_results]
    final_modules = [r["tcd"]["final_modules"] for r in all_results]
    conv_epochs = [r["convergence_epoch"] for r in all_results if r["convergence_epoch"] is not None]

    stats = {
        "dataset": dataset_type,
        "num_seeds": len(seeds),
        "num_epochs": num_epochs,
        "seeds": seeds,
        "vanilla_final_loss": {
            "mean": float(np.mean(vanilla_finals)),
            "std": float(np.std(vanilla_finals)),
            "min": float(np.min(vanilla_finals)),
            "max": float(np.max(vanilla_finals)),
        },
        "tcd_final_loss": {
            "mean": float(np.mean(tcd_finals)),
            "std": float(np.std(tcd_finals)),
            "min": float(np.min(tcd_finals)),
            "max": float(np.max(tcd_finals)),
        },
        "final_loss_improvement_pct": {
            "mean": float(np.mean(improvements)),
            "std": float(np.std(improvements)),
            "min": float(np.min(improvements)),
            "max": float(np.max(improvements)),
            "all": improvements,
        },
        "best_loss_improvement_pct": {
            "mean": float(np.mean(best_improvements)),
            "std": float(np.std(best_improvements)),
            "all": best_improvements,
        },
        "final_modules": {
            "mean": float(np.mean(final_modules)),
            "std": float(np.std(final_modules)),
            "all": final_modules,
        },
        "convergence_epoch": {
            "mean": float(np.mean(conv_epochs)) if conv_epochs else None,
            "std": float(np.std(conv_epochs)) if conv_epochs else None,
            "count": len(conv_epochs),
        },
        "per_seed_results": all_results,
    }

    # Per-epoch averaged loss curves
    vanilla_curves = np.array([r["vanilla"]["losses"] for r in all_results])
    tcd_curves = np.array([r["tcd"]["losses"] for r in all_results])
    stats["avg_vanilla_curve"] = vanilla_curves.mean(axis=0).tolist()
    stats["avg_tcd_curve"] = tcd_curves.mean(axis=0).tolist()
    stats["std_vanilla_curve"] = vanilla_curves.std(axis=0).tolist()
    stats["std_tcd_curve"] = tcd_curves.std(axis=0).tolist()

    return stats


def format_report(stats: dict) -> str:
    """Format a human-readable report."""
    lines = [
        "=" * 70,
        f"COMPREHENSIVE EXPERIMENT REPORT: {stats['dataset'].upper()}",
        f"Seeds: {stats['num_seeds']} | Epochs: {stats['num_epochs']}",
        "=" * 70,
        "",
        "FINAL LOSS:",
        f"  Vanilla JEPA:  {stats['vanilla_final_loss']['mean']:.6f} +/- {stats['vanilla_final_loss']['std']:.6f}",
        f"  TCD-JEPA:      {stats['tcd_final_loss']['mean']:.6f} +/- {stats['tcd_final_loss']['std']:.6f}",
        "",
        "IMPROVEMENT (final loss):",
        f"  Mean: {stats['final_loss_improvement_pct']['mean']:+.1f}% +/- {stats['final_loss_improvement_pct']['std']:.1f}%",
        f"  Range: [{stats['final_loss_improvement_pct']['min']:+.1f}%, {stats['final_loss_improvement_pct']['max']:+.1f}%]",
        f"  Per-seed: {[f'{x:+.1f}%' for x in stats['final_loss_improvement_pct']['all']]}",
        "",
        "IMPROVEMENT (best loss across training):",
        f"  Mean: {stats['best_loss_improvement_pct']['mean']:+.1f}% +/- {stats['best_loss_improvement_pct']['std']:.1f}%",
        f"  Per-seed: {[f'{x:+.1f}%' for x in stats['best_loss_improvement_pct']['all']]}",
        "",
        "MODULE FORMATION:",
        f"  Final modules: {stats['final_modules']['mean']:.1f} +/- {stats['final_modules']['std']:.1f}",
        f"  Per-seed: {stats['final_modules']['all']}",
        "",
    ]

    if stats["convergence_epoch"]["count"] > 0:
        lines.extend([
            "CONVERGENCE SPEED (epoch where TCD < vanilla final):",
            f"  Mean: {stats['convergence_epoch']['mean']:.1f} +/- {stats['convergence_epoch']['std']:.1f}",
            f"  ({stats['convergence_epoch']['count']}/{stats['num_seeds']} seeds converged faster)",
            "",
        ])

    # Epoch-by-epoch improvement trajectory
    avg_v = np.array(stats["avg_vanilla_curve"])
    avg_t = np.array(stats["avg_tcd_curve"])
    imp_curve = (avg_v - avg_t) / (avg_v + 1e-10) * 100
    milestones = [0, len(imp_curve)//4, len(imp_curve)//2, 3*len(imp_curve)//4, len(imp_curve)-1]
    lines.append("IMPROVEMENT TRAJECTORY (epoch: improvement):")
    for m in milestones:
        lines.append(f"  Epoch {m:3d}: {imp_curve[m]:+.1f}%  (vanilla={avg_v[m]:.6f}, tcd={avg_t[m]:.6f})")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive TCD-JEPA evaluation")
    parser.add_argument("--dataset", choices=["cifar10", "two_rooms"], required=True)
    parser.add_argument("--config", default=None, help="Config path (auto-selected if not given)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default="./results/comprehensive")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.config is None:
        args.config = "configs/small_scale.yaml" if args.dataset == "cifar10" else "configs/two_rooms.yaml"

    stats = run_multi_seed(args.config, args.dataset, args.seeds, args.epochs, args.overrides)

    report = format_report(stats)
    logger.info("\n" + report)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "stats.json", "w") as f:
        # Remove per_seed_results losses for compact JSON
        compact = {k: v for k, v in stats.items() if k != "per_seed_results"}
        compact["per_seed_summary"] = [
            {
                "seed": r["seed"],
                "vanilla_final": r["vanilla"]["final_loss"],
                "tcd_final": r["tcd"]["final_loss"],
                "improvement": r["final_loss_improvement_pct"],
                "modules": r["tcd"]["final_modules"],
            }
            for r in stats["per_seed_results"]
        ]
        json.dump(compact, f, indent=2)

    with open(output_dir / "report.txt", "w") as f:
        f.write(report)

    # Also save full per-seed loss curves
    with open(output_dir / "loss_curves.json", "w") as f:
        curves = {
            f"vanilla_seed{r['seed']}": r["vanilla"]["losses"]
            for r in stats["per_seed_results"]
        }
        curves.update({
            f"tcd_seed{r['seed']}": r["tcd"]["losses"]
            for r in stats["per_seed_results"]
        })
        json.dump(curves, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
