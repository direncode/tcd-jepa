"""Ablation study runner for TCD-JEPA.

Systematically compares:
1. Vanilla JEPA (baseline)
2. JEPA + System 2 only (exploration, no crystallization)
3. JEPA + System 3 only (crystallization with random trajectories)
4. Full TCD-JEPA (Systems 1+2+3 with recursive loop)

Produces comparison tables and plots.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.visualization import plot_loss_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ablation")


ABLATION_CONFIGS = {
    "vanilla": {
        "use_dynamic_predictor": False,
        "use_recursive_loop": False,
        "description": "Vanilla JEPA (baseline)",
    },
    "system2_only": {
        "use_dynamic_predictor": False,
        "use_recursive_loop": True,
        "crystallize_every": 999999,  # effectively never crystallize
        "description": "JEPA + System 2 (exploration only)",
    },
    "full_tcd": {
        "use_dynamic_predictor": True,
        "use_recursive_loop": True,
        "description": "Full TCD-JEPA (Systems 1+2+3)",
    },
}


def _build_dataloader(cfg, device):
    """Build CIFAR-10 dataloader for ablation."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    enc_cfg = cfg["model"]["encoder"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

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
        DropLabel(dataset),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("data", {}).get("num_workers", 2),
        collate_fn=mask_collator,
        drop_last=True,
    )
    return dataloader


def _run_single_ablation(name, ablation_cfg, cfg, dataloader, device):
    """Run a single ablation experiment."""
    logger.info(f"=== {ablation_cfg['description']} ===")

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]

    model = build_tcd_jepa(
        img_size=enc_cfg["img_size"],
        patch_size=enc_cfg["patch_size"],
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=ablation_cfg["use_dynamic_predictor"],
    ).to(device)

    recursive_loop = None
    stream_encoder = None
    if ablation_cfg["use_recursive_loop"]:
        crystallize_every = ablation_cfg.get("crystallize_every", 10)
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2,
            crystallize_every=crystallize_every,
            langevin_steps=20,
            device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        if ablation_cfg["use_dynamic_predictor"]:
            model.set_module_registry(recursive_loop.crystallizer.registry)

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
        train_cfg.get("ema_start", cfg["model"]["ema"]["start"]),
        train_cfg.get("ema_end", cfg["model"]["ema"]["end"]),
        total_steps,
    )

    model.train()
    epoch_losses = []
    module_counts = []
    known_module_ids: set[str] = set()

    for epoch in range(num_epochs):
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

        # Recursive loop step
        n_mods = 0
        if recursive_loop is not None and last_images is not None:
            with torch.no_grad():
                z = model.context_encoder(last_images)
                t = model.target_encoder(last_images)
            energy_fn = stream_encoder.make_energy_fn(t)
            loop_result = recursive_loop.step(z, energy_fn, epoch=epoch)
            n_mods = recursive_loop.num_modules

            # Add new module params to optimizer
            if loop_result.get("crystallized"):
                registry = recursive_loop.crystallizer.registry
                for mid, mod in registry.get_all_modules():
                    if mid not in known_module_ids:
                        known_module_ids.add(mid)
                        params = list(mod.parameters())
                        if params:
                            mod.to(device)
                            optimizer.add_param_group({"params": params, "lr": optimizer.param_groups[0]["lr"], "weight_decay": 0.0})

        module_counts.append(n_mods)
        logger.info(f"  [{name}] Epoch {epoch}: loss={avg:.4f} modules={n_mods}")

    return {
        "losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "module_counts": module_counts,
        "description": ablation_cfg["description"],
    }


def run_ablation(config_path: str, overrides: list[str] = None) -> dict:
    """Run full ablation study."""
    cfg = load_config_with_overrides(config_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataloader = _build_dataloader(cfg, device)
    logger.info(f"Ablation on {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    results = {}
    for name, ablation_cfg in ABLATION_CONFIGS.items():
        # Reset seed for fair comparison
        torch.manual_seed(seed)
        np.random.seed(seed)
        results[name] = _run_single_ablation(name, ablation_cfg, cfg, dataloader, device)

    # Save results
    output_dir = Path(cfg.get("logging", {}).get("log_dir", "./logs")) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss comparison plot
    loss_dict = {r["description"]: r["losses"] for r in results.values()}
    plot_loss_curves(loss_dict, str(output_dir / "ablation_loss_curves.png"),
                     title="Ablation: Loss Comparison")

    # Summary table
    summary = _format_summary(results)
    logger.info("\n" + summary)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "losses"}
                   for k, v in results.items()}, f, indent=2)

    with open(output_dir / "ablation_summary.txt", "w") as f:
        f.write(summary)

    logger.info(f"Ablation results saved to {output_dir}")
    return results


def _format_summary(results: dict) -> str:
    """Format a text comparison table."""
    lines = [
        "=" * 60,
        "ABLATION STUDY RESULTS",
        "=" * 60,
        f"{'Configuration':<35} {'Final Loss':>10} {'Modules':>8}",
        "-" * 60,
    ]
    for name, res in results.items():
        final = res["final_loss"]
        mods = res["module_counts"][-1] if res["module_counts"] else 0
        lines.append(f"{res['description']:<35} {final:>10.4f} {mods:>8}")

    # Compute improvements relative to vanilla
    vanilla_loss = results.get("vanilla", {}).get("final_loss", float("inf"))
    if vanilla_loss and vanilla_loss > 0:
        lines.append("-" * 60)
        for name, res in results.items():
            if name == "vanilla":
                continue
            imp = (vanilla_loss - res["final_loss"]) / vanilla_loss * 100
            lines.append(f"{res['description']:<35} improvement: {imp:>+.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TCD-JEPA ablation study")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_ablation(args.config, args.overrides)
