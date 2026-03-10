"""Image experiment runner (CIFAR-10).

Runs JEPA and TCD-JEPA on CIFAR-10 for comparison.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.visualization import plot_loss_curves, plot_module_formation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_experiment")


def run_experiment(config_path: str, overrides: list[str] = None) -> dict:
    """Run CIFAR-10 experiment comparing vanilla JEPA and TCD-JEPA."""
    cfg = load_config_with_overrides(config_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]

    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    # CIFAR-10
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

    # Wrap dataset to drop labels
    class UnlabeledWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return self.ds[idx][0]  # drop label

    dataloader = DataLoader(
        UnlabeledWrapper(dataset),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("data", {}).get("num_workers", 2),
        collate_fn=mask_collator,
        drop_last=True,
    )

    def build_model(use_dynamic=False):
        return build_tcd_jepa(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            predictor_embed_dim=pred_cfg["predictor_embed_dim"],
            predictor_depth=pred_cfg["predictor_depth"],
            predictor_num_heads=pred_cfg["num_heads"],
            use_dynamic_predictor=use_dynamic,
        ).to(device)

    # Vanilla JEPA
    logger.info("=== Vanilla JEPA ===")
    vanilla_losses = _train(build_model(), dataloader, train_cfg, device)

    # TCD-JEPA
    logger.info("=== TCD-JEPA ===")
    tcd_model = build_model(use_dynamic=True)
    recursive_loop = RecursiveLoop(
        embed_dim=enc_cfg["embed_dim"],
        explore_every=2,
        crystallize_every=5,
        langevin_steps=20,
        device=device,
    )
    # Share crystallizer registry with dynamic predictor
    tcd_model.set_module_registry(recursive_loop.crystallizer.registry)
    tcd_losses, module_counts, conv_scores = _train_tcd(
        tcd_model, dataloader, train_cfg, device, recursive_loop
    )

    # Save
    output_dir = Path(cfg.get("logging", {}).get("log_dir", "./logs")) / "cifar10"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(
        {"Vanilla JEPA": vanilla_losses, "TCD-JEPA": tcd_losses},
        str(output_dir / "loss_comparison.png"),
    )

    if module_counts:
        plot_module_formation(module_counts, conv_scores, str(output_dir / "module_formation.png"))

    results = {
        "vanilla_final_loss": vanilla_losses[-1] if vanilla_losses else None,
        "tcd_final_loss": tcd_losses[-1] if tcd_losses else None,
        "num_modules": module_counts[-1] if module_counts else 0,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _make_schedulers(model, train_cfg, total_steps):
    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    steps_per_epoch = total_steps // train_cfg["epochs"]
    lr_sched = WarmupCosineSchedule(
        optimizer, warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4), ref_lr=train_cfg["learning_rate"], T_max=total_steps,
    )
    wd_sched = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_sched = momentum_schedule(
        train_cfg.get("ema_start", 0.996), train_cfg.get("ema_end", 1.0), total_steps
    )
    return optimizer, lr_sched, wd_sched, ema_sched


def _train(model, dataloader, train_cfg, device) -> list[float]:
    num_epochs = train_cfg["epochs"]
    total_steps = num_epochs * len(dataloader)
    optimizer, lr_sched, wd_sched, ema_sched = _make_schedulers(model, train_cfg, total_steps)

    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
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
        logger.info(f"  Epoch {epoch}: loss={avg:.4f}")
    return epoch_losses


def _train_tcd(model, dataloader, train_cfg, device, recursive_loop) -> tuple:
    num_epochs = train_cfg["epochs"]
    total_steps = num_epochs * len(dataloader)
    optimizer, lr_sched, wd_sched, ema_sched = _make_schedulers(model, train_cfg, total_steps)
    stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)

    model.train()
    epoch_losses, module_counts, conv_scores = [], [], []
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

        # Recursive loop
        if last_images is not None:
            with torch.no_grad():
                z = model.context_encoder(last_images)
                t = model.target_encoder(last_images)
            energy_fn = stream_encoder.make_energy_fn(t)
            result = recursive_loop.step(z, energy_fn, epoch=epoch)
            module_counts.append(recursive_loop.num_modules)
            if "convergence" in result:
                conv_scores.append(result["convergence"]["convergence_score"])

        logger.info(f"  Epoch {epoch}: loss={avg:.4f} modules={recursive_loop.num_modules}")
    return epoch_losses, module_counts, conv_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_scale.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_experiment(args.config, args.overrides)
