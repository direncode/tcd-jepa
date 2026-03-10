"""Two Rooms experiment: TCD-JEPA vs vanilla JEPA.

Demonstrates module formation and capability discovery in the
Two Rooms gridworld environment.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.logging import MetricLogger
from tcd_jepa.utils.visualization import plot_loss_curves, plot_module_formation
from experiments.two_rooms.environment import TwoRoomsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("two_rooms")


def run_experiment(config_path: str, overrides: list[str] = None) -> dict:
    """Run the Two Rooms experiment.

    Args:
        config_path: Path to YAML config.
        overrides: CLI overrides.

    Returns:
        Results dict with loss histories and module formation data.
    """
    cfg = load_config_with_overrides(config_path, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]

    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    # Build dataset
    dataset = TwoRoomsDataset(
        num_episodes=200,
        episode_length=50,
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

    # ---- Run 1: Vanilla JEPA ----
    logger.info("=== Vanilla JEPA ===")
    vanilla_model = build_tcd_jepa(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
    ).to(device)

    vanilla_losses = _train_model(vanilla_model, dataloader, train_cfg, device)

    # ---- Run 2: TCD-JEPA ----
    logger.info("=== TCD-JEPA ===")
    tcd_model = build_tcd_jepa(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=True,
    ).to(device)

    recursive_loop = RecursiveLoop(
        embed_dim=enc_cfg["embed_dim"],
        explore_every=2,
        crystallize_every=5,
        langevin_steps=20,
        device=device,
    )

    # Share crystallizer registry with dynamic predictor
    tcd_model.set_module_registry(recursive_loop.crystallizer.registry)

    tcd_losses, module_counts, convergence_scores = _train_model_tcd(
        tcd_model, dataloader, train_cfg, device, recursive_loop
    )

    # Save results
    output_dir = Path(cfg.get("logging", {}).get("log_dir", "./logs")) / "two_rooms"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(
        {"Vanilla JEPA": vanilla_losses, "TCD-JEPA": tcd_losses},
        str(output_dir / "loss_comparison.png"),
        title="Two Rooms: Vanilla JEPA vs TCD-JEPA",
    )

    if module_counts:
        plot_module_formation(
            module_counts, convergence_scores,
            str(output_dir / "module_formation.png"),
        )

    results = {
        "vanilla_final_loss": vanilla_losses[-1] if vanilla_losses else None,
        "tcd_final_loss": tcd_losses[-1] if tcd_losses else None,
        "num_modules_final": module_counts[-1] if module_counts else 0,
        "converged": recursive_loop.is_converged,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return results


def _train_model(model, dataloader, train_cfg, device) -> list[float]:
    """Train vanilla JEPA and return per-epoch losses."""
    num_epochs = train_cfg["epochs"]
    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_schedule = momentum_schedule(
        train_cfg.get("ema_start", 0.996),
        train_cfg.get("ema_end", 1.0),
        total_steps,
    )

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            result = model(images, masks_enc, masks_pred)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            wd_scheduler.step()
            m = next(ema_schedule)
            model.update_target_encoder(m)

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        epoch_losses.append(avg_loss)
        logger.info(f"  Epoch {epoch}: loss={avg_loss:.4f}")

    return epoch_losses


def _train_model_tcd(model, dataloader, train_cfg, device, recursive_loop) -> tuple:
    """Train TCD-JEPA with recursive loop."""
    num_epochs = train_cfg["epochs"]
    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_schedule = momentum_schedule(
        train_cfg.get("ema_start", 0.996),
        train_cfg.get("ema_end", 1.0),
        total_steps,
    )

    stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)

    model.train()
    epoch_losses = []
    module_counts = []
    convergence_scores = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            result = model(images, masks_enc, masks_pred)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            wd_scheduler.step()
            m = next(ema_schedule)
            model.update_target_encoder(m)

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        epoch_losses.append(avg_loss)

        # Recursive loop step (uses last batch's representations)
        with torch.no_grad():
            z_context = model.context_encoder(images)
            target_repr = model.target_encoder(images)

        energy_fn = stream_encoder.make_energy_fn(target_repr)
        loop_result = recursive_loop.step(z_context, energy_fn, epoch=epoch)

        module_counts.append(recursive_loop.num_modules)
        if "convergence" in loop_result:
            convergence_scores.append(loop_result["convergence"]["convergence_score"])

        logger.info(
            f"  Epoch {epoch}: loss={avg_loss:.4f} "
            f"modules={recursive_loop.num_modules} "
            f"converged={recursive_loop.is_converged}"
        )

    return epoch_losses, module_counts, convergence_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/two_rooms.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_experiment(args.config, args.overrides)
