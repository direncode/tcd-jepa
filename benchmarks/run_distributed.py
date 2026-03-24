#!/usr/bin/env python3
"""Distributed benchmark runner for multi-GPU training (e.g. 8xH100).

Runs all benchmarks (CIFAR-10, STL-10, ImageNet-100) with both Vanilla JEPA
and TCD-JEPA using PyTorch DDP across all available GPUs.

With 8xH100 GPUs:
  - CIFAR-10 300ep: ~20 min (was ~2h single GPU)
  - STL-10 300ep:   ~25 min (was ~3h single GPU)
  - ImageNet-100:   ~60 min (was ~8-12h single GPU)
  - All three:      ~2h total (was ~15h+ single GPU)

Usage:
    # All datasets, 8 GPUs
    torchrun --nproc_per_node=8 -m benchmarks.run_distributed --dataset all

    # Single dataset
    torchrun --nproc_per_node=8 -m benchmarks.run_distributed --dataset stl10

    # Resume from crash (auto-detects latest checkpoint)
    torchrun --nproc_per_node=8 -m benchmarks.run_distributed --dataset stl10 --resume

    # Quick test
    torchrun --nproc_per_node=8 -m benchmarks.run_distributed --dataset cifar10 --quick
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.training.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    make_distributed_loader, save_distributed_checkpoint,
    load_distributed_checkpoint, reduce_metric,
)
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.logging import MetricLogger
from tcd_jepa.utils.config import load_config_with_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark.distributed")


# ── Dataset helpers (same as single-GPU version) ─────────────────────────────

from benchmarks.run_standard_benchmarks import (
    SSLWrapper, ImageNet100Dataset,
    get_ssl_transforms, get_eval_transforms,
    get_ssl_dataset, get_eval_datasets,
    extract_features, linear_probe, knn_evaluate, representation_quality,
    PUBLISHED_BENCHMARKS, format_report,
)


# ── Distributed pretraining ──────────────────────────────────────────────────

def pretrain_distributed(cfg, dataset_name, data_dir, rank, local_rank,
                         world_size, use_tcd=False, resume_path=None):
    """Pretrain with DDP across all GPUs. Returns encoder on rank 0."""
    device = torch.device(f"cuda:{local_rank}")
    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]
    method = "TCD-JEPA" if use_tcd else "Vanilla JEPA"

    # Scale learning rate linearly with world size
    base_lr = train_cfg["learning_rate"]
    effective_lr = base_lr * world_size
    effective_batch = train_cfg["batch_size"]  # per-GPU batch size
    global_batch = effective_batch * world_size

    if is_main_process():
        logger.info(f"[{method}] DDP: {world_size} GPUs, "
                     f"per-GPU batch={effective_batch}, global batch={global_batch}, "
                     f"effective LR={effective_lr:.6f}")

    # Build model
    model = build_tcd_jepa(
        img_size=img_size,
        patch_size=enc_cfg["patch_size"],
        in_chans=enc_cfg.get("in_chans", 3),
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=use_tcd,
    ).to(device)

    if is_main_process():
        param_count = sum(p.numel() for p in model.parameters())
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

    # Build dataset with DistributedSampler
    ssl_dataset = get_ssl_dataset(dataset_name, data_dir, img_size)
    num_workers = train_cfg.get("num_workers", cfg.get("data", {}).get("num_workers", 4))
    dataloader, sampler = make_distributed_loader(
        ssl_dataset,
        batch_size=effective_batch,
        num_workers=num_workers,
        collate_fn=mask_collator,
        seed=train_cfg.get("seed", 42),
    )

    if is_main_process():
        logger.info(f"[{method}] SSL dataset: {len(ssl_dataset)} samples, "
                     f"{len(dataloader)} batches/epoch/GPU")

    # Schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=effective_lr, weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg.get("warmup_epochs", 40) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4) * world_size,
        ref_lr=effective_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_sched = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps)

    # Optional TCD loop (runs on each rank independently — exploration is local)
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

    # Mixed precision
    use_amp = train_cfg.get("use_bfloat16", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16

    # Wrap in DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_tcd)

    # Logging (rank 0 only)
    metric_logger = None
    if is_main_process():
        log_dir = cfg.get("logging", {}).get("log_dir", f"./logs/benchmark_{dataset_name}")
        suffix = "_tcd_ddp" if use_tcd else "_vanilla_ddp"
        run_log_dir = f"{log_dir}{suffix}"
        use_wandb = cfg.get("logging", {}).get("use_wandb", False)
        metric_logger = MetricLogger(
            log_dir=run_log_dir,
            use_wandb=use_wandb,
            wandb_project=cfg.get("logging", {}).get("wandb_project", "tcd-jepa-benchmarks"),
            wandb_config={**cfg, "method": method, "dataset": dataset_name,
                          "world_size": world_size, "global_batch": global_batch},
        )

    # Checkpoint setup
    ckpt_dir = Path(cfg.get("logging", {}).get("log_dir",
                    f"./logs/benchmark_{dataset_name}")) / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    suffix = "tcd" if use_tcd else "vanilla"
    ckpt_path = str(ckpt_dir / f"latest_{suffix}.pt")
    start_epoch = 0
    global_step = 0

    # Resume from checkpoint
    if resume_path or (os.path.exists(ckpt_path) and resume_path is not False):
        actual_path = resume_path if isinstance(resume_path, str) else ckpt_path
        ckpt_data = load_distributed_checkpoint(
            actual_path, ddp_model, optimizer, device=device)
        if ckpt_data:
            start_epoch = ckpt_data.get("epoch", 0) + 1
            global_step = ckpt_data.get("global_step", 0)
            # Advance schedulers
            for _ in range(global_step):
                lr_scheduler.step()
                wd_scheduler.step()
                next(ema_sched)
            # Rebuild ema_sched from the right position
            if is_main_process():
                logger.info(f"[{method}] Resuming from epoch {start_epoch}, "
                             f"step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main_process():
        logger.info(f"[{method}] Starting pretraining "
                     f"(epochs {start_epoch}-{num_epochs-1})...")
    t0 = time.time()
    _known_module_ids = set()

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        ddp_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, masks_enc, masks_pred in dataloader:
            images = images.to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = ddp_model(images, masks_enc, masks_pred)
                loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update schedules
            new_lr = lr_scheduler.step()
            new_wd = wd_scheduler.step()
            new_m = next(ema_sched)

            # EMA update (on underlying model, not DDP wrapper)
            model.update_target_encoder(new_m)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        # Average loss across ranks
        avg_loss = reduce_metric(epoch_loss / max(num_batches, 1))

        # TCD recursive loop (each rank runs independently)
        if recursive_loop is not None and stream_encoder is not None:
            with torch.no_grad():
                z = model.context_encoder(images)
                t = model.target_encoder(images)
            energy_fn = stream_encoder.make_energy_fn(t)
            loop_result = recursive_loop.step(z, energy_fn, epoch=epoch)
            if loop_result.get("crystallized") and is_main_process():
                n_mods = loop_result["crystallization"]["num_active_modules"]
                logger.info(f"  TCD: {n_mods} active modules")
                # Register new module params
                registry = recursive_loop.crystallizer.registry
                new_params = []
                for module_id, module in registry.get_all_modules():
                    if module_id not in _known_module_ids:
                        _known_module_ids.add(module_id)
                        params = list(module.parameters())
                        if params:
                            new_params.extend(params)
                            module.to(device)
                if new_params:
                    current_lr = optimizer.param_groups[0]["lr"]
                    optimizer.add_param_group({
                        "params": new_params, "lr": current_lr, "weight_decay": 0.0})

        epoch_time = (time.time() - t0) / (epoch - start_epoch + 1)

        if is_main_process():
            logger.info(
                f"Epoch {epoch}: loss={avg_loss:.4f} lr={new_lr:.6f} "
                f"time={epoch_time:.1f}s/epoch"
                + (f" modules={recursive_loop.num_modules}" if recursive_loop else ""))

            if metric_logger:
                metric_logger.log({
                    "epoch": epoch, "loss": avg_loss, "lr": new_lr,
                    "wd": new_wd, "momentum": new_m,
                    "epoch_time": epoch_time,
                    **({"num_modules": recursive_loop.num_modules}
                       if recursive_loop else {}),
                }, step=epoch)

        # Save checkpoint every 25 epochs (crash resilience)
        ckpt_freq = train_cfg.get("checkpoint_freq", 25)
        if (epoch + 1) % ckpt_freq == 0 or epoch == num_epochs - 1:
            save_distributed_checkpoint(
                ckpt_path, epoch, ddp_model, optimizer,
                extra={"global_step": global_step,
                       "method": suffix,
                       "dataset": dataset_name})
            # Also save numbered checkpoint at milestones
            if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
                save_distributed_checkpoint(
                    str(ckpt_dir / f"checkpoint_{suffix}_epoch{epoch:04d}.pt"),
                    epoch, ddp_model, optimizer,
                    extra={"global_step": global_step})

    train_time = time.time() - t0

    if is_main_process():
        logger.info(f"[{method}] Pretraining done in {train_time / 60:.1f} min "
                     f"({train_time / 3600:.2f}h)")

    if metric_logger:
        metric_logger.close()

    num_modules = recursive_loop.num_modules if recursive_loop else 0

    # Extract encoder (unwrap DDP)
    encoder = model.context_encoder

    # Cleanup
    del ddp_model, optimizer, lr_scheduler, wd_scheduler, dataloader, sampler
    del recursive_loop, stream_encoder
    torch.cuda.empty_cache()
    gc.collect()

    return encoder, train_time, num_modules


# ── Evaluation (rank 0 only) ─────────────────────────────────────────────────

def evaluate_on_rank0(encoder, cfg, dataset_name, device, embed_dim, num_modules,
                      train_time, method_name):
    """Run eval on rank 0 only. Returns result dict or None."""
    if not is_main_process():
        return None

    data_dir = cfg.get("data", {}).get("data_dir", "./data")
    img_size = cfg["model"]["encoder"]["img_size"]

    train_ds, test_ds, num_classes = get_eval_datasets(dataset_name, data_dir, img_size)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    label = "TCD-JEPA" if method_name == "tcd" else "Vanilla JEPA"
    logger.info(f"[{label}] Extracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    logger.info(f"[{label}] Features: train={train_feats.shape}, test={test_feats.shape}")

    logger.info(f"[{label}] Linear probe ({num_classes} classes)...")
    lin_acc = linear_probe(
        train_feats, train_labels, test_feats, test_labels,
        embed_dim=embed_dim, num_classes=num_classes, device=device)
    logger.info(f"[{label}] Linear probe: {lin_acc:.2f}%")

    logger.info(f"[{label}] k-NN evaluation...")
    knn_results = knn_evaluate(
        train_feats, train_labels, test_feats, test_labels, device=device)
    logger.info(f"[{label}] k-NN: {knn_results}")

    quality = representation_quality(test_feats)
    logger.info(f"[{label}] Quality: {quality}")

    return {
        "method": method_name,
        "linear_probe_acc": lin_acc,
        **knn_results,
        **quality,
        "train_time_s": train_time,
        "train_time_h": train_time / 3600,
        "num_modules": num_modules,
        "epochs": cfg["training"]["epochs"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_distributed_benchmark(dataset_name, cfg, rank, local_rank, world_size,
                              seeds=(42,), resume=False):
    """Run full benchmark with DDP."""
    device = torch.device(f"cuda:{local_rank}")
    enc_cfg = cfg["model"]["encoder"]
    embed_dim = enc_cfg["embed_dim"]
    img_size = enc_cfg["img_size"]
    data_dir = cfg.get("data", {}).get("data_dir", "./data")

    if is_main_process():
        logger.info(f"\n{'='*70}")
        logger.info(f"DISTRIBUTED BENCHMARK: {dataset_name.upper()} | "
                     f"{world_size}x GPU | ViT-S/16 | {img_size}px")
        logger.info(f"{'='*70}")

    all_results = {"dataset": dataset_name, "num_classes": 10,
                   "img_size": img_size, "embed_dim": embed_dim,
                   "world_size": world_size, "vanilla": [], "tcd": []}

    # Load any previously saved results for resume
    out_dir = Path(f"results/benchmark_{dataset_name}")
    results_path = out_dir / "results.json"
    if resume and results_path.exists() and is_main_process():
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            logger.info(f"Loaded existing results: "
                        f"{len(all_results.get('vanilla', []))} vanilla, "
                        f"{len(all_results.get('tcd', []))} tcd")
        except Exception:
            pass

    for seed in seeds:
        if is_main_process():
            logger.info(f"\n{'─'*50} SEED {seed} {'─'*50}")

        for method_name, use_tcd in [("vanilla", False), ("tcd", True)]:
            # Check if already done (for resume)
            already_done = any(
                r.get("seed") == seed
                for r in all_results.get(method_name, [])
            )
            if already_done and resume:
                if is_main_process():
                    logger.info(f"[{method_name}] seed={seed} already done, skipping")
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)
            label = "TCD-JEPA" if use_tcd else "Vanilla JEPA"

            if is_main_process():
                logger.info(f"\n>>> {label} (seed={seed})")

            try:
                resume_ckpt = resume if isinstance(resume, str) else resume
                encoder, train_time, num_modules = pretrain_distributed(
                    cfg, dataset_name,
                    data_dir, rank, local_rank, world_size,
                    use_tcd=use_tcd,
                    resume_path=resume_ckpt if resume else None,
                )

                # Eval on rank 0 only
                result = evaluate_on_rank0(
                    encoder, cfg, dataset_name, device, embed_dim,
                    num_modules, train_time, method_name)

                if result is not None:
                    result["seed"] = seed
                    all_results[method_name].append(result)
                    # Save incrementally
                    save_results(all_results, dataset_name)

            except Exception as e:
                if is_main_process():
                    logger.error(f"[{label}] FAILED (seed={seed}): {e}", exc_info=True)
            finally:
                torch.cuda.empty_cache()
                gc.collect()

            # Sync before next method
            dist.barrier()

    return all_results


def save_results(results, dataset_name):
    """Save results to disk (rank 0 only)."""
    if not is_main_process():
        return
    out_dir = Path(f"results/benchmark_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    report = format_report(results)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="TCD-JEPA Distributed Benchmarks")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "stl10", "imagenet100", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size (global = this * num_gpus)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    # Setup DDP
    rank, local_rank, world_size = setup_distributed()

    datasets = ["cifar10", "stl10", "imagenet100"] if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        config_path = args.config or f"configs/benchmark_{ds_name}.yaml"
        if not Path(config_path).exists():
            if is_main_process():
                logger.error(f"Config not found: {config_path}")
            continue

        cfg = load_config_with_overrides(config_path, [])

        if args.epochs:
            cfg["training"]["epochs"] = args.epochs
            cfg["training"]["warmup_epochs"] = min(
                cfg["training"].get("warmup_epochs", 40), args.epochs // 4)
        if args.batch_size:
            cfg["training"]["batch_size"] = args.batch_size
        if args.no_wandb:
            cfg["logging"]["use_wandb"] = False
        if args.data_dir:
            cfg["data"]["data_dir"] = args.data_dir

        # Adjust checkpoint frequency for faster saves with DDP
        cfg["training"]["checkpoint_freq"] = 25

        if args.quick:
            cfg["training"]["epochs"] = 5
            cfg["training"]["warmup_epochs"] = 1
            cfg["logging"]["use_wandb"] = False
            seeds = [42]
        else:
            seeds = [42 + i * 81 for i in range(args.seeds)]

        run_distributed_benchmark(
            ds_name, cfg, rank, local_rank, world_size,
            seeds=seeds, resume=args.resume)

    cleanup_distributed()
    if is_main_process():
        logger.info("\nAll distributed benchmarks complete!")


if __name__ == "__main__":
    main()
