"""Overnight TCD-JEPA training with automatic evaluation.

Runs multiple training phases with fresh LR schedules to avoid plateauing.
Evaluates after each phase and saves all results.

Usage (on RunPod):
    nohup python run_overnight.py > overnight.log 2>&1 &
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import Trainer, build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.utils.masking import MaskCollator
from tcd_jepa.utils.logging import MetricLogger
from experiments.benchmark_eval import extract_features, linear_probe, knn_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("overnight")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CFG = {
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "predictor_embed_dim": 192,
    "predictor_depth": 6,
}

# Training phases: each gets a fresh LR schedule to prevent plateauing
# Phase 1: TCD 300 epochs (compare directly to vanilla baseline)
# Phase 2: Continue TCD 600 more epochs with fresh LR (warm restart)
# Phase 3: Continue TCD 1000 more epochs with lower peak LR
PHASES = [
    {"name": "tcd_300",  "epochs": 300,  "lr": 0.0015, "warmup": 30,  "use_tcd": True},
    {"name": "tcd_900",  "epochs": 600,  "lr": 0.001,  "warmup": 20,  "use_tcd": True},
    {"name": "tcd_1900", "epochs": 1000, "lr": 0.0007, "warmup": 15,  "use_tcd": True},
]

RESULTS_DIR = Path("./logs/overnight/results")
CHECKPOINT_DIR = Path("./logs/overnight/checkpoints")


def get_ssl_dataloader(batch_size, mask_collator):
    """Get CIFAR-10 SSL dataloader (no labels)."""
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform,
    )

    class DropLabel(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return self.ds[idx][0]

    return DataLoader(
        DropLabel(dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=mask_collator,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )


def get_eval_loaders():
    """Get CIFAR-10 train/test loaders with labels for evaluation."""
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False,
        transform=T.Compose([T.ToTensor(), normalize]),
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False,
        transform=T.Compose([T.ToTensor(), normalize]),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader


def evaluate(encoder, device, phase_name):
    """Run linear probe + k-NN evaluation and save results."""
    logger.info(f"[{phase_name}] Starting evaluation...")
    train_loader, test_loader = get_eval_loaders()

    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    logger.info(f"[{phase_name}] Features: train={train_feats.shape}, test={test_feats.shape}")

    lin_acc = linear_probe(
        train_feats, train_labels, test_feats, test_labels,
        embed_dim=MODEL_CFG["embed_dim"], device=device,
    )
    logger.info(f"[{phase_name}] Linear probe: {lin_acc:.2f}%")

    knn_results = knn_evaluate(
        train_feats, train_labels, test_feats, test_labels, device=device,
    )
    logger.info(f"[{phase_name}] k-NN: {knn_results}")

    results = {
        "phase": phase_name,
        "linear_probe_acc": lin_acc,
        **knn_results,
    }

    print(f"\n{'='*60}")
    print(f"EVAL RESULTS — {phase_name}")
    print(f"{'='*60}")
    print(f"  Linear Probe:  {lin_acc:.2f}%")
    for k, v in knn_results.items():
        print(f"  {k}:  {v:.2f}%")
    print(f"{'='*60}\n")

    return results


def run_phase(phase, model, device, mask_collator, prev_optimizer_state=None):
    """Run a single training phase with fresh LR schedule."""
    name = phase["name"]
    epochs = phase["epochs"]
    lr = phase["lr"]
    warmup = phase["warmup"]
    use_tcd = phase["use_tcd"]

    logger.info(f"\n{'#'*60}")
    logger.info(f"PHASE: {name} — {epochs} epochs, lr={lr}, tcd={use_tcd}")
    logger.info(f"{'#'*60}")

    cfg = {
        "training": {
            "batch_size": 512,
            "learning_rate": lr,
            "start_lr": 0.0001,
            "weight_decay": 0.05,
            "warmup_epochs": warmup,
            "use_bfloat16": True,
            "grad_clip_norm": 1.0,
            "checkpoint_freq": 50,
        },
        "masking": {
            "num_enc_masks": 4,
            "num_pred_masks": 1,
            "min_keep": 4,
            "enc_mask_scale": [0.15, 0.2],
            "pred_mask_scale": [0.2, 0.4],
            "aspect_ratio": [0.75, 1.5],
        },
        "model": {"ema": {"start": 0.996, "end": 1.0}},
    }

    dataloader = get_ssl_dataloader(
        batch_size=cfg["training"]["batch_size"],
        mask_collator=mask_collator,
    )

    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch

    # Fresh optimizer with new LR schedule (warm restart)
    optimizer = build_optimizer(model, lr=lr, weight_decay=cfg["training"]["weight_decay"])

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup * steps_per_epoch,
        start_lr=cfg["training"]["start_lr"],
        ref_lr=lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=cfg["training"]["weight_decay"], T_max=total_steps,
    )
    ema_schedule = momentum_schedule(0.996, 1.0, total_steps)

    # TCD recursive loop
    recursive_loop = None
    stream_encoder = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=MODEL_CFG["embed_dim"],
            explore_every=2,
            crystallize_every=5,
            langevin_steps=20,
            device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    phase_log_dir = f"./logs/overnight/{name}"
    metric_logger = MetricLogger(log_dir=phase_log_dir, use_wandb=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule=ema_schedule,
        train_loader=dataloader,
        device=device,
        cfg=cfg,
        metric_logger=metric_logger,
        checkpoint_dir=str(CHECKPOINT_DIR),
        scaler=scaler,
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
    )

    t0 = time.time()
    trainer.train(epochs)
    elapsed = time.time() - t0
    metric_logger.close()

    logger.info(f"[{name}] Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{name}_final.pt"
    torch.save({
        "encoder": model.context_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
    }, ckpt_path)
    logger.info(f"[{name}] Checkpoint saved: {ckpt_path}")

    return elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Starting overnight run — {len(PHASES)} phases")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Build model once — reused across phases (weights carry over)
    c = MODEL_CFG
    model = build_tcd_jepa(
        img_size=c["img_size"],
        patch_size=c["patch_size"],
        embed_dim=c["embed_dim"],
        depth=c["depth"],
        num_heads=c["num_heads"],
        predictor_embed_dim=c["predictor_embed_dim"],
        predictor_depth=c["predictor_depth"],
        predictor_num_heads=c["num_heads"],
        use_dynamic_predictor=True,  # TCD mode
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    mask_collator = MaskCollator(
        input_size=(c["img_size"], c["img_size"]),
        patch_size=c["patch_size"],
        nenc=4, npred=1, min_keep=4,
        enc_mask_scale=(0.15, 0.2),
        pred_mask_scale=(0.2, 0.4),
        aspect_ratio=(0.75, 1.5),
    )

    all_results = []
    total_time = 0

    for phase in PHASES:
        # Train
        elapsed = run_phase(phase, model, device, mask_collator)
        total_time += elapsed

        # Evaluate
        eval_results = evaluate(model.context_encoder, device, phase["name"])
        eval_results["train_time_s"] = elapsed
        eval_results["total_time_s"] = total_time
        eval_results["total_epochs"] = sum(
            p["epochs"] for p in PHASES[:PHASES.index(phase) + 1]
        )
        all_results.append(eval_results)

        # Save cumulative results after each phase
        results_path = RESULTS_DIR / "overnight_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("OVERNIGHT RUN COMPLETE — ALL RESULTS")
    print("=" * 70)
    print(f"{'Phase':<15} {'Epochs':>8} {'Lin Probe':>10} {'k-NN@20':>10} {'Time':>10}")
    print("-" * 70)
    for r in all_results:
        mins = r["train_time_s"] / 60
        print(f"{r['phase']:<15} {r['total_epochs']:>8} {r['linear_probe_acc']:>9.2f}% "
              f"{r['knn_k20']:>9.2f}% {mins:>8.1f}min")
    print("-" * 70)
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")

    # Compare against vanilla baseline
    print(f"\nVanilla JEPA baseline (300ep): 69.65% linear probe")
    best = max(all_results, key=lambda r: r["linear_probe_acc"])
    diff = best["linear_probe_acc"] - 69.65
    print(f"Best TCD result ({best['phase']}): {best['linear_probe_acc']:.2f}% "
          f"({diff:+.2f}% vs vanilla)")
    print("=" * 70)

    # Auto-shutdown pod to stop billing
    import subprocess
    logger.info("Training complete. Shutting down pod to stop billing...")
    subprocess.run(["runpodctl", "stop", "pod"], capture_output=True)
    # Fallback: direct shutdown if runpodctl not available
    subprocess.run(["shutdown", "-h", "now"], capture_output=True)


if __name__ == "__main__":
    main()
