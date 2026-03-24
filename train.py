"""Main training entry point for TCD-JEPA.

Usage:
    python train.py --config configs/small_scale.yaml [overrides...]

Examples:
    python train.py --config configs/small_scale.yaml training.epochs=10
    python train.py --config configs/two_rooms.yaml --tcd
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tcd_jepa")


def build_dataloader(cfg: dict, mask_collator: MaskCollator) -> DataLoader:
    """Build dataloader from config."""
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    dataset_name = data_cfg.get("dataset", "cifar10")
    img_size = data_cfg.get("img_size", 32)

    if dataset_name == "cifar10":
        transform = T.Compose([
            T.Resize(img_size) if img_size != 32 else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_cfg.get("data_dir", "./data"),
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

    elif dataset_name == "stl10":
        transform = T.Compose([
            T.Resize(img_size),
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        dataset = torchvision.datasets.STL10(
            root=data_cfg.get("data_dir", "./data"),
            split="train+unlabeled", download=True, transform=transform,
        )

        class DropLabel(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                return self.ds[idx][0]

        dataset = DropLabel(dataset)

    elif dataset_name == "imagenet100":
        from torchvision.datasets import ImageFolder
        transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_dir = data_cfg.get("data_dir", "./data/imagenet100")
        dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform)

        class DropLabel(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                return self.ds[idx][0]

        dataset = DropLabel(dataset)

    elif dataset_name == "two_rooms":
        from experiments.two_rooms.environment import TwoRoomsDataset
        dataset = TwoRoomsDataset(
            num_episodes=200,
            episode_length=50,
            render_size=img_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=mask_collator,
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train TCD-JEPA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--tcd", action="store_true", help="Enable TCD recursive loop")
    parser.add_argument("--backend", choices=["langevin", "ocean"], default="langevin", help="Physics backend for System 2")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., training.epochs=10)")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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

    # Build model
    model = build_tcd_jepa(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=enc_cfg.get("in_chans", 3),
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=args.tcd,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Build data
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
    dataloader = build_dataloader(cfg, mask_collator)
    logger.info(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    # Build optimizer and schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"],
        cfg["model"]["ema"]["end"],
        total_steps,
    )

    # Logging
    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    metric_logger = MetricLogger(
        log_dir=log_dir,
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project", "tcd-jepa"),
        wandb_config=cfg,
    )

    # Optional TCD recursive loop
    recursive_loop = None
    stream_encoder = None
    if args.tcd:
        tcd_cfg = cfg.get("tcd", {})
        backend = None
        backend_choice = args.backend
        if tcd_cfg.get("physics_backend") == "ocean":
            backend_choice = "ocean"

        if backend_choice == "ocean":
            from ocean_core import OceanConfig, set_config
            from tcd_jepa.backends.ocean import OceanBackend
            ocean_cfg = cfg.get("ocean", {})
            set_config(OceanConfig(
                shcg_base_dimension=ocean_cfg.get("shcg_base_dimension", 64),
                btut_l_max=ocean_cfg.get("btut_l_max", 128),
                irdb_max_entities=ocean_cfg.get("irdb_max_entities", 500000),
                embedding_model=ocean_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            ))
            backend = OceanBackend(embed_dim=enc_cfg["embed_dim"])
            logger.info("Using Ocean physics backend")

        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=tcd_cfg.get("explore_every", 2),
            crystallize_every=tcd_cfg.get("crystallize_every", 5),
            langevin_steps=tcd_cfg.get("langevin_steps", 20),
            device=device,
            backend=backend,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)
        logger.info("TCD recursive loop enabled")

    # Scaler for mixed precision
    scaler = None
    if train_cfg.get("use_bfloat16", False) and device.type == "cuda":
        scaler = torch.amp.GradScaler()

    # Build trainer
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
        checkpoint_dir=str(Path(log_dir) / "checkpoints"),
        scaler=scaler,
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
    )

    logger.info(f"Starting training for {num_epochs} epochs")
    trainer.train(num_epochs)

    metric_logger.close()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
