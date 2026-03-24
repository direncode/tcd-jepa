"""Distributed training for TCD-JEPA on multiple GPUs.

Uses PyTorch DDP (DistributedDataParallel) with bf16 mixed precision
and torch.compile for maximum throughput on H100 GPUs.

Launch with torchrun:
    torchrun --nproc_per_node=8 train_distributed.py --config configs/dist_cifar10.yaml
    torchrun --nproc_per_node=8 train_distributed.py --config configs/dist_imagenet.yaml --tcd

For full error tracebacks on crash:
    TORCHELASTIC_ERROR_FILE=/tmp/torch_error.json torchrun ...
"""

import argparse
import logging
import subprocess
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel, build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.losses import jepa_loss
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.checkpointing import save_checkpoint
from tcd_jepa.utils.logging import MetricLogger
from tcd_jepa.utils.masking import MaskCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tcd_jepa.distributed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size()


def log_main(msg: str):
    """Log only on rank 0."""
    if is_main_process():
        logger.info(msg)


class DropLabel(torch.utils.data.Dataset):
    """Wrapper that drops labels for unsupervised training."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds[idx][0]


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_cifar10(cfg: dict):
    data_cfg = cfg.get("data", {})
    img_size = data_cfg.get("img_size", 32)
    transforms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if img_size != 32:
        transforms.append(T.Resize(img_size))
    transforms += [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    dataset = torchvision.datasets.CIFAR10(
        root=data_cfg.get("data_dir", "./data"),
        train=True, download=is_main_process(),
        transform=T.Compose(transforms),
    )
    return DropLabel(dataset)


def build_stl10(cfg: dict):
    data_cfg = cfg.get("data", {})
    img_size = data_cfg.get("img_size", 96)
    transforms = [
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ]
    dataset = torchvision.datasets.STL10(
        root=data_cfg.get("data_dir", "./data"),
        split="train+unlabeled", download=is_main_process(),
        transform=T.Compose(transforms),
    )
    return DropLabel(dataset)


def build_imagenet(cfg: dict):
    data_cfg = cfg.get("data", {})
    img_size = data_cfg.get("img_size", 224)
    data_dir = data_cfg.get("data_dir", "./data/imagenet")
    train_dir = os.path.join(data_dir, "train")

    transforms = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(train_dir, transform=transforms)
    return DropLabel(dataset)


DATASET_BUILDERS = {
    "cifar10": build_cifar10,
    "stl10": build_stl10,
    "imagenet": build_imagenet,
}


def build_dataset(cfg: dict):
    dataset_name = cfg.get("data", {}).get("dataset", "cifar10")
    builder = DATASET_BUILDERS.get(dataset_name)
    if builder is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_BUILDERS.keys())}")
    return builder(cfg)


# ---------------------------------------------------------------------------
# Distributed Trainer
# ---------------------------------------------------------------------------

class DistributedTrainer:
    """DDP trainer with bf16 autocast, gradient clipping, and optional TCD loop."""

    def __init__(
        self,
        model: TCDJEPAModel,
        ddp_model: DDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: CosineWDSchedule,
        ema_schedule,
        train_loader: DataLoader,
        sampler: DistributedSampler,
        device: torch.device,
        cfg: dict,
        metric_logger=None,
        checkpoint_dir: str = "checkpoints",
        recursive_loop=None,
        stream_encoder=None,
    ):
        self.model = model  # unwrapped model for EMA / target encoder access
        self.ddp_model = ddp_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.ema_schedule = ema_schedule
        self.train_loader = train_loader
        self.sampler = sampler
        self.device = device
        self.cfg = cfg
        self.metric_logger = metric_logger
        self.checkpoint_dir = Path(checkpoint_dir)
        if is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.recursive_loop = recursive_loop
        self.stream_encoder = stream_encoder
        self.global_step = 0
        self._known_module_ids: set = set()

        self.use_bf16 = cfg.get("training", {}).get("use_bfloat16", True)
        self.grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)

    def train(self, num_epochs: int):
        self.model.train()

        for epoch in range(num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0
            t0 = time.time()

            for images, masks_enc, masks_pred in self.train_loader:
                loss_val, lr, wd, mom = self._train_step(images, masks_enc, masks_pred)
                epoch_loss += loss_val
                num_batches += 1
                self.global_step += 1

            # Average loss across all ranks
            avg_loss = epoch_loss / max(num_batches, 1)
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

            # Recursive loop (on rank 0 only, then broadcast)
            if self.recursive_loop is not None and self.stream_encoder is not None:
                self._run_recursive_loop(images, epoch)

            epoch_time = time.time() - t0

            if is_main_process():
                modules_str = ""
                if self.recursive_loop is not None:
                    modules_str = f" modules={self.recursive_loop.num_modules}"
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: loss={avg_loss:.4f} "
                    f"lr={lr:.6f} time={epoch_time:.1f}s{modules_str}"
                )

                if self.metric_logger is not None:
                    metrics = {
                        "epoch": epoch,
                        "loss": avg_loss,
                        "lr": lr,
                        "wd": wd,
                        "momentum": mom,
                        "epoch_time": epoch_time,
                    }
                    if self.recursive_loop:
                        metrics["num_modules"] = self.recursive_loop.num_modules
                    self.metric_logger.log(metrics, step=epoch)

                # Checkpoint
                save_freq = self.cfg.get("training", {}).get("checkpoint_freq", 10)
                if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                    save_checkpoint(
                        path=str(self.checkpoint_dir / f"checkpoint_{epoch:04d}.pt"),
                        epoch=epoch,
                        encoder=self.model.context_encoder,
                        predictor=self.model.predictor,
                        target_encoder=self.model.target_encoder,
                        optimizer=self.optimizer,
                    )

            dist.barrier()

    def _train_step(self, images, masks_enc, masks_pred):
        images = images.to(self.device, non_blocking=True)
        masks_enc = [m.to(self.device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(self.device, non_blocking=True) for m in masks_pred]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            result = self.ddp_model(images, masks_enc, masks_pred)
            loss = result["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.grad_clip)

        self.optimizer.step()

        # Update schedules
        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.ema_schedule)

        # EMA update (on unwrapped model)
        self.model.update_target_encoder(new_m)

        return loss.item(), new_lr, new_wd, new_m

    def _run_recursive_loop(self, images, epoch):
        """Run TCD recursive loop at end of epoch."""
        with torch.no_grad():
            imgs = images.to(self.device, non_blocking=True)
            z = self.model.context_encoder(imgs)
            t = self.model.target_encoder(imgs)
        energy_fn = self.stream_encoder.make_energy_fn(t)
        loop_result = self.recursive_loop.step(z, energy_fn, epoch=epoch)

        if loop_result.get("crystallized"):
            n_mods = loop_result["crystallization"]["num_active_modules"]
            log_main(f"  Recursive loop: {n_mods} active modules")
            self._register_new_module_params()

    def _register_new_module_params(self):
        if self.recursive_loop is None:
            return
        registry = self.recursive_loop.crystallizer.registry
        new_params = []
        for module_id, module in registry.get_all_modules():
            if module_id not in self._known_module_ids:
                self._known_module_ids.add(module_id)
                params = list(module.parameters())
                if params:
                    new_params.extend(params)
                    module.to(self.device)
        if new_params:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.add_param_group({
                "params": new_params,
                "lr": current_lr,
                "weight_decay": 0.0,
            })
            log_main(f"  Added {len(new_params)} new module params to optimizer")


# ---------------------------------------------------------------------------
# Linear Probe Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, dataloader, device, use_bf16=True):
    """Extract features from the context encoder for linear probing."""
    model.eval()
    all_features = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            features = model.context_encoder(images)  # [B, N, D]
        # Global average pool over patches
        features = features.mean(dim=1).float()  # [B, D]
        all_features.append(features.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe(model, cfg, device):
    """Run a fast linear probe to evaluate representation quality."""
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "cifar10")
    img_size = data_cfg.get("img_size", 32)

    if dataset_name == "cifar10":
        transform = T.Compose([
            T.Resize(img_size) if img_size != 32 else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = torchvision.datasets.CIFAR10(
            root=data_cfg.get("data_dir", "./data"), train=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            root=data_cfg.get("data_dir", "./data"), train=False, transform=transform)
        num_classes = 10
    elif dataset_name == "stl10":
        transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        train_ds = torchvision.datasets.STL10(
            root=data_cfg.get("data_dir", "./data"), split="train", transform=transform)
        test_ds = torchvision.datasets.STL10(
            root=data_cfg.get("data_dir", "./data"), split="test", transform=transform)
        num_classes = 10
    elif dataset_name == "imagenet":
        data_dir = data_cfg.get("data_dir", "./data/imagenet")
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_ds = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=transform)
        test_ds = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "val"), transform=transform)
        num_classes = 1000
    else:
        log_main(f"Linear probe not supported for dataset: {dataset_name}")
        return

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    log_main("Extracting training features...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    log_main("Extracting test features...")
    test_feats, test_labels = extract_features(model, test_loader, device)

    embed_dim = train_feats.shape[1]
    log_main(f"Linear probe: {train_feats.shape[0]} train, {test_feats.shape[0]} test, dim={embed_dim}")

    # Train linear classifier
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    # Move features to device
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    probe_bs = 1024
    classifier.train()
    for ep in range(50):
        perm = torch.randperm(len(train_feats), device=device)
        for i in range(0, len(train_feats), probe_bs):
            idx = perm[i:i + probe_bs]
            logits = classifier(train_feats[idx])
            loss = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100

    log_main(f"Linear probe accuracy: {acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distributed TCD-JEPA Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--tcd", action="store_true", help="Enable TCD recursive loop")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--probe", action="store_true", help="Run linear probe after training")
    parser.add_argument("--eval", action="store_true", help="Run full benchmark eval after training")
    parser.add_argument("--eval-epochs", type=int, default=30, help="Pretraining epochs for benchmark eval")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    # -- Distributed setup --
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    world_size = get_world_size()

    from tcd_jepa.utils.config import load_config_with_overrides
    cfg = load_config_with_overrides(args.config, args.overrides)

    # Seed (offset by rank for data diversity)
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed + dist.get_rank())
    np.random.seed(seed + dist.get_rank())

    log_main(f"World size: {world_size}, local_rank: {local_rank}")
    log_main(f"Dataset: {cfg.get('data', {}).get('dataset', 'cifar10')}")

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    # -- Build model --
    model = build_tcd_jepa(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=args.tcd,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log_main(f"Model parameters: {param_count:,}")

    # Disable context encoder hooks — they cause torch.compile recompilations
    # and CPU–GPU sync overhead.  Stats can be re-enabled for analysis later.
    model.context_encoder.disable_hooks()

    # Optional torch.compile
    if args.compile:
        log_main("Compiling model with torch.compile...")
        model.context_encoder.encoder = torch.compile(model.context_encoder.encoder)
        model.predictor = torch.compile(model.predictor)
        log_main("Compilation complete")

    # Wrap in DDP (find_unused_parameters needed for dynamic predictor)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=args.tcd,
    )

    # -- Build dataset + distributed sampler --
    # Ensure only rank 0 downloads
    if not is_main_process():
        dist.barrier()
    dataset = build_dataset(cfg)
    if is_main_process():
        dist.barrier()

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True,
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

    per_gpu_batch = train_cfg["batch_size"]
    effective_batch = per_gpu_batch * world_size
    num_workers = cfg.get("data", {}).get("num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=mask_collator,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=3 if num_workers > 0 else None,
    )

    log_main(f"Dataset: {len(dataset)} samples, batch_size={per_gpu_batch}x{world_size}={effective_batch}")
    log_main(f"Batches/epoch: {len(dataloader)}")

    # -- Optimizer & schedulers --
    # Linear scaling rule: scale LR by world_size
    base_lr = train_cfg["learning_rate"]
    scaled_lr = base_lr * world_size
    log_main(f"LR scaling: {base_lr} -> {scaled_lr} (x{world_size})")

    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=scaled_lr, weight_decay=train_cfg["weight_decay"])

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=scaled_lr,
        T_max=total_steps,
        final_lr=train_cfg.get("final_lr", 0.0),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=train_cfg["weight_decay"],
        T_max=total_steps,
        final_wd=train_cfg.get("final_weight_decay", train_cfg["weight_decay"]),
    )
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"],
        cfg["model"]["ema"]["end"],
        total_steps,
    )

    # -- Logging (rank 0 only) --
    metric_logger = None
    if is_main_process():
        log_cfg = cfg.get("logging", {})
        log_dir = log_cfg.get("log_dir", "./logs")
        metric_logger = MetricLogger(
            log_dir=log_dir,
            use_wandb=log_cfg.get("use_wandb", False),
            wandb_project=log_cfg.get("wandb_project", "tcd-jepa"),
            wandb_config=cfg,
        )

    # -- Optional TCD recursive loop --
    recursive_loop = None
    stream_encoder = None
    if args.tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2,
            crystallize_every=5,
            langevin_steps=20,
            device=device,
        )
        stream_encoder = StreamEncoder(model.context_encoder, model.target_encoder)
        model.set_module_registry(recursive_loop.crystallizer.registry)
        log_main("TCD recursive loop enabled")

    # -- Train --
    log_cfg = cfg.get("logging", {})
    checkpoint_dir = str(Path(log_cfg.get("log_dir", "./logs")) / "checkpoints")

    trainer = DistributedTrainer(
        model=model,
        ddp_model=ddp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        ema_schedule=ema_schedule,
        train_loader=dataloader,
        sampler=sampler,
        device=device,
        cfg=cfg,
        metric_logger=metric_logger,
        checkpoint_dir=checkpoint_dir,
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
    )

    log_main(f"Starting distributed training for {num_epochs} epochs")
    t_start = time.time()
    trainer.train(num_epochs)
    total_time = time.time() - t_start
    log_main(f"Training complete in {total_time:.1f}s ({total_time / 60:.1f} min)")

    # -- Linear probe --
    if args.probe and is_main_process():
        log_main("Running linear probe evaluation...")
        acc = linear_probe(model, cfg, device)
        if metric_logger is not None and acc is not None:
            metric_logger.log({"linear_probe_acc": acc}, step=num_epochs)

    # -- Full benchmark eval (rank 0 only, after DDP cleanup) --
    run_full_eval = args.eval and is_main_process()
    eval_epochs = args.eval_epochs

    if metric_logger is not None:
        metric_logger.close()

    cleanup_distributed()

    if run_full_eval:
        logger.info("Running full benchmark evaluation (experiments/benchmark_eval.py)...")
        result = subprocess.run(
            [sys.executable, "-m", "experiments.benchmark_eval",
             "--epochs", str(eval_epochs), "--seeds", "2"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            logger.error(f"Benchmark eval exited with code {result.returncode}")
        else:
            logger.info("Benchmark evaluation complete. Results in results/benchmark/")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
