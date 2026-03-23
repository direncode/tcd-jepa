"""Wait for training to finish, then evaluate the checkpoint.
Run this in a second terminal: python run_eval_after.py
"""
import time
import json
import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from experiments.benchmark_eval import extract_features, linear_probe, knn_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_after")

CHECKPOINT_DIR = Path("./logs/h100_run/checkpoints")
RESULTS_DIR = Path("./logs/h100_run/results")

# Model config matching h100_runpod.yaml
IMG_SIZE = 32
PATCH_SIZE = 4
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
PREDICTOR_EMBED_DIM = 192
PREDICTOR_DEPTH = 6

def wait_for_checkpoint():
    """Wait for final checkpoint to appear."""
    target = CHECKPOINT_DIR / "checkpoint_0299.pt"
    # Also check 0049, 0099, etc. in case numbering differs
    logger.info(f"Waiting for {target} ...")
    while not target.exists():
        # Also check if training wrote any checkpoint recently
        time.sleep(10)
    # Wait a bit more to ensure file is fully written
    time.sleep(5)
    logger.info(f"Found checkpoint: {target}")
    return target


def load_encoder(checkpoint_path, device):
    """Load encoder from checkpoint."""
    model = build_tcd_jepa(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        predictor_embed_dim=PREDICTOR_EMBED_DIM,
        predictor_depth=PREDICTOR_DEPTH,
        predictor_num_heads=NUM_HEADS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.context_encoder.load_state_dict(ckpt["encoder"])
    logger.info("Encoder loaded successfully")
    return model.context_encoder


def get_eval_loaders(data_dir="./data"):
    """Get CIFAR-10 train/test loaders with labels."""
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False,
        transform=T.Compose([T.ToTensor(), normalize]),
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False,
        transform=T.Compose([T.ToTensor(), normalize]),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    return train_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Wait for training to finish
    ckpt_path = wait_for_checkpoint()

    # Load encoder
    encoder = load_encoder(ckpt_path, device)

    # Get eval data
    train_loader, test_loader = get_eval_loaders()

    # Extract features
    logger.info("Extracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    logger.info(f"Features: train={train_feats.shape}, test={test_feats.shape}")

    # Linear probe
    logger.info("Running linear probe (100 epochs)...")
    lin_acc = linear_probe(
        train_feats, train_labels, test_feats, test_labels,
        embed_dim=EMBED_DIM, device=device,
    )
    logger.info(f"Linear probe accuracy: {lin_acc:.2f}%")

    # k-NN
    logger.info("Running k-NN evaluation...")
    knn_results = knn_evaluate(
        train_feats, train_labels, test_feats, test_labels, device=device,
    )
    logger.info(f"k-NN results: {knn_results}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(ckpt_path),
        "linear_probe_acc": lin_acc,
        **knn_results,
        "embed_dim": EMBED_DIM,
        "epochs": 300,
    }
    results_path = RESULTS_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Linear Probe:  {lin_acc:.2f}%")
    for k, v in knn_results.items():
        print(f"  {k}:  {v:.2f}%")
    print(f"\n  Saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
