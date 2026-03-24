"""Evaluate a trained checkpoint with linear probe + k-NN.

Usage:
    python eval_checkpoint.py                                    # auto-find latest
    python eval_checkpoint.py --checkpoint path/to/checkpoint.pt
    python eval_checkpoint.py --config configs/dist_cifar10.yaml
"""
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.utils.checkpointing import load_checkpoint


def find_latest_checkpoint(log_dir="logs"):
    """Find the most recent checkpoint file."""
    patterns = [f"{log_dir}/**/checkpoint_*.pt", "logs/**/checkpoint_*.pt"]
    for pattern in patterns:
        ckpts = sorted(glob.glob(pattern, recursive=True))
        if ckpts:
            return ckpts[-1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/dist_cifar10.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    e, p = cfg["model"]["encoder"], cfg["model"]["predictor"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_tcd_jepa(
        img_size=e["img_size"], patch_size=e["patch_size"],
        embed_dim=e["embed_dim"], depth=e["depth"], num_heads=e["num_heads"],
        predictor_embed_dim=p["predictor_embed_dim"],
        predictor_depth=p["predictor_depth"],
        predictor_num_heads=p["num_heads"],
        drop_path_rate=e.get("drop_path_rate", 0.1),
    ).to(device)

    # Find checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint(
        cfg.get("logging", {}).get("log_dir", "logs")
    )
    if ckpt_path is None:
        print("No checkpoint found. Run training first.")
        return

    ckpt = load_checkpoint(
        ckpt_path,
        encoder=model.context_encoder,
        predictor=model.predictor,
        target_encoder=model.target_encoder,
        device=args.device,
    )
    print(f"Loaded: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # Dataset
    dataset_name = cfg.get("data", {}).get("dataset", "cifar10")
    if dataset_name == "cifar10":
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        trn = DataLoader(torchvision.datasets.CIFAR10("./data", True, transform=tfm), 512, num_workers=4)
        tst = DataLoader(torchvision.datasets.CIFAR10("./data", False, transform=tfm), 512, num_workers=4)
        num_classes = 10
    else:
        print(f"Eval not implemented for {dataset_name}")
        return

    # Extract features
    model.eval()
    def extract(loader):
        fs, ls = [], []
        for x, y in loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fs.append(model.context_encoder(x.to(device)).mean(1).float().cpu())
            ls.append(y)
        return torch.cat(fs).to(device), torch.cat(ls).to(device)

    with torch.no_grad():
        print("Extracting features...")
        trf, trl = extract(trn)
        tef, tel = extract(tst)
    print(f"Features: train={trf.shape}, test={tef.shape}")

    # Normalize
    mu, std = trf.mean(0), trf.std(0).clamp(min=1e-6)
    trf_n = (trf - mu) / std
    tef_n = (tef - mu) / std

    # Linear probe
    clf = nn.Linear(trf.shape[1], num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    best_acc = 0.0
    for ep in range(100):
        clf.train()
        perm = torch.randperm(len(trf_n), device=device)
        for i in range(0, len(trf_n), 1024):
            idx = perm[i:i + 1024]
            loss = F.cross_entropy(clf(trf_n[idx]), trl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 20 == 0:
            clf.eval()
            with torch.no_grad():
                acc = (clf(tef_n).argmax(1) == tel).float().mean().item() * 100
                best_acc = max(best_acc, acc)
                print(f"  Epoch {ep+1}: {acc:.2f}%")
    clf.eval()
    with torch.no_grad():
        acc = (clf(tef_n).argmax(1) == tel).float().mean().item() * 100
        best_acc = max(best_acc, acc)
    print(f"\nLinear probe: {best_acc:.2f}%")

    # k-NN
    tn = F.normalize(trf, dim=1)
    en = F.normalize(tef, dim=1)
    for k in [1, 5, 20]:
        _, topk = (en @ tn.T).topk(k, dim=1)
        knn_acc = (trl[topk].mode(1).values == tel).float().mean().item() * 100
        print(f"k-NN k={k}: {knn_acc:.2f}%")

    # Collapse metrics
    from tcd_jepa.training.losses import compute_collapse_metrics
    metrics = compute_collapse_metrics(trf[:1000])
    print(f"\nCollapse metrics:")
    print(f"  Effective rank: {metrics['effective_rank']:.1f} / {trf.shape[1]}")
    print(f"  Std mean: {metrics['std_mean']:.4f}")
    print(f"  Std min:  {metrics['std_min']:.6f}")
    print(f"  Uniformity: {metrics['uniformity']:.4f}")


if __name__ == "__main__":
    main()
