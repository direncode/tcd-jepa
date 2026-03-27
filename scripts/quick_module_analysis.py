#!/usr/bin/env python
"""Quick module analysis — run immediately after train_manifold.py --tcd.

Retrains TCD on semiconductor data and analyzes modules in-memory.

Usage:
    cd /workspace/tcd-jepa
    python scripts/quick_module_analysis.py
"""

import json
import logging
import sys

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("analysis")

sys.path.insert(0, ".")

from tcd_jepa.core.recursive_loop import RecursiveLoop  # noqa: E402
from tcd_jepa.manifold.dataset import LatentOceanDataset  # noqa: E402
from tcd_jepa.manifold.masking import ManifoldMaskCollator  # noqa: E402
from tcd_jepa.manifold.model import build_manifold_jepa  # noqa: E402
from tcd_jepa.manifold.trainer import ManifoldTrainer  # noqa: E402
from tcd_jepa.models.target_encoder import momentum_schedule  # noqa: E402
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule  # noqa: E402
from tcd_jepa.training.trainer import build_optimizer  # noqa: E402

# Parse data dir from command line or default to semiconductor
data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/semiconductor"
logger.info(f"Data dir: {data_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_manifold_jepa(
    fingerprint_dim=384, num_tokens=64, coord_dim=3,
    embed_dim=192, depth=6, num_heads=3,
    predictor_embed_dim=96, predictor_depth=4, predictor_num_heads=3,
    use_dynamic_predictor=True, use_causal_encoding=True,
    use_velocity_encoding=True, sphere_radius=4.5,
).to(device)

dataset = LatentOceanDataset(data_dir, num_tokens=64, num_samples=1000)
collator = ManifoldMaskCollator(
    num_tokens=64, context_ratio=(0.3, 0.5), target_ratio=(0.15, 0.3),
    num_context_masks=4, num_target_masks=1, min_keep=4,
    use_geodesic=True, sphere_radius=4.5,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                    collate_fn=collator, drop_last=True)

steps = len(loader)
total = 100 * steps
opt = build_optimizer(model, lr=0.001, weight_decay=0.05)
lr_s = WarmupCosineSchedule(opt, 5 * steps, 1e-4, 0.001, total)
wd_s = CosineWDSchedule(opt, 0.05, total)
ema_s = momentum_schedule(0.996, 1.0, total)

rl = RecursiveLoop(
    embed_dim=192, explore_every=3, crystallize_every=6,
    langevin_steps=30, langevin_step_size=0.01,
    persistence_threshold=0.2, max_modules=16, device=device,
)
model.set_module_registry(rl.crystallizer.registry)

trainer = ManifoldTrainer(
    model=model, optimizer=opt, lr_scheduler=lr_s,
    wd_scheduler=wd_s, momentum_schedule=ema_s,
    train_loader=loader, device=device,
    cfg={"training": {"grad_clip_norm": 1.0, "checkpoint_freq": 999}},
    checkpoint_dir="./logs/module_analysis/ckpt",
    recursive_loop=rl,
)

logger.info("Training 100 epochs with TCD...")
trainer.train(100)

# Analyze modules in memory
modules = rl.crystallizer.registry.get_all_modules()
logger.info(f"\n{'=' * 60}")
logger.info(f"CRYSTALLIZED MODULES: {len(modules)}")
logger.info(f"{'=' * 60}")

entities_path = f"{data_dir}/entities.json"
with open(entities_path) as f:
    entities = json.load(f)

model.eval()
all_emb, all_idx = [], []
with torch.no_grad():
    for i in range(min(300, len(dataset))):
        s = dataset[i]
        z = model.context_encoder(
            s["fingerprints"].unsqueeze(0).to(device),
            coords=s["coords"].unsqueeze(0).to(device),
            adjacency=s["adjacency"].unsqueeze(0).to(device),
            velocity=s["velocity"].unsqueeze(0).to(device),
        )
        all_emb.append(z.squeeze(0))
        all_idx.append(s["indices"])

emb = torch.cat(all_emb)
idx = torch.cat(all_idx)
z_flat = emb.reshape(-1, emb.shape[-1])

for mid, mod in modules:
    mod = mod.to(device)
    with torch.no_grad():
        act = mod(z_flat).norm(dim=-1)
    topk_v, topk_i = act.topk(min(15, len(act)))
    topk_i = topk_i.cpu()
    names, attrs = [], {}
    for gid in idx[topk_i].tolist():
        if gid < len(entities):
            e = entities[gid]
            name = e.get("name", e.get("company", str(gid)))
            names.append(name)
            # Collect all string attributes for clustering
            for key in ["type", "stage", "country", "actor1_type", "company"]:
                if key in e and e[key]:
                    attrs[f"{key}={e[key]}"] = attrs.get(f"{key}={e[key]}", 0) + 1
    dom_attr = max(attrs, key=attrs.get) if attrs else "?"
    print(f"\nModule {mid}:")
    print(f"  Dominant: {dom_attr}")
    print(f"  Attributes: {dict(sorted(attrs.items(), key=lambda x: -x[1])[:5])}")
    print(f"  Entities: {', '.join(names[:8])}")

print("\nANALYSIS COMPLETE")
