"""Evaluate a trained checkpoint with linear probe + k-NN."""
import torch, yaml, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
import torch.nn as nn, torch.nn.functional as F

cfg = yaml.safe_load(open("configs/dist_cifar10.yaml"))
e, p = cfg["model"]["encoder"], cfg["model"]["predictor"]
device = torch.device("cuda:0")

model = build_tcd_jepa(
    img_size=e["img_size"], patch_size=e["patch_size"], embed_dim=e["embed_dim"],
    depth=e["depth"], num_heads=e["num_heads"],
    predictor_embed_dim=p["predictor_embed_dim"],
    predictor_depth=p["predictor_depth"], predictor_num_heads=p["num_heads"],
).to(device)

ckpt = torch.load("logs/dist_cifar10/checkpoints/checkpoint_0049.pt", map_location=device, weights_only=True)
model.context_encoder.encoder.load_state_dict(
    {k.replace("_orig_mod.", ""): v for k, v in ckpt["encoder"].items()})
model.target_encoder.load_state_dict(ckpt["target_encoder"])
print(f"Loaded epoch {ckpt['epoch']}")

tfm = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
trn = DataLoader(torchvision.datasets.CIFAR10("./data", True, transform=tfm), 512, num_workers=4)
tst = DataLoader(torchvision.datasets.CIFAR10("./data", False, transform=tfm), 512, num_workers=4)

model.eval()
def extract(loader):
    fs, ls = [], []
    for x, y in loader:
        fs.append(model.context_encoder(x.to(device)).mean(1).float().cpu())
        ls.append(y)
    return torch.cat(fs).to(device), torch.cat(ls).to(device)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    print("Extracting features...")
    trf, trl = extract(trn)
    tef, tel = extract(tst)
print(f"Features: train={trf.shape}, test={tef.shape}")

# Linear probe
clf = nn.Linear(trf.shape[1], 10).to(device)
opt = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
for _ in range(50):
    clf.train()
    perm = torch.randperm(len(trf), device=device)
    for i in range(0, len(trf), 1024):
        idx = perm[i:i + 1024]
        loss = F.cross_entropy(clf(trf[idx]), trl[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
clf.eval()
with torch.no_grad():
    acc = (clf(tef).argmax(1) == tel).float().mean().item() * 100
print(f"Linear probe: {acc:.2f}%")

# k-NN
tn, en = F.normalize(trf, dim=1), F.normalize(tef, dim=1)
for k in [1, 5, 20]:
    _, topk = (en @ tn.T).topk(k, dim=1)
    knn_acc = (trl[topk].mode(1).values == tel).float().mean().item() * 100
    print(f"k-NN k={k}: {knn_acc:.2f}%")
