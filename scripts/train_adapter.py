"""Train the domain-adaptation projection on the 29-species auxiliary corpus.

Supervised contrastive learning over BirdNET embeddings of Indian endemic
species (targets excluded by construction). A small held-out share of aux
recordings monitors convergence. The trained adapter is frozen thereafter.

Output: results/adapter.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from birdsong.models import AdapterProjection, supcon_loss  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIM = 256
EPOCHS = 40
BATCH = 512
SEED = 7


def main():
    z = np.load(REPO / "data" / "aux_embeddings.npz")
    X = torch.tensor(z["embeddings"])
    species = z["species"]
    classes = sorted(set(species))
    y = torch.tensor([classes.index(s) for s in species])
    rids = z["ids"]
    print(f"{len(X)} aux windows, {len(classes)} species")

    # recording-level holdout of ~10% for monitoring
    rng = np.random.default_rng(SEED)
    uniq = sorted(set(rids))
    hold = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
    is_hold = np.array([r in hold for r in rids])
    Xtr, ytr = X[~is_hold], y[~is_hold]
    Xho, yho = X[is_hold], y[is_hold]

    torch.manual_seed(SEED)
    model = AdapterProjection(out_dim=OUT_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 8:
                continue
            zb = model(Xtr[idx].to(DEVICE))
            loss = supcon_loss(zb, ytr[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            zh = model(Xho.to(DEVICE))
            lh = supcon_loss(zh, yho.to(DEVICE)).item()
        print(f"epoch {epoch:02d}  train {np.mean(losses):.4f}  holdout {lh:.4f}")

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "out_dim": OUT_DIM, "classes": classes},
               REPO / "results" / "adapter.pt")
    print("saved results/adapter.pt")


if __name__ == "__main__":
    main()
