"""Model definitions: embedding probe, gated-attention MIL, adaptation projector, CNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 1024


class MLPProbe(nn.Module):
    """Window-level classifier on frozen embeddings (Baseline B)."""

    def __init__(self, n_classes: int, in_dim: int = EMB_DIM, hidden: int = 256, p_drop: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class GatedAttentionMIL(nn.Module):
    """Recording-level classifier over a bag of window embeddings.

    Ilse et al. (2018) gated attention pooling: each window k gets weight
    a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k))); the bag representation
    is the attention-weighted sum, classified by a linear head. Attention
    weights expose which windows carried the species-diagnostic signal.
    """

    def __init__(self, n_classes: int, in_dim: int = EMB_DIM, hidden: int = 256,
                 attn_dim: int = 128, p_drop: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
        )
        self.attn_V = nn.Linear(hidden, attn_dim)
        self.attn_U = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, bag, mask=None):
        """bag: (B, K, D) padded bags; mask: (B, K) True where valid."""
        h = self.encoder(bag)                                   # (B, K, H)
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))  # (B, K, 1)
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(a, dim=1)
        z = (a * h).sum(dim=1)                                  # (B, H)
        return self.head(z), a.squeeze(-1)


class AdapterProjection(nn.Module):
    """Residual projection head adapting BirdNET space to Indian endemic avifauna.

    Trained with supervised contrastive loss on the 29-species auxiliary corpus
    (never on the 4 target species), then frozen for target-species classifiers.
    """

    def __init__(self, in_dim: int = EMB_DIM, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return F.normalize(self.proj(x) + self.skip(x), dim=-1)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalized z."""
    sim = z @ z.T / temp
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, float("-inf"))
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos.sum(1).clamp(min=1)
    # masked_fill (not multiply) so -inf entries outside the positive set can't
    # poison the sum with 0 * -inf = NaN
    loss = -log_prob.masked_fill(~pos, 0.0).sum(1) / pos_counts
    return loss[pos.sum(1) > 0].mean()


def build_cnn(n_classes: int):
    """EfficientNet-B0 on 1-channel log-mels (Baseline A, done right)."""
    import timm
    return timm.create_model("efficientnet_b0", pretrained=True,
                             in_chans=1, num_classes=n_classes, drop_rate=0.3)
