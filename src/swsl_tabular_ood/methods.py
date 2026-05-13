from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPClassifier, clone_model


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.15
    ssl_weight: float = 0.2
    mask_prob: float = 0.2
    device: str = "cpu"


@dataclass
class TTAConfig:
    steps: int = 20
    batch_size: int = 512
    lr: float = 5e-4
    entropy_weight: float = 1.0
    consistency_weight: float = 1.0
    stats_weight: float = 0.05
    mask_prob: float = 0.1
    update: str = "bn"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MaskedFeatureAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mask_features(x: torch.Tensor, mask_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(x) < mask_prob
    corrupted = x.masked_fill(mask, 0.0)
    return corrupted, mask.float()


def train_supervised(x: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> MLPClassifier:
    set_seed(cfg.seed)
    num_classes = int(np.max(y)) + 1
    model = MLPClassifier(x.shape[1], cfg.hidden_dim, cfg.depth, cfg.dropout, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_masked_ssl(x: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> tuple[MLPClassifier, MaskedFeatureAutoencoder]:
    """Train classifier with a tabular masked-reconstruction auxiliary signal.

    The self-supervised term reconstructs only intentionally masked feature cells,
    preserving column identity and avoiding image-style exchangeability assumptions.
    """

    set_seed(cfg.seed)
    num_classes = int(np.max(y)) + 1
    model = MLPClassifier(x.shape[1], cfg.hidden_dim, cfg.depth, cfg.dropout, num_classes).to(cfg.device)
    decoder = MaskedFeatureAutoencoder(x.shape[1], cfg.hidden_dim).to(cfg.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    for _ in range(cfg.epochs):
        model.train()
        decoder.train()
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            corrupted, mask = _mask_features(xb, cfg.mask_prob)
            logits = model(corrupted)
            recon = decoder(corrupted)
            denom = mask.sum().clamp_min(1.0)
            ssl_loss = (((recon - xb) ** 2) * mask).sum() / denom
            loss = F.cross_entropy(logits, yb) + cfg.ssl_weight * ssl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, decoder


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _source_feature_stats(model: MLPClassifier, x: np.ndarray, cfg: TTAConfig) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=cfg.batch_size, shuffle=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            _, z = model(xb.to(cfg.device), return_features=True)
            feats.append(z)
    z_all = torch.cat(feats, dim=0)
    return z_all.mean(dim=0), z_all.var(dim=0, unbiased=False)


def _select_tta_parameters(model: nn.Module, update: str) -> list[nn.Parameter]:
    if update == "all":
        return list(model.parameters())
    if update == "head":
        return list(model.classifier.parameters())
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            params.extend([module.weight, module.bias])
    return params


def adapt_frc_tta(source_model: MLPClassifier, x_source: np.ndarray, x_target: np.ndarray, cfg: TTAConfig) -> MLPClassifier:
    """Feature-Reliability Consistent TTA (FRC-TTA).

    Uses unlabeled target features only. It updates a chosen parameter subset by
    minimizing (i) entropy, (ii) consistency under column-wise masking, and
    (iii) a weak feature-statistics anchor to prevent target overfitting.
    """

    model = clone_model(source_model).to(cfg.device)
    source_mean, source_var = _source_feature_stats(model, x_source, cfg)
    params = _select_tta_parameters(model, cfg.update)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_target)), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(cfg.steps):
        for (xb,) in loader:
            xb = xb.to(cfg.device)
            corrupted, _ = _mask_features(xb, cfg.mask_prob)
            logits, z = model(xb, return_features=True)
            logits_aug = model(corrupted)
            probs = logits.softmax(dim=1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            consistency = F.kl_div(
                F.log_softmax(logits_aug, dim=1),
                probs.detach(),
                reduction="batchmean",
            )
            batch_mean = z.mean(dim=0)
            batch_var = z.var(dim=0, unbiased=False)
            stats = F.mse_loss(batch_mean, source_mean.detach()) + F.mse_loss(batch_var, source_var.detach())
            loss = cfg.entropy_weight * entropy + cfg.consistency_weight * consistency + cfg.stats_weight * stats
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
