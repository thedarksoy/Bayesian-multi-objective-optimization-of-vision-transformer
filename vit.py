"""
vit.py
------
ViT-Base baseline using timm's pre-built Vision Transformer.

Paper reference:
  Şahin et al., "Multi-objective optimization of ViT architecture for
  efficient brain tumor classification", BSPC 91 (2024) 105938

ViT-Base config (Table 9):
  patch_size=16 | dim=768 | heads=12 | depth=12 | mlp_dim=3072 | ~85.8M params

Install:
  pip install timm torch torchvision scikit-learn

NOTE on dataset split:
  get_data_loaders() uses Testing/ as the validation split throughout training,
  matching the paper's methodology exactly (Table 2 caption: "Brain Tumor
  validation dataset"). This is NOT a held-out test set — it is seen at every
  epoch. Results reported here are therefore validation-set results, consistent
  with the paper's Tables 2/6/8.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

import timm                          # pip install timm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────

def get_vit_base(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """
    ViT-Base from timm with paper's exact config (Table 9):
      patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0
      (mlp_dim = 768 * 4 = 3072)
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained  = pretrained,
        num_classes = num_classes,
        img_size    = 224,
    )
    return model


def build_vit(
    patch_size:  int,
    dim:         int,
    depth:       int,
    heads:       int,
    mlp_dim:     int,
    num_classes: int = 4,
    image_size:  int = 224,
) -> nn.Module:
    """
    Build an arbitrary ViT config from timm using the 5 hyperparameters
    searched by the paper (Fig. 1 / Section 3):
      patch_size, dim, depth, heads, mlp_dim

    IMPORTANT: timm requires dim % heads == 0.  The caller is responsible for
    ensuring divisibility — use snap_heads_to_dim() before calling this.
    """
    mlp_ratio = mlp_dim / dim          # timm uses ratio internally
    model = timm.create_model(
        "vit_base_patch16_224",        # base architecture template
        pretrained   = False,
        num_classes  = num_classes,
        img_size     = image_size,
        patch_size   = patch_size,
        embed_dim    = dim,
        depth        = depth,
        num_heads    = heads,
        mlp_ratio    = mlp_ratio,
    )
    return model


def snap_heads_to_dim(dim: int, heads: int, candidates: list[int]) -> int:
    """
    Return the largest value in `candidates` that evenly divides `dim`.
    If `heads` already divides `dim`, return it unchanged.

    Why this exists:
      timm raises AssertionError if dim % num_heads != 0.
      The paper's search space contains invalid (dim, heads) pairs
      (e.g. dim=512, heads=24 → 512%24=8≠0), so we snap to the nearest
      valid head count rather than crashing.

    The BMO-ViT paper best config (dim=512, heads=24) is itself invalid
    for standard ViT.  Nearest valid options: heads=16 (512/16=32 ✓)
    or use dim=768 (768/24=32 ✓).  We default to heads=16 for dim=512.
    """
    if dim % heads == 0:
        return heads
    valid = [h for h in sorted(candidates, reverse=True) if dim % h == 0]
    if valid:
        return valid[0]
    # absolute fallback: largest power of 2 that divides dim and ≤ dim
    for h in [16, 8, 4, 2, 1]:
        if dim % h == 0:
            return h
    return 1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Loss  –  Multi-Class Binary Cross-Entropy  (Equation 5 of paper)
# ──────────────────────────────────────────────────────────────────────────────

class MultiClassBCELoss(nn.Module):
    """
    L = -1/N  Σ_i Σ_c [ y_ic log(ŷ_ic) + (1-y_ic) log(1-ŷ_ic) ]

    Each class is treated as an independent binary problem (Section 2.3.1).
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(targets, self.num_classes).float()
        return self.bce(logits, one_hot)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def get_data_loaders(
    data_dir:    str,
    batch_size:  int = 64,
    num_workers: int = 4,
    image_size:  int = 224,
) -> tuple[DataLoader, DataLoader]:
    """
    Brain Tumor MRI dataset loaders.

    Expected layout (Kaggle masoudnickparvar/brain-tumor-mri-dataset):
        <data_dir>/Training/{glioma, meningioma, notumor, pituitary}/
        <data_dir>/Testing/ {glioma, meningioma, notumor, pituitary}/

    Paper: 5 712 training images, 1 311 test images, 224x224x3.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    root = Path(data_dir)
    train_ds = datasets.ImageFolder(str(root / "Training"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(root / "Testing"),  transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler=None,
) -> dict:
    """Single training epoch. Returns {loss, accuracy}."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def _quick_val_accuracy(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Lightweight per-epoch validation: accuracy only, no metric overhead.
    Used inside full_train() to avoid running full evaluate_model() every
    epoch (which was adding unnecessary inference-timing noise to the history).
    """
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        correct += (logits.argmax(1) == labels.to(device)).sum().item()
        total   += labels.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_model(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int = 4,
) -> dict:
    """
    Full evaluation returning all metrics from Tables 2 / 4 / 6 / 8:
      accuracy, f1_score, specificity, precision, recall, inference_time

    inference_time unit: seconds per iteration (s/it), matching Table 2/6/8
    paper values (~0.00255 – 0.00446 s/it).  Computed as elapsed / n_batches.
    NOT batches-per-second.
    """
    model.eval()
    all_preds, all_labels = [], []
    n_batches = 0

    t0 = time.perf_counter()
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        n_batches += 1
    elapsed = time.perf_counter() - t0

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy  = float((y_true == y_pred).mean())
    f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Specificity = TN / (TN + FP), macro-averaged across classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specs = []
    for c in range(num_classes):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)

    # FIX: inference_time = elapsed / n_batches  (seconds per iteration)
    # Paper values: ViT-Base ≈ 0.00446 s/it, BMO-ViT ≈ 0.00255 s/it (Table 2/6/8)
    # Previous code had n_batches/elapsed which gave batches/second (~20-80),
    # making the metric unrecognisable vs the paper.
    inference_time = elapsed / n_batches if n_batches > 0 else 0.0

    return {
        "accuracy":       accuracy,
        "f1_score":       f1,
        "specificity":    float(np.mean(specs)),
        "precision":      precision,
        "recall":         recall,
        "inference_time": inference_time,   # s/it  (matches paper Tables 2/6/8)
    }


def full_train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_epochs:     int   = 100,
    lr:           float = 1e-4,
    num_classes:  int   = 4,
    verbose:      bool  = True,
) -> list[dict]:
    """
    100-epoch full training loop (paper: Adam lr=1e-4, Section 3).

    FIX vs previous version:
      - Per-epoch validation now uses _quick_val_accuracy() (accuracy only),
        avoiding inference-timing noise and the ~10-15% overhead of running
        full evaluate_model() every epoch.
      - Full evaluate_model() is called by the caller (run_vit_base /
        run_bmo_vit) on the test split after training finishes.
      - Cosine annealing LR scheduler added (eta_min=1e-6) to help
        convergence over 100 epochs.
    """
    criterion = MultiClassBCELoss(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    scaler  = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    history = []

    for epoch in range(1, n_epochs + 1):
        tr  = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc = _quick_val_accuracy(model, val_loader, device)
        scheduler.step()

        history.append({"epoch": epoch, "loss": tr["loss"],
                         "train_acc": tr["accuracy"], "val_acc": val_acc})

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"loss={tr['loss']:.4f} | train_acc={tr['accuracy']:.4f} | "
                  f"val_acc={val_acc:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_vit_base(num_classes=4).to(device)
    n      = count_parameters(model)

    print("ViT-Base (timm)  –  patch=16 | dim=768 | heads=12 | depth=12 | mlp=3072")
    print(f"  Parameters : {n / 1e6:.1f} M   (paper: 85.8 M)")

    dummy = torch.randn(2, 3, 224, 224, device=device)
    out   = model(dummy)
    print(f"  Output     : {tuple(out.shape)}")

    # Verify inference_time unit is in the paper's range
    from torch.utils.data import TensorDataset
    ds     = TensorDataset(torch.randn(64, 3, 224, 224), torch.randint(0, 4, (64,)))
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    m      = evaluate_model(model, loader, device, num_classes=4)
    print(f"  inference_time: {m['inference_time']:.5f} s/it  "
          f"(paper ViT-Base ≈ 0.00446 s/it on RTX 3060 Ti)")
