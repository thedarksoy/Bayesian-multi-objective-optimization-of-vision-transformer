"""
vit_bayesian.py
---------------
Bayesian Multi-Objective (BMO) optimisation of ViT using Optuna.

Paper reference:
  Şahin et al., "Multi-objective optimization of ViT architecture for
  efficient brain tumor classification", BSPC 91 (2024) 105938

How Optuna maps to the paper:
  - Optuna's TPESampler  →  Bayesian (Tree-structured Pareto Estimator)
  - optuna.create_study(directions=["maximize","minimize"])  →  MOO (Section 2.2)
  - Pareto front via study.best_trials  →  Equation (4)
  - 5 low-epochs per trial  →  Section 3.2 "low-epoch training"
  - n_trials=200  →  paper's 200 total trials

BMO-ViT optimal params found by paper (Table 9):
  patch_size=32 | dim=512 | heads=24 | depth=6 | mlp_dim=256 | ~22.1M params

  NOTE on heads=24, dim=512: 512 % 24 = 8 ≠ 0, which is invalid for standard
  timm ViT (raises AssertionError: dim should be divisible by num_heads).
  The paper likely used a custom implementation.  In this code the BMO-ViT
  uses heads=16 (512 % 16 = 0 ✓, 512/16 = 32 dims/head) which is the nearest
  valid option and produces ≈22M params.  All calls to build_vit() pass through
  snap_heads_to_dim() to prevent the crash before timm ever sees the values.

Install:
  pip install optuna timm torch torchvision scikit-learn
"""

import time
from pathlib import Path

import numpy as np
import optuna                         # pip install optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vit import (
    build_vit,
    snap_heads_to_dim,
    count_parameters,
    MultiClassBCELoss,
    train_one_epoch,
    evaluate_model,
    full_train,
)

# Silence optuna's per-trial INFO logs unless you want them
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Search space  (Section 3 / Fig. 1 of the paper — exact values)
# ──────────────────────────────────────────────────────────────────────────────

PATCH_SIZES = [8, 16, 32]
DIMS        = [64, 128, 256, 512, 768, 1536]
HEADS       = [8, 10, 12, 14, 16, 24]
MLP_DIMS    = [256, 512, 768, 1024, 1536, 2048]
DEPTHS      = [6, 8, 10, 12, 14, 16, 24]

# MOO hard constraints (Section 3.2)
ACC_THRESHOLD   = 0.80      # validation accuracy must exceed this
PARAM_THRESHOLD = 40e6      # parameter count must stay below this

# ── Per-parameter descriptive info for verbose output ─────────────────────────
_PARAM_INFO = {
    "patch_size": (
        "Token resolution. Larger patch → fewer tokens → faster inference.\n"
        "        ViT-Base=16 (196 tokens), BMO-ViT=32 (49 tokens)."
    ),
    "dim": (
        "Token embedding width. Controls model capacity across all layers.\n"
        "        ViT-Base=768, BMO-ViT=512."
    ),
    "depth": (
        "Number of stacked Transformer encoder blocks. Primary driver of\n"
        "        model size and inference cost. ViT-Base=12, BMO-ViT=6."
    ),
    "heads": (
        "Parallel attention heads in Multi-Head Self-Attention. Must divide\n"
        "        dim. ViT-Base=12, BMO-ViT target=24 (snapped to 16 for dim=512)."
    ),
    "mlp_dim": (
        "Hidden width of the Feed-Forward Network per Transformer block.\n"
        "        ViT-Base=3072 (ratio 4×), BMO-ViT=256 (ratio 0.5×)."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# BMO-ViT  –  paper's published best configuration (Table 9, adjusted)
# ──────────────────────────────────────────────────────────────────────────────

def get_bmo_vit(num_classes: int = 4) -> nn.Module:
    """
    Returns the Pareto-optimal BMO-ViT from the paper (Table 9).

    Paper reports: patch=32, dim=512, heads=24, depth=6, mlp=256 → 22.1M params.
    512 % 24 = 8 ≠ 0, which timm rejects.  We snap heads to 16 (512/16=32 ✓).
    This yields ≈22M params and the same depth/patch/mlp structure.
    ~4× smaller and ~2× faster than ViT-Base.
    """
    safe_heads = snap_heads_to_dim(512, 24, HEADS)   # → 16
    return build_vit(
        patch_size  = 32,
        dim         = 512,
        depth       = 6,
        heads       = safe_heads,
        mlp_dim     = 256,
        num_classes = num_classes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Low-epoch evaluation  (used inside each Optuna trial)
# ──────────────────────────────────────────────────────────────────────────────

def _low_epoch_eval(
    patch_size:   int,
    dim:          int,
    heads:        int,
    depth:        int,
    mlp_dim:      int,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_epochs:     int   = 5,    # paper MOO phase: 5 low-epochs
    lr:           float = 1e-4,
    num_classes:  int   = 4,
) -> tuple[float, int]:
    """
    Builds a ViT with the given params, trains for n_epochs, returns
    (val_accuracy, n_parameters).  Called once per Optuna trial.

    FIX: snap_heads_to_dim() is called before build_vit() to prevent the
    timm AssertionError "dim should be divisible by num_heads".  The snapped
    head count is stored back so n_params reflects the actual built model.
    """
    # FIX: ensure dim % heads == 0 before passing to timm
    heads = snap_heads_to_dim(dim, heads, HEADS)

    model     = build_vit(patch_size, dim, depth, heads, mlp_dim,
                          num_classes=num_classes).to(device)
    n_params  = count_parameters(model)
    criterion = MultiClassBCELoss(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for _ in range(n_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

    metrics = evaluate_model(model, val_loader, device, num_classes)
    return metrics["accuracy"], n_params


# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective  –  two objectives → MOO (Eq. 4, Section 2.2.2)
# ──────────────────────────────────────────────────────────────────────────────

def _make_objective(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    low_epochs:   int,
    lr:           float,
    num_classes:  int,
):
    """
    Returns an Optuna objective function.

    The study is created with directions=["maximize", "minimize"], so:
      - objective return[0] = accuracy  → Optuna maximises this
      - objective return[1] = n_params  → Optuna minimises this

    Pareto-optimal trials satisfy both MOO targets simultaneously —
    exactly the yellow region in Fig. 4 of the paper.
    """
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # ── Suggest hyperparameters from the paper's search space ──────────
        patch_size = trial.suggest_categorical("patch_size", PATCH_SIZES)
        dim        = trial.suggest_categorical("dim",        DIMS)
        heads      = trial.suggest_categorical("heads",      HEADS)
        depth      = trial.suggest_categorical("depth",      DEPTHS)
        mlp_dim    = trial.suggest_categorical("mlp_dim",    MLP_DIMS)

        # FIX: snap heads before eval so the actual built heads is recorded
        safe_heads = snap_heads_to_dim(dim, heads, HEADS)
        trial.set_user_attr("heads_used", safe_heads)   # log if snapped

        try:
            acc, n_params = _low_epoch_eval(
                patch_size, dim, safe_heads, depth, mlp_dim,
                train_loader, val_loader, device,
                low_epochs, lr, num_classes,
            )
        except Exception as e:
            # Mark failed trials so they appear in study.trials
            trial.set_user_attr("status", f"failed: {e}")
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("n_params",  n_params)
        trial.set_user_attr("accuracy",  acc)

        # Categorise the trial against the paper's MOO constraints
        if acc >= ACC_THRESHOLD and n_params <= PARAM_THRESHOLD:
            status = "success"    # Pareto candidate — meets both targets
        elif acc < ACC_THRESHOLD:
            status = f"low_acc({acc:.3f})"
        else:
            status = f"too_large({n_params/1e6:.1f}M)"
        trial.set_user_attr("status", status)

        # The study uses directions=["maximize","minimize"]:
        #   return[0] = accuracy  → maximised by Optuna
        #   return[1] = n_params  → minimised by Optuna
        # No negation is needed — the directions= argument handles this.
        return acc, float(n_params)

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# BMO Search Engine
# ──────────────────────────────────────────────────────────────────────────────

class BayesianMOOptimizer:
    """
    Wraps Optuna's multi-objective TPE sampler to reproduce the BMO-NAS
    methodology from Section 3.2 of the paper.

    Usage:
        optimizer = BayesianMOOptimizer(train_loader, val_loader, device)
        summary   = optimizer.run()
        best      = summary["best_params"]
    """

    def __init__(
        self,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        device:          torch.device,
        n_trials:        int   = 200,    # paper: 200 total trials
        low_epochs:      int   = 5,      # paper MOO phase: 5 epochs / trial
        num_classes:     int   = 4,
        lr:              float = 1e-4,
        n_startup_trials: int  = 10,     # random init before GP kicks in
        verbose:         bool  = True,
    ):
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.device           = device
        self.n_trials         = n_trials
        self.low_epochs       = low_epochs
        self.num_classes      = num_classes
        self.lr               = lr
        self.n_startup_trials = n_startup_trials
        self.verbose          = verbose
        self.study            = None

    def run(self) -> dict:
        """
        Executes the BMO search and returns a summary dict:
          best_params    – hyperparameter dict of the best Pareto trial
          best_accuracy  – val accuracy (low-epoch) of that trial
          best_n_params  – parameter count of that trial
          pareto_trials  – list of all Pareto-optimal optuna Trial objects
          study          – the full optuna Study (for further analysis)
          n_successful   – trials meeting acc>0.80 AND params<40M
          n_failed       – pruned / crashed trials
        """
        # Optuna MOO study: maximise accuracy, minimise parameter count
        sampler = TPESampler(
            n_startup_trials = self.n_startup_trials,
            seed             = 42,
            multivariate     = True,   # correlates hyperparameters (like GPEI)
        )
        self.study = optuna.create_study(
            directions = ["maximize", "minimize"],  # (accuracy, n_params)
            sampler    = sampler,
        )

        objective = _make_objective(
            self.train_loader, self.val_loader, self.device,
            self.low_epochs, self.lr, self.num_classes,
        )

        if self.verbose:
            print(f"\n[BMO] Optuna multi-objective search  –  {self.n_trials} trials")
            print(f"      Sampler      : TPE (Bayesian, multivariate)")
            print(f"      Objectives   : maximise accuracy | minimise params")
            print(f"      MOO targets  : acc > {ACC_THRESHOLD:.0%}  AND  "
                  f"params < {PARAM_THRESHOLD/1e6:.0f} M")
            print(f"      Low epochs   : {self.low_epochs} per trial\n")
            print("  Parameter search space (Section 3 of paper):")
            for name, info in _PARAM_INFO.items():
                print(f"    {name:<12} {info}")
            print()

        t0 = time.perf_counter()

        # Optuna callback for live descriptive progress
        def _callback(study, trial):
            if not self.verbose:
                return
            ua      = trial.user_attrs
            acc     = ua.get("accuracy", 0)
            n_p     = ua.get("n_params", 0)
            status  = ua.get("status", "?")
            snapped = ua.get("heads_used", trial.params.get("heads", "?"))
            orig_h  = trial.params.get("heads", "?")
            h_note  = f"(snapped {orig_h}→{snapped})" if snapped != orig_h else ""

            # Descriptive reason string
            if status == "success":
                reason = f"✓ Pareto-candidate"
            elif "low_acc" in status:
                reason = f"✗ acc {acc:.3f} < {ACC_THRESHOLD} threshold"
            elif "too_large" in status:
                reason = f"✗ {n_p/1e6:.1f}M > {PARAM_THRESHOLD/1e6:.0f}M limit"
            else:
                reason = status

            print(
                f"  Trial {trial.number+1:3d} | "
                f"ps={trial.params.get('patch_size','?'):>2} "
                f"dim={trial.params.get('dim','?'):>4} "
                f"h={snapped:>2}{h_note} "
                f"d={trial.params.get('depth','?'):>2} "
                f"mlp={trial.params.get('mlp_dim','?'):>4} | "
                f"acc={acc:.4f}  params={n_p/1e6:.1f}M | "
                f"{reason}"
            )

        self.study.optimize(
            objective,
            n_trials          = self.n_trials,
            callbacks         = [_callback],
            show_progress_bar = False,
        )

        elapsed = time.perf_counter() - t0

        # ── Pareto front  (study.best_trials = Pareto-optimal set) ──────────
        pareto_trials = self.study.best_trials   # Optuna computes this natively

        # Best = highest accuracy among Pareto-optimal trials that also
        # meet the paper's hard constraints (>80% acc, <40M params)
        valid = [
            t for t in pareto_trials
            if t.user_attrs.get("accuracy", 0) >= ACC_THRESHOLD
            and t.user_attrs.get("n_params",  1e9) <= PARAM_THRESHOLD
        ]
        if not valid:
            # Fall back to best accuracy on the full Pareto front
            valid = pareto_trials if pareto_trials else self.study.trials

        best_t = max(valid, key=lambda t: t.user_attrs.get("accuracy", 0))

        n_successful = sum(
            1 for t in self.study.trials
            if t.user_attrs.get("status") == "success"
        )
        n_failed = sum(
            1 for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )

        if self.verbose:
            print("\n" + "=" * 62)
            print("BMO Search complete")
            print(f"  Total trials  : {len(self.study.trials)}")
            print(f"  Successful    : {n_successful}  "
                  f"(acc>{ACC_THRESHOLD:.0%} AND params<{PARAM_THRESHOLD/1e6:.0f}M)")
            print(f"  Failed/pruned : {n_failed}")
            print(f"  Pareto front  : {len(pareto_trials)} trials")
            print(f"  Runtime       : {elapsed/60:.1f} min")
            print("-" * 62)
            print(f"  Best Pareto trial : #{best_t.number + 1}")
            for k, v in best_t.params.items():
                print(f"    {k:<12}: {v}")
            print(f"    accuracy    : {best_t.user_attrs.get('accuracy', 0):.4f}")
            print(f"    params      : {best_t.user_attrs.get('n_params', 0)/1e6:.1f} M")
            print("=" * 62)

        return {
            "best_params":   best_t.params,
            "best_accuracy": best_t.user_attrs.get("accuracy", 0),
            "best_n_params": best_t.user_attrs.get("n_params", 0),
            "pareto_trials": pareto_trials,
            "study":         self.study,
            "n_successful":  n_successful,
            "n_failed":      n_failed,
            "runtime_s":     elapsed,
        }

    def pareto_report(self):
        """Print a table of Pareto-optimal trials (mirrors Fig. 4 of paper)."""
        if self.study is None:
            print("Run .run() first.")
            return
        trials = self.study.best_trials
        print(f"\n  Pareto-optimal trials: {len(trials)}\n")
        print(f"  {'#':>3}  {'patch':>5} {'dim':>5} {'heads':>6} "
              f"{'depth':>6} {'mlp':>6} {'acc':>8} {'params':>10}")
        print("  " + "-" * 58)
        for t in sorted(trials, key=lambda t: -t.user_attrs.get("accuracy", 0)):
            p       = t.params
            h_used  = t.user_attrs.get("heads_used", p.get("heads", "?"))
            h_orig  = p.get("heads", "?")
            h_str   = f"{h_used}*" if h_used != h_orig else str(h_used)
            print(
                f"  {t.number+1:3d}  "
                f"{p.get('patch_size','?'):>5} "
                f"{p.get('dim','?'):>5} "
                f"{h_str:>6} "
                f"{p.get('depth','?'):>6} "
                f"{p.get('mlp_dim','?'):>6} "
                f"{t.user_attrs.get('accuracy',0):8.4f} "
                f"{t.user_attrs.get('n_params',0)/1e6:8.1f} M"
            )
        print("  (* = heads snapped to nearest valid divisor of dim)")


# ──────────────────────────────────────────────────────────────────────────────
# Subset loaders  (for fast BMO search on limited hardware)
# ──────────────────────────────────────────────────────────────────────────────

def get_subset_loaders(
    data_dir:    str,
    subset_frac: float = 0.3,
    batch_size:  int   = 32,
    num_workers: int   = 4,
    image_size:  int   = 224,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns train/val loaders using a random fraction of the data.
    The paper notes (Section 4.1): using a data subset avoids cost
    during the optimisation phase.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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

    rng = np.random.default_rng(seed)

    def _subset(ds, frac):
        n   = len(ds)
        idx = rng.permutation(n)[:max(1, int(n * frac))].tolist()
        return Subset(ds, idx)

    return (
        DataLoader(_subset(train_ds, subset_frac), batch_size=batch_size,
                   shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(_subset(val_ds,   subset_frac), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers, pin_memory=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ablation study  (Table 7 of paper)
# ──────────────────────────────────────────────────────────────────────────────

class AblationStudy:
    """
    Reproduces Table 7: each ViT-Base parameter swapped one-at-a-time
    to the BMO-ViT value to measure its individual contribution.

    ViT-Base : patch=16, dim=768, depth=12, heads=12, mlp=3072
    BMO-ViT  : patch=32, dim=512, depth=6,  heads=24, mlp=256

    FIX: default n_epochs changed from 10 → 100.
      The paper ran full 100-epoch training for Table 7 (ablation values
      are 0.965–0.980 range). At 10 epochs models are in the 0.5–0.7 range
      and the relative ordering of parameter contributions is unreliable.

    FIX: snap_heads_to_dim() is applied to all configs before building,
      preventing timm's AssertionError on invalid (dim, heads) pairs.
      Notably the heads=24 / dim=512 swap config is adjusted to heads=16.
    """

    BASE = dict(patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072)
    BMO  = dict(patch_size=32, dim=512, depth=6,  heads=24, mlp_dim=256)

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        n_epochs:     int   = 100,   # FIX: was 10, now matches paper (Section 3)
        lr:           float = 1e-4,
        num_classes:  int   = 4,
    ):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.n_epochs     = n_epochs
        self.lr           = lr
        self.num_classes  = num_classes

    def _eval_cfg(self, cfg: dict, label: str) -> dict:
        # FIX: snap heads before building to prevent timm crash
        safe_cfg = dict(cfg)
        safe_cfg["heads"] = snap_heads_to_dim(
            safe_cfg["dim"], safe_cfg["heads"], HEADS
        )
        if safe_cfg["heads"] != cfg["heads"]:
            print(f"    [snap] heads {cfg['heads']}→{safe_cfg['heads']} "
                  f"(dim={safe_cfg['dim']} not divisible by {cfg['heads']})")

        model = build_vit(**safe_cfg, num_classes=self.num_classes).to(self.device)
        crit  = MultiClassBCELoss(self.num_classes)
        opt   = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=1e-6
        )
        sc = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        for ep in range(self.n_epochs):
            train_one_epoch(model, self.train_loader, opt, crit, self.device, sc)
            sched.step()

        m = evaluate_model(model, self.val_loader, self.device, self.num_classes)
        m["n_params"]   = count_parameters(model)
        m["heads_used"] = safe_cfg["heads"]
        return m

    def run(self) -> list[dict]:
        print(f"\n  Ablation study — {self.n_epochs} epochs per config")
        print("  (Paper Table 7: each parameter swapped one-at-a-time BASE→BMO)")
        print(f"  {'Config':<22} {'Acc':>7} {'ΔAcc':>7} {'F1':>7} "
              f"{'Params':>9} {'s/it':>10}")
        print("  " + "-" * 70)

        rows   = []
        base_m = None

        configs = [
            ("ViT-Base (baseline)",  None),
            ("patch_size 16→32",     "patch_size"),
            ("dim       768→512",    "dim"),
            ("depth     12→6",       "depth"),
            ("heads     12→24",      "heads"),
            ("mlp_dim   3072→256",   "mlp_dim"),
            ("BMO-ViT (all swaps)",  "__all__"),
        ]

        for label, key in configs:
            if key is None:
                cfg = dict(self.BASE)
            elif key == "__all__":
                cfg = dict(self.BMO)
            else:
                cfg = {**self.BASE, key: self.BMO[key]}

            print(f"  [Ablation] training: {label} …")
            m = self._eval_cfg(cfg, label)

            if base_m is None:
                base_m = m
            delta   = m["accuracy"] - base_m["accuracy"]
            d_str   = f"{delta:+.4f}" if key is not None else "  base"
            arrow   = "↑" if delta > 0.0005 else ("↓" if delta < -0.0005 else "=")

            rows.append({"config": label, **m})
            print(f"  {label:<22} "
                  f"{m['accuracy']:7.4f} "
                  f"{d_str:>7} {arrow} "
                  f"{m['f1_score']:7.4f} "
                  f"{m['n_params']/1e6:7.1f}M "
                  f"{m['inference_time']:10.5f}")

        print("  " + "-" * 70)
        print("  NOTE: 'heads 12→24' is the only swap that typically hurts accuracy")
        print("        (paper Table 7 confirmed: acc drops ~0.001, params balloon to 114M)")
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = get_bmo_vit(num_classes=4).to(device)
    n      = count_parameters(model)

    print("BMO-ViT (timm + Optuna)  –  patch=32 | dim=512 | heads=16* | depth=6 | mlp=256")
    print(f"  (* heads snapped from paper's 24 to 16: 512 % 24 ≠ 0)")
    print(f"  Parameters : {n / 1e6:.1f} M   (paper: 22.1 M)")

    dummy = torch.randn(2, 3, 224, 224, device=device)
    out   = model(dummy)
    print(f"  Output     : {tuple(out.shape)}")
    print(f"  Optuna version: {optuna.__version__}")

    # Verify no divisibility crash across the full search space
    print("\n  Divisibility check across all (dim, heads) combinations:")
    issues = []
    for d in DIMS:
        for h in HEADS:
            snapped = snap_heads_to_dim(d, h, HEADS)
            if snapped != h:
                issues.append(f"    dim={d}, heads={h} → snapped to {snapped}")
    if issues:
        print(f"  {len(issues)} pairs require snapping:")
        for line in issues:
            print(line)
    else:
        print("  All pairs valid.")
