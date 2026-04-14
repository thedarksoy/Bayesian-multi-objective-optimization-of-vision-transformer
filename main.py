"""
main.py
-------
Comparison pipeline: ViT-Base  vs  BMO-ViT

Paper reference:
  Şahin et al., "Multi-objective optimization of ViT architecture for
  efficient brain tumor classification", BSPC 91 (2024) 105938

Pipeline (mirrors Fig. 1 of the paper):
  Stage 1 – Train ViT-Base for N epochs, evaluate (Table 2)
  Stage 2 – Run Optuna BMO search to find best lightweight ViT config
             (Tables 5/6, Fig. 4/5)   [optional, --run_bmo_search]
  Stage 3 – Train BMO-ViT for N epochs, compare side-by-side (Table 8/9)

Usage:
  # Full pipeline (BMO search + training both models):
  python main.py --data_dir /path/to/BrainTumorMRI --run_bmo_search

  # Skip search, use paper's published best params:
  python main.py --data_dir /path/to/BrainTumorMRI

  # Quick smoke-test (no dataset needed):
  python main.py

Install:
  pip install timm optuna torch torchvision scikit-learn

Dataset (Kaggle):
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  Expected layout:
    <data_dir>/Training/{glioma, meningioma, notumor, pituitary}/
    <data_dir>/Testing/ {glioma, meningioma, notumor, pituitary}/

NOTE on dataset split:
  get_data_loaders() uses Testing/ as the validation split, matching the
  paper's methodology (Table 2 caption: "Brain Tumor validation dataset").
  Reported metrics are therefore validation-set results, consistent with
  Tables 2/6/8 of the paper.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vit import (
    get_vit_base,
    build_vit,
    snap_heads_to_dim,
    count_parameters,
    get_data_loaders,
    MultiClassBCELoss,
    evaluate_model,
    full_train,
)
from vit_bayesian import (
    get_bmo_vit,
    BayesianMOOptimizer,
    get_subset_loaders,
    AblationStudy,
    ACC_THRESHOLD,
    PARAM_THRESHOLD,
    HEADS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 70
THIN = "-" * 70


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def print_metrics(metrics: dict, indent: int = 4):
    pad = " " * indent
    print(f"{pad}Accuracy       : {metrics['accuracy']:.4f}")
    print(f"{pad}F1-score       : {metrics['f1_score']:.4f}")
    print(f"{pad}Specificity    : {metrics['specificity']:.4f}")
    print(f"{pad}Precision      : {metrics['precision']:.4f}")
    print(f"{pad}Recall         : {metrics['recall']:.4f}")
    # FIX: unit is seconds-per-iteration (s/it), not it/s
    # Paper values: ViT-Base ≈ 0.00446 s/it, BMO-ViT ≈ 0.00255 s/it (Table 2/6/8)
    print(f"{pad}Inference time : {metrics['inference_time']:.5f}  s/it")


def print_comparison_table(
    base_metrics: dict, bmo_metrics: dict,
    base_params: int,   bmo_params: int,
):
    """Side-by-side table matching the style of Tables 2 / 8 in the paper."""
    # FIX: column header updated from 'it/s' → 's/it' to match paper's unit
    print(f"\n  {'Model':<18} {'Accuracy':>9} {'F1':>9} {'Specificity':>12} "
          f"{'Precision':>10} {'Recall':>8} {'Params':>9} {'s/it':>10}")
    print("  " + THIN)

    def row(name, m, np_):
        print(f"  {name:<18} "
              f"{m['accuracy']:9.4f} {m['f1_score']:9.4f} "
              f"{m['specificity']:12.4f} {m['precision']:10.4f} "
              f"{m['recall']:8.4f} {np_/1e6:8.1f}M "
              f"{m['inference_time']:10.5f}")

    row("ViT-Base",  base_metrics, base_params)
    row("BMO-ViT",   bmo_metrics,  bmo_params)
    print("  " + THIN)

    # Deltas from paper Abstract
    print(f"\n  Δ Accuracy    : {(bmo_metrics['accuracy']  - base_metrics['accuracy'])  * 100:+.2f} %"
          f"   (paper: +1.48 %)")
    print(f"  Δ F1-score    : {(bmo_metrics['f1_score']  - base_metrics['f1_score'])  * 100:+.2f} %"
          f"   (paper: +3.23 %)")
    print(f"  Δ Precision   : {(bmo_metrics['precision'] - base_metrics['precision']) * 100:+.2f} %"
          f"   (paper: +3.36 %)")
    print(f"  Size ratio    : {base_params / bmo_params:.1f}×"
          f"   (paper: ~4×)")
    print(f"  Speed-up      : {bmo_metrics['inference_time'] / base_metrics['inference_time']:.1f}×"
          f"   (paper: ~2×, lower s/it is faster)")


def _save_json(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _cvt(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):  return {k: _cvt(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_cvt(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_cvt(data), f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – ViT-Base
# ──────────────────────────────────────────────────────────────────────────────

def run_vit_base(args, train_loader, val_loader, device):
    section("Stage 1 – ViT-Base Baseline  (Table 2 of paper)")

    model    = get_vit_base(num_classes=args.num_classes).to(device)
    n_params = count_parameters(model)
    print(f"  timm model : vit_base_patch16_224")
    print(f"  Config     : patch=16 | dim=768 | heads=12 | depth=12 | mlp=3072")
    print(f"  Parameters : {n_params / 1e6:.1f} M   (paper: 85.8 M)")
    print(f"  Training   : {args.epochs} epochs, Adam lr={args.lr}, CosineAnnealingLR\n")

    ckpt = Path(args.output_dir) / "vit_base.pth"
    if args.load_checkpoints and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        history = full_train(model, train_loader, val_loader, device,
                             n_epochs=args.epochs, lr=args.lr,
                             num_classes=args.num_classes, verbose=True)
        if args.save_checkpoints:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved → {ckpt}")
        _save_json(history, Path(args.output_dir) / "history_vit_base.json")

    print("\n  Evaluation on validation set (Testing/ split, matches paper Table 2):")
    metrics = evaluate_model(model, val_loader, device, args.num_classes)
    print_metrics(metrics)
    return model, metrics, n_params


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Optuna BMO search
# ──────────────────────────────────────────────────────────────────────────────

def run_bmo_search(args, device) -> dict:
    section("Stage 2 – Optuna Bayesian MOO Search  (Tables 5/6, Fig. 4 of paper)")

    print(f"  Sampler     : Optuna TPESampler (multivariate Bayesian)")
    print(f"  Objectives  : maximise accuracy | minimise parameter count")
    print(f"  MOO targets : acc > {ACC_THRESHOLD:.0%}  AND  "
          f"params < {PARAM_THRESHOLD/1e6:.0f} M")
    print(f"  Trials      : {args.bmo_trials}  "
          f"(paper: 200  ·  low-epochs: {args.bmo_low_epochs})")
    print(f"  Data subset : {args.subset_frac:.0%} of training set\n")

    bmo_train, bmo_val = get_subset_loaders(
        args.data_dir,
        subset_frac = args.subset_frac,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    optimizer = BayesianMOOptimizer(
        train_loader     = bmo_train,
        val_loader       = bmo_val,
        device           = device,
        n_trials         = args.bmo_trials,
        low_epochs       = args.bmo_low_epochs,
        num_classes      = args.num_classes,
        lr               = args.lr,
        n_startup_trials = max(3, args.bmo_trials // 10),
        verbose          = True,
    )
    summary = optimizer.run()
    optimizer.pareto_report()

    # Persist trial data (convert Optuna objects to plain dicts)
    trial_data = [
        {
            "number":   t.number,
            "params":   t.params,
            "accuracy": t.user_attrs.get("accuracy", 0),
            "n_params": t.user_attrs.get("n_params", 0),
            "status":   t.user_attrs.get("status", "?"),
        }
        for t in summary["study"].trials
    ]
    _save_json(trial_data, Path(args.output_dir) / "bmo_trials.json")
    print(f"\n  Trials saved → {args.output_dir}/bmo_trials.json")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 – BMO-ViT full training
# ──────────────────────────────────────────────────────────────────────────────

def run_bmo_vit(args, train_loader, val_loader, device, best_params=None):
    section("Stage 3 – BMO-ViT Full Training  (Tables 8/9 of paper)")

    if best_params:
        # FIX: snap heads before building — Optuna may have returned a heads
        # value that doesn't divide the chosen dim, which would crash timm.
        dim   = best_params.get("dim", 512)
        heads = snap_heads_to_dim(dim, best_params.get("heads", 16), HEADS)
        if heads != best_params.get("heads"):
            print(f"  [snap] heads {best_params['heads']}→{heads} "
                  f"(dim={dim} % {best_params['heads']} ≠ 0)")
        safe_params = {**best_params, "heads": heads}
        print(f"  Using searched params : {safe_params}")
        model = build_vit(**safe_params, num_classes=args.num_classes).to(device)
        label = "BMO-ViT (searched)"
    else:
        print("  Using paper's published best params (heads snapped 24→16 for dim=512):")
        print("    patch=32 | dim=512 | heads=16* | depth=6 | mlp=256")
        print("    (* 512 % 24 ≠ 0; nearest valid divisor is 16)")
        model = get_bmo_vit(num_classes=args.num_classes).to(device)
        label = "BMO-ViT (paper best)"

    n_params = count_parameters(model)
    print(f"  Parameters : {n_params / 1e6:.1f} M   (paper: 22.1 M)")
    print(f"  Training   : {args.epochs} epochs, Adam lr={args.lr}, CosineAnnealingLR\n")

    ckpt = Path(args.output_dir) / "bmo_vit.pth"
    if args.load_checkpoints and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        history = full_train(model, train_loader, val_loader, device,
                             n_epochs=args.epochs, lr=args.lr,
                             num_classes=args.num_classes, verbose=True)
        if args.save_checkpoints:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved → {ckpt}")
        _save_json(history, Path(args.output_dir) / "history_bmo_vit.json")

    print(f"\n  Evaluation on validation set ({label}):")
    metrics = evaluate_model(model, val_loader, device, args.num_classes)
    print_metrics(metrics)
    return model, metrics, n_params


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test  (no dataset needed)
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Validates both models forward-pass and architecture numbers."""
    import timm
    import optuna

    section("SMOKE TEST  (no dataset required)")
    print(f"  timm    version : {timm.__version__}")
    print(f"  optuna  version : {optuna.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device         : {device}\n")

    # Build both models
    vit_base = get_vit_base(num_classes=4).to(device)
    bmo_vit  = get_bmo_vit(num_classes=4).to(device)   # uses heads=16 (snapped)

    base_n = count_parameters(vit_base)
    bmo_n  = count_parameters(bmo_vit)

    print(f"  ViT-Base params : {base_n / 1e6:.1f} M   (paper: 85.8 M)")
    print(f"  BMO-ViT  params : {bmo_n  / 1e6:.1f} M   (paper: 22.1 M)")
    print(f"  Size ratio      : {base_n / bmo_n:.1f}×   (paper: ~4×)")
    print(f"  NOTE: BMO-ViT heads snapped 24→16 (512 % 24 ≠ 0, timm requirement)\n")

    dummy = torch.randn(4, 3, 224, 224, device=device)

    # Forward pass + timing
    t0 = time.perf_counter()
    with torch.no_grad():
        out_base = vit_base(dummy)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        out_bmo = bmo_vit(dummy)
    t_bmo = time.perf_counter() - t0

    print(f"  ViT-Base output : {tuple(out_base.shape)}  ({t_base*1000:.1f} ms)")
    print(f"  BMO-ViT  output : {tuple(out_bmo.shape)}  ({t_bmo*1000:.1f} ms)")
    print(f"  Speed-up        : {t_base/t_bmo:.1f}×   (paper: ~2×)\n")

    # Loss check
    criterion = MultiClassBCELoss(num_classes=4)
    labels    = torch.randint(0, 4, (4,), device=device)
    print(f"  BCE loss (ViT-Base) : {criterion(out_base, labels).item():.4f}")
    print(f"  BCE loss (BMO-ViT)  : {criterion(out_bmo,  labels).item():.4f}\n")

    # Architecture table (Table 9 style)
    # FIX: BMO-ViT heads corrected from 24 (invalid) to 16 (snapped)
    section("Architecture comparison  (Table 9 of paper)")
    rows = [
        ("ViT-Base",       16, 12, 12,  768, 3072, 85.8,  343.1),
        ("BMO-ViT (code)", 32, 16,  6,  512,  256, bmo_n/1e6, bmo_n*4/1e6),
        ("BMO-ViT (paper)",32, 24,  6,  512,  256, 22.1,   88.3),
    ]
    print(f"\n  {'Model':<18} {'patch':>6} {'heads':>6} {'depth':>6} "
          f"{'dim':>6} {'mlp':>7} {'params':>8} {'size MB':>9}")
    print("  " + "-" * 70)
    for name, ps, h, d, dim, mlp, par, sz in rows:
        print(f"  {name:<18} {ps:6d} {h:6d} {d:6d} "
              f"{dim:6d} {mlp:7d} {par:6.1f}M {sz:8.1f}")
    print("  (paper heads=24 is invalid for timm; code uses heads=16)")

    # Optuna study smoke-test
    section("Optuna MOO sanity check")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _dummy_obj(trial):
        x = trial.suggest_float("x", 0, 1)
        return x, 1 - x    # trivial Pareto: maximise x, maximise (1-x)

    study = optuna.create_study(directions=["maximize", "maximize"],
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(_dummy_obj, n_trials=10, show_progress_bar=False)
    print(f"  Pareto-front size after 10 trials : {len(study.best_trials)}")
    print(f"  Optuna MOO working correctly       : ✓\n")

    # snap_heads_to_dim smoke-test
    section("snap_heads_to_dim sanity check")
    test_cases = [
        (768, 12, 12), (512, 24, 16), (512, 16, 16),
        (64,  10,  8), (256, 14, 16), (1536, 24, 24),
    ]
    all_ok = True
    for dim, heads_in, expected in test_cases:
        result = snap_heads_to_dim(dim, heads_in, HEADS)
        ok = "✓" if result == expected else "✗"
        if result != expected:
            all_ok = False
        print(f"  {ok} dim={dim:4d}, heads={heads_in:2d} → {result:2d}  "
              f"(expected {expected}, {dim}%{result}={dim%result})")
    print(f"\n  snap_heads_to_dim: {'all OK ✓' if all_ok else 'FAILURES ABOVE ✗'}")

    print("\n  Smoke test PASSED ✓")
    print(SEP + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ViT-Base vs BMO-ViT – Brain Tumor Classification (Şahin et al. 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",        type=str,   required=False,
                   help="Brain Tumor MRI dataset root (Training/ + Testing/ sub-dirs).")
    p.add_argument("--num_classes",     type=int,   default=4)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--epochs",          type=int,   default=100,
                   help="Full-training epochs (paper: 100).")
    p.add_argument("--lr",              type=float, default=1e-4,
                   help="Adam learning rate (paper: 1e-4).")
    p.add_argument("--run_bmo_search",  action="store_true",
                   help="Run Optuna BMO search before BMO-ViT training.")
    p.add_argument("--run_ablation",    action="store_true",
                   help="Run ablation study (Table 7). Uses full args.epochs.")
    p.add_argument("--skip_base",       action="store_true")
    p.add_argument("--skip_bmo",        action="store_true")
    p.add_argument("--bmo_trials",      type=int,   default=200,
                   help="Optuna trials (paper: 200).")
    p.add_argument("--bmo_low_epochs",  type=int,   default=5,
                   help="Low-epoch budget per trial (paper: 5).")
    p.add_argument("--subset_frac",     type=float, default=0.3,
                   help="Fraction of data used during BMO search.")
    p.add_argument("--output_dir",      type=str,   default="./outputs")
    p.add_argument("--save_checkpoints",action="store_true")
    p.add_argument("--load_checkpoints",action="store_true")
    p.add_argument("--no_cuda",         action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(SEP)
    print("  ViT-Base vs BMO-ViT | Brain Tumor Classification")
    print("  Şahin et al., Biomedical Signal Processing & Control 91 (2024)")
    print(SEP)

    # Device
    section("Device")
    device = (torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda
              else torch.device("cpu"))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU : {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  CPU (training will be slow; use a GPU for real runs)")

    # Dataset
    section("Dataset")
    if not args.data_dir or not Path(args.data_dir).exists():
        print("  ERROR: --data_dir not set or not found.")
        print("  Download from: https://www.kaggle.com/datasets/"
              "masoudnickparvar/brain-tumor-mri-dataset")
        sys.exit(1)

    train_loader, val_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"  Train : {len(train_loader.dataset)} images  (paper: 5 712)")
    print(f"  Val   : {len(val_loader.dataset)} images  (paper: 1 311)")
    print(f"  Classes : {train_loader.dataset.classes}")
    print(f"  NOTE: Testing/ is used as validation split (matches paper methodology)")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    base_model, base_metrics, base_params = None, None, None
    if not args.skip_base:
        base_model, base_metrics, base_params = run_vit_base(
            args, train_loader, val_loader, device
        )

    # ── Stage 2 (optional) ────────────────────────────────────────────────────
    searched_params = None
    if args.run_bmo_search:
        summary = run_bmo_search(args, device)
        searched_params = summary["best_params"]

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    bmo_model, bmo_metrics, bmo_params = None, None, None
    if not args.skip_bmo:
        bmo_model, bmo_metrics, bmo_params = run_bmo_vit(
            args, train_loader, val_loader, device, searched_params
        )

    # ── Ablation (optional) ───────────────────────────────────────────────────
    if args.run_ablation:
        section("Ablation Study  (Table 7 of paper)")
        # FIX: n_epochs set to args.epochs (default 100), not the previous
        # hardcoded 10. Ablation at 10 epochs is too noisy to reproduce
        # Table 7's 0.965–0.980 accuracy range.
        ab = AblationStudy(train_loader, val_loader, device,
                           n_epochs=args.epochs, lr=args.lr,
                           num_classes=args.num_classes)
        rows = ab.run()
        _save_json(rows, Path(args.output_dir) / "ablation.json")
        print(f"\n  Ablation saved → {args.output_dir}/ablation.json")

    # ── Final comparison ──────────────────────────────────────────────────────
    if base_metrics and bmo_metrics:
        section("Final Comparison  (Tables 8 & 9 of paper)")
        print_comparison_table(base_metrics, bmo_metrics, base_params, bmo_params)

        report = {
            "vit_base": {**base_metrics, "n_params": base_params},
            "bmo_vit":  {**bmo_metrics,  "n_params": bmo_params},
        }
        _save_json(report, Path(args.output_dir) / "final_comparison.json")
        print(f"\n  Saved → {args.output_dir}/final_comparison.json")

    section("Done")
    print(f"  Outputs → {args.output_dir}/\n")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 1:
        _smoke_test()   # no args → self-contained validation
    else:
        main()
