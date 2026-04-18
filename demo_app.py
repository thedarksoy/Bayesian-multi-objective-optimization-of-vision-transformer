"""
demo_app.py
-----------
Gradio web UI for Brain Tumor Classification demo.
Automatically loads test images from dataset directory.

Usage:
    pip install gradio
    python demo_app.py --data_dir /path/to/BrainTumorMRI --output_dir ./outputs

Requires trained checkpoints:
    outputs/vit_base.pth
    outputs/bmo_vit.pth

Generate them with:
    python main.py --data_dir /path/to/BrainTumorMRI --epochs 100 --save_checkpoints
"""

import argparse
import json
import random
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vit import get_vit_base, build_vit, count_parameters
from vit_bayesian import get_bmo_vit

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS  = ["#ef4444", "#f97316", "#22c55e", "#3b82f6"]   # red, orange, green, blue
IMAGE_SIZE    = 224

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_models(output_dir: str, device: torch.device):
    out = Path(output_dir)

    vit_base = get_vit_base(num_classes=4).to(device)
    bmo_vit  = get_bmo_vit(num_classes=4).to(device)

    base_ckpt = out / "vit_base.pth"
    bmo_ckpt  = out / "bmo_vit.pth"

    base_loaded = bmo_loaded = False

    if base_ckpt.exists():
        vit_base.load_state_dict(torch.load(base_ckpt, map_location=device))
        vit_base.eval()
        base_loaded = True
        print(f"✓ Loaded ViT-Base from {base_ckpt}")
    else:
        print(f"⚠ No checkpoint found at {base_ckpt} — running with random weights")
        vit_base.eval()

    if bmo_ckpt.exists():
        bmo_vit.load_state_dict(torch.load(bmo_ckpt, map_location=device))
        bmo_vit.eval()
        bmo_loaded = True
        print(f"✓ Loaded BMO-ViT  from {bmo_ckpt}")
    else:
        print(f"⚠ No checkpoint found at {bmo_ckpt} — running with random weights")
        bmo_vit.eval()

    return vit_base, bmo_vit, base_loaded, bmo_loaded


# ──────────────────────────────────────────────────────────────────────────────
# Test image index
# ──────────────────────────────────────────────────────────────────────────────

def build_image_index(data_dir: str):
    """
    Scans <data_dir>/Testing/{class}/*.jpg|png  (case-insensitive folder names).
    Returns: index dict { "Glioma": [path,...], ... } and flat list.
    """
    # Find the Testing folder (handles capitalisation differences)
    root = None
    for candidate in ["Testing", "testing", "Test", "test"]:
        p = Path(data_dir) / candidate
        if p.exists():
            root = p
            break

    if root is None:
        print(f"ERROR: Could not find Testing folder inside: {data_dir}")
        try:
            print(f"  Contents: {[x.name for x in Path(data_dir).iterdir()]}")
        except Exception:
            pass
        return {name: [] for name in CLASS_NAMES}, []

    print(f"Using test folder: {root}")
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    print(f"Subfolders found: {subdirs}")

    # Map any casing variant → display name
    def _normalise(s):
        return s.lower().replace(" ", "").replace("_", "").replace("-", "")

    folder_map = {
        _normalise("glioma"):     "Glioma",
        _normalise("meningioma"): "Meningioma",
        _normalise("notumor"):    "No Tumor",
        _normalise("no tumor"):   "No Tumor",
        _normalise("pituitary"):  "Pituitary",
    }

    index = {name: [] for name in CLASS_NAMES}
    flat  = []

    for folder_path in sorted(root.iterdir()):
        if not folder_path.is_dir():
            continue
        display = folder_map.get(_normalise(folder_path.name))
        if display is None:
            print(f"  Skipping unrecognised folder: {folder_path.name}")
            continue
        # Use a set to deduplicate (*.jpg and *.JPG both match on Windows)
        seen = set()
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in folder_path.glob(ext):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    index[display].append(str(p))
                    flat.append((str(p), display))
        count = len(seen)
        print(f"  {folder_path.name} -> {display}: {count} images")

    total = sum(len(v) for v in index.values())
    print(f"Total: {total} test images found")
    return index, flat


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_path: str, device: torch.device):
    img    = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)

    t0     = time.perf_counter()
    logits = model(tensor)
    ms     = (time.perf_counter() - t0) * 1000

    probs      = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs, ms


# ──────────────────────────────────────────────────────────────────────────────
# HTML builders
# ──────────────────────────────────────────────────────────────────────────────

def _confidence_bar(label, prob, color, is_pred):
    pct      = prob * 100
    border   = f"border: 2px solid {color};" if is_pred else "border: 2px solid transparent;"
    bg_pulse = "background: rgba(255,255,255,0.06);" if is_pred else ""
    return f"""
    <div style="margin-bottom:10px; padding:8px 12px; border-radius:8px;
                {border} {bg_pulse}">
      <div style="display:flex; justify-content:space-between;
                  margin-bottom:5px; font-size:0.85rem;">
        <span style="color:#e2e8f0; font-weight:{'700' if is_pred else '400'}">
          {'▶ ' if is_pred else ''}{label}
        </span>
        <span style="color:{color}; font-weight:600">{pct:.1f}%</span>
      </div>
      <div style="background:#2d3748; border-radius:4px; height:8px; overflow:hidden;">
        <div style="width:{pct:.1f}%; height:100%; background:{color};
                    border-radius:4px; transition:width 0.4s ease;"></div>
      </div>
    </div>"""


def build_result_html(
    pred_class, confidence, probs,
    infer_ms, model_name, n_params,
    true_label=None,
):
    pred_idx = CLASS_NAMES.index(pred_class)
    is_correct = (true_label == pred_class) if true_label else None

    # Header badge
    status_html = ""
    if true_label:
        if is_correct:
            status_html = '<span style="background:#22c55e22; color:#22c55e; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;">✓ CORRECT</span>'
        else:
            status_html = f'<span style="background:#ef444422; color:#ef4444; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;">✗ WRONG (true: {true_label})</span>'

    bars = "".join(
        _confidence_bar(CLASS_NAMES[i], probs[i], CLASS_COLORS[i], i == pred_idx)
        for i in range(4)
    )

    color = CLASS_COLORS[pred_idx]
    return f"""
    <div style="background:#1a1f2e; border:1px solid #2d3748; border-radius:12px;
                padding:20px; font-family:'Segoe UI',sans-serif; height:100%;">
      <div style="display:flex; justify-content:space-between; align-items:center;
                  margin-bottom:16px;">
        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                    letter-spacing:1px;">{model_name}</div>
        {status_html}
      </div>

      <div style="text-align:center; padding:14px 0; margin-bottom:16px;
                  border-radius:10px; background:{color}18; border:1px solid {color}44;">
        <div style="font-size:1.6rem; font-weight:700; color:{color};">
          {pred_class}
        </div>
        <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
          {confidence*100:.1f}% confidence
        </div>
      </div>

      <div>{bars}</div>

      <div style="display:flex; justify-content:space-between; margin-top:14px;
                  padding-top:14px; border-top:1px solid #2d3748;
                  font-size:0.78rem; color:#64748b;">
        <span>⚡ {infer_ms:.1f} ms</span>
        <span>⚙ {n_params/1e6:.1f}M params</span>
      </div>
    </div>"""


def build_image_gallery_html(index: dict, selected_path: str = None):
    """Compact class summary with image counts."""
    rows = ""
    for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
        n = len(index.get(cls, []))
        rows += f"""
        <div style="display:flex; align-items:center; gap:10px; padding:6px 0;
                    border-bottom:1px solid #1e2736;">
          <div style="width:10px; height:10px; border-radius:50%;
                      background:{color}; flex-shrink:0;"></div>
          <span style="color:#e2e8f0; font-size:0.85rem; flex:1;">{cls}</span>
          <span style="color:#64748b; font-size:0.8rem;">{n} images</span>
        </div>"""
    return f"""
    <div style="background:#1a1f2e; border:1px solid #2d3748; border-radius:12px;
                padding:16px; font-family:'Segoe UI',sans-serif;">
      <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                  letter-spacing:1px; margin-bottom:12px;">Test Dataset</div>
      {rows}
    </div>"""



# ──────────────────────────────────────────────────────────────────────────────
# Comparison table  (reads final_comparison.json + paper reference values)
# ──────────────────────────────────────────────────────────────────────────────

# Paper reference metrics (Table 8, Şahin et al. 2024)
PAPER_REFERENCE = {
    "vit_base": {
        "accuracy": 0.9511, "f1_score": 0.9487,
        "precision": 0.9502, "recall": 0.9511, "n_params": 85_800_000,
    },
    "bmo_vit": {
        "accuracy": 0.9659, "f1_score": 0.9810,
        "precision": 0.9838, "recall": 0.9659, "n_params": 22_100_000,
    },
}

# BMO search space from paper (Section 3 / Fig. 1)
BMO_SEARCH_SPACE = {
    "Patch Size":  {"values": [8, 16, 32],                    "vit_base": 16,   "bmo_paper": 32},
    "Embed Dim":   {"values": [64, 128, 256, 512, 768, 1536], "vit_base": 768,  "bmo_paper": 512},
    "Depth":       {"values": [6, 8, 10, 12, 14, 16, 24],    "vit_base": 12,   "bmo_paper": 6},
    "Attn Heads":  {"values": [8, 10, 12, 14, 16, 24],       "vit_base": 12,   "bmo_paper": 24},
    "MLP Dim":     {"values": [256, 512, 768, 1024, 1536, 2048], "vit_base": 3072, "bmo_paper": 256},
}


def _metric_row(label, paper_base, paper_bmo, your_base, your_bmo, fmt=".4f", pct=False):
    def _td(val, ref=None):
        if val is None:
            return '<td style="text-align:center;color:#475569;">—</td>'
        disp = f"{val*100:.2f}%" if pct else f"{val:{fmt}}"
        if ref is not None and val is not None:
            d = (val - ref) * (100 if pct else 1)
            color = "#22c55e" if d >= 0 else "#ef4444"
            sign  = "+" if d >= 0 else ""
            delta = f'<span style="font-size:0.7rem;color:{color};margin-left:4px;">({sign}{d:.2f}{"pp" if pct else ""})</span>'
        else:
            delta = ""
        return f'<td style="text-align:center;color:#e2e8f0;padding:8px 10px;">{disp}{delta}</td>'

    return f"""<tr style="border-bottom:1px solid #1e2736;">
      <td style="padding:8px 14px;color:#94a3b8;font-size:0.83rem;white-space:nowrap;">{label}</td>
      {_td(paper_base)}
      {_td(paper_bmo)}
      {_td(your_base,  paper_base)}
      {_td(your_bmo,   paper_bmo)}
    </tr>"""


def build_comparison_table_html(output_dir: str) -> str:
    json_path = Path(output_dir) / "final_comparison.json"
    yours = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                yours = json.load(f)
        except Exception:
            pass

    yb  = yours.get("vit_base", {})
    ybm = yours.get("bmo_vit",  {})
    pb  = PAPER_REFERENCE["vit_base"]
    pbm = PAPER_REFERENCE["bmo_vit"]

    # ── Table 1: Performance metrics ─────────────────────────────────────────
    th = """<tr style="border-bottom:2px solid #334155;">
      <th style="text-align:left;padding:10px 14px;color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Metric</th>
      <th style="text-align:center;padding:10px 8px;color:#a78bfa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">ViT-Base<br><span style="font-weight:400;color:#64748b;">Paper</span></th>
      <th style="text-align:center;padding:10px 8px;color:#34d399;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">BMO-ViT<br><span style="font-weight:400;color:#64748b;">Paper</span></th>
      <th style="text-align:center;padding:10px 8px;color:#60a5fa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">ViT-Base<br><span style="font-weight:400;color:#64748b;">Yours</span></th>
      <th style="text-align:center;padding:10px 8px;color:#f472b6;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">BMO-ViT<br><span style="font-weight:400;color:#64748b;">Yours</span></th>
    </tr>"""

    rows = ""
    rows += _metric_row("Accuracy",  pb["accuracy"],  pbm["accuracy"],  yb.get("accuracy"),  ybm.get("accuracy"),  pct=True)
    rows += _metric_row("F1-Score",  pb["f1_score"],  pbm["f1_score"],  yb.get("f1_score"),  ybm.get("f1_score"),  pct=True)
    rows += _metric_row("Precision", pb["precision"], pbm["precision"], yb.get("precision"), ybm.get("precision"), pct=True)
    rows += _metric_row("Recall",    pb["recall"],    pbm["recall"],    yb.get("recall"),    ybm.get("recall"),    pct=True)

    def _param_td(val, ref=None):
        if val is None:
            return '<td style="text-align:center;color:#475569;">—</td>'
        disp = f"{val/1e6:.1f}M"
        if ref:
            ratio = ref / val
            color = "#22c55e" if ratio >= 1 else "#ef4444"
            note  = f'<span style="font-size:0.7rem;color:{color};margin-left:4px;">({ratio:.1f}× smaller)</span>' if ratio != 1 else ""
        else:
            note = ""
        return f'<td style="text-align:center;color:#e2e8f0;padding:8px 10px;">{disp}{note}</td>'

    rows += f"""<tr style="border-bottom:1px solid #1e2736;">
      <td style="padding:8px 14px;color:#94a3b8;font-size:0.83rem;">Parameters</td>
      {_param_td(pb["n_params"])}
      {_param_td(pbm["n_params"], pb["n_params"])}
      {_param_td(yb.get("n_params"), pb["n_params"] if yb.get("n_params") else None)}
      {_param_td(ybm.get("n_params"), pbm["n_params"] if ybm.get("n_params") else None)}
    </tr>"""

    metrics_table = f"""
    <div style="font-family:'Segoe UI',sans-serif; margin-bottom:24px;">
      <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
        Performance Comparison &nbsp;·&nbsp; <span style="color:#475569;">Δ shown vs paper reference</span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
          <thead>{th}</thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""

    # ── Table 2: BMO Architecture parameters ─────────────────────────────────
    arch_rows = ""
    for param, info in BMO_SEARCH_SPACE.items():
        your_bmo_val = None
        # try to derive from n_params hint (we can't know exact arch from JSON alone)
        # so we show "—" for your BMO arch values — user knows their own config
        space_str = "{" + ", ".join(str(v) for v in info["values"]) + "}"
        vb_val    = info["vit_base"]
        bp_val    = info["bmo_paper"]

        # highlight if BMO differs from base
        vb_color  = "#e2e8f0"
        bp_color  = "#34d399" if bp_val != vb_val else "#e2e8f0"

        arch_rows += f"""<tr style="border-bottom:1px solid #1e2736;">
          <td style="padding:8px 14px;color:#94a3b8;font-size:0.83rem;white-space:nowrap;">{param}</td>
          <td style="text-align:center;padding:8px 10px;color:#64748b;font-size:0.78rem;">{space_str}</td>
          <td style="text-align:center;padding:8px 10px;color:{vb_color};font-weight:500;">{vb_val}</td>
          <td style="text-align:center;padding:8px 10px;color:{bp_color};font-weight:600;">{bp_val}</td>
        </tr>"""

    arch_th = """<tr style="border-bottom:2px solid #334155;">
      <th style="text-align:left;padding:10px 14px;color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Parameter</th>
      <th style="text-align:center;padding:10px 8px;color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Search Space</th>
      <th style="text-align:center;padding:10px 8px;color:#a78bfa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">ViT-Base</th>
      <th style="text-align:center;padding:10px 8px;color:#34d399;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">BMO-ViT (Paper Optimal)</th>
    </tr>"""

    arch_table = f"""
    <div style="font-family:'Segoe UI',sans-serif; margin-bottom:16px;">
      <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
        BMO Architecture Search — 5 Optimised Parameters
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
          <thead>{arch_th}</thead>
          <tbody>{arch_rows}</tbody>
        </table>
      </div>
      <div style="margin-top:8px;font-size:0.72rem;color:#475569;">
        Green = BMO found a different (better) value than ViT-Base default
      </div>
    </div>"""

    no_json = "" if json_path.exists() else """
    <div style="margin:12px 0; padding:10px 14px; background:#1e2736;
                border-radius:8px; color:#f97316; font-size:0.8rem;">
      ⚠ No final_comparison.json found — train both models with --save_checkpoints
      to populate the "Yours" columns.
    </div>"""

    footer = '<div style="margin-top:8px;font-size:0.72rem;color:#475569;">Paper: Şahin et al., BSPC 91 (2024) 105938 &nbsp;·&nbsp; Δ = your result minus paper reference</div>'

    return f"""
    <div style="padding:4px;">
      {no_json}
      {metrics_table}
      {arch_table}
      {footer}
    </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# Build Gradio app
# ──────────────────────────────────────────────────────────────────────────────

def build_app(data_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vit_base, bmo_vit, base_loaded, bmo_loaded = load_models(output_dir, device)
    base_params = count_parameters(vit_base)
    bmo_params  = count_parameters(bmo_vit)

    image_index, flat_list = build_image_index(data_dir)
    total_images = len(flat_list)

    # state: current index into flat_list
    state = {"idx": 0, "current_path": None, "current_true": None}

    # ── CSS ──────────────────────────────────────────────────────────────────
    app_css = """
    body, .gradio-container { background: #0f1117 !important; }
    .gr-block, .gr-box       { background: #0f1117 !important; }
    footer { display: none !important; }
    #title-row { text-align: center; padding: 24px 0 8px; }
    #title-row h1 {
        color: #e2e8f0; font-size: 1.7rem;
        font-family: 'Segoe UI', sans-serif; font-weight: 700;
        margin: 0;
    }
    #title-row p { color: #64748b; margin: 4px 0 0; font-size: 0.9rem; }
    .gr-button-primary {
        background: #3b82f6 !important;
        border: none !important; color: white !important;
        font-weight: 600 !important; border-radius: 8px !important;
    }
    .gr-button-primary:hover { background: #2563eb !important; }
    .gr-button-secondary {
        background: #1e2736 !important;
        border: 1px solid #2d3748 !important; color: #94a3b8 !important;
        border-radius: 8px !important;
    }
    .gr-button-secondary:hover { background: #252f42 !important; }
    .image-preview img { border-radius: 10px; }
    """

    # ── Gradio blocks ─────────────────────────────────────────────────────────
    with gr.Blocks(title="Brain Tumor Classifier") as app:

        gr.HTML("""
        <div id="title-row">
          <h1>🧠 Brain Tumor Classification</h1>
          <p>ViT-Base vs BMO-ViT &nbsp;·&nbsp; Şahin et al., BSPC 2024</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Inference ─────────────────────────────────────────────
            with gr.Tab("🔬 Inference"):

                with gr.Row():
                    # Left column: image + controls
                    with gr.Column(scale=4):
                        image_display = gr.Image(
                            label="MRI Scan",
                            elem_classes=["image-preview"],
                            height=320,
                        )
                        with gr.Row():
                            btn_random = gr.Button("🔀 Random Image", variant="primary", scale=2)
                            btn_next   = gr.Button("⏭ Next",          variant="secondary", scale=1)
                            btn_prev   = gr.Button("⏮ Prev",          variant="secondary", scale=1)

                        with gr.Row():
                            for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
                                gr.Button(
                                    f"{cls} ({len(image_index.get(cls,[]))})",
                                    variant="secondary", size="sm",
                                    elem_id=f"cls-{cls.lower().replace(' ','-')}"
                                )

                        status_box = gr.HTML(
                            value='<div style="color:#64748b; font-size:0.8rem; '
                                  'padding:8px 0; text-align:center;">No image selected</div>'
                        )

                    # Right column: dataset stats + model info
                    with gr.Column(scale=2):
                        gr.HTML(value=build_image_gallery_html(image_index))

                        gr.HTML(f"""
                        <div style="background:#1a1f2e; border:1px solid #2d3748;
                                    border-radius:12px; padding:16px; margin-top:12px;
                                    font-family:'Segoe UI',sans-serif;">
                          <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                                      letter-spacing:1px; margin-bottom:12px;">Model Info</div>
                          <div style="font-size:0.82rem; color:#94a3b8; line-height:1.8;">
                            <b style="color:#e2e8f0;">ViT-Base</b><br>
                            patch=16 · dim=768 · heads=12<br>
                            depth=12 · {base_params/1e6:.1f}M params<br>
                            {'<span style="color:#22c55e;">✓ checkpoint loaded</span>' if base_loaded else '<span style="color:#f97316;">⚠ random weights</span>'}
                            <br><br>
                            <b style="color:#e2e8f0;">BMO-ViT</b><br>
                            patch=32 · dim=512 · heads=24<br>
                            depth=6 · {bmo_params/1e6:.1f}M params<br>
                            {'<span style="color:#22c55e;">✓ checkpoint loaded</span>' if bmo_loaded else '<span style="color:#f97316;">⚠ random weights</span>'}
                          </div>
                        </div>
                        """)

                # Prediction cards
                gr.HTML('<div style="height:16px;"></div>')
                with gr.Row():
                    base_result = gr.HTML(
                        value='<div style="background:#1a1f2e; border:1px solid #2d3748; '
                              'border-radius:12px; padding:20px; color:#475569; '
                              'text-align:center; font-family:sans-serif;">'
                              'Run inference to see ViT-Base results</div>'
                    )
                    bmo_result = gr.HTML(
                        value='<div style="background:#1a1f2e; border:1px solid #2d3748; '
                              'border-radius:12px; padding:20px; color:#475569; '
                              'text-align:center; font-family:sans-serif;">'
                              'Run inference to see BMO-ViT results</div>'
                    )

                speed_bar   = gr.HTML(visible=False)
                class_state = gr.State(value=None)

                # ── Event handlers ───────────────────────────────────────────

                def _pick(path: str, true_label: str):
                    state["current_path"] = path
                    state["current_true"] = true_label

                    base_cls, base_conf, base_probs, base_ms = predict(vit_base, path, device)
                    bmo_cls,  bmo_conf,  bmo_probs,  bmo_ms  = predict(bmo_vit,  path, device)

                    base_html = build_result_html(
                        base_cls, base_conf, base_probs, base_ms,
                        "ViT-Base", base_params, true_label
                    )
                    bmo_html = build_result_html(
                        bmo_cls, bmo_conf, bmo_probs, bmo_ms,
                        "BMO-ViT", bmo_params, true_label
                    )

                    faster  = "ViT-Base" if base_ms < bmo_ms else "BMO-ViT"
                    ratio   = max(base_ms, bmo_ms) / max(min(base_ms, bmo_ms), 0.01)
                    spd_html = f"""
                    <div style="background:#1a1f2e; border:1px solid #2d3748; border-radius:10px;
                                padding:12px 20px; display:flex; align-items:center; gap:16px;
                                font-family:'Segoe UI',sans-serif; font-size:0.82rem;">
                      <span style="color:#94a3b8;">⚡ Speed</span>
                      <span style="color:#e2e8f0;">ViT-Base <b style="color:#a78bfa">{base_ms:.1f}ms</b></span>
                      <span style="color:#64748b;">vs</span>
                      <span style="color:#e2e8f0;">BMO-ViT <b style="color:#34d399">{bmo_ms:.1f}ms</b></span>
                      <span style="color:#64748b;">→</span>
                      <span style="color:#fbbf24;">{faster} is {ratio:.1f}× faster</span>
                      <span style="color:#64748b; margin-left:auto;">
                        True label: <b style="color:#e2e8f0;">{true_label}</b>
                      </span>
                    </div>"""

                    fname  = Path(path).name
                    status = f'<div style="color:#94a3b8; font-size:0.78rem; padding:4px 0; text-align:center;">{fname}</div>'
                    return gr.update(value=path), base_html, bmo_html,                            gr.update(value=spd_html, visible=True), status

                def random_image(cls_filter):
                    pool = flat_list if not cls_filter else [
                        (p, l) for p, l in flat_list if l == cls_filter
                    ]
                    if not pool:
                        return [gr.update()] * 5
                    path, label = random.choice(pool)
                    state["idx"] = flat_list.index((path, label))
                    return _pick(path, label)

                def next_image(cls_filter):
                    if not flat_list:
                        return [gr.update()] * 5
                    state["idx"] = (state["idx"] + 1) % len(flat_list)
                    path, label  = flat_list[state["idx"]]
                    return _pick(path, label)

                def prev_image(cls_filter):
                    if not flat_list:
                        return [gr.update()] * 5
                    state["idx"] = (state["idx"] - 1) % len(flat_list)
                    path, label  = flat_list[state["idx"]]
                    return _pick(path, label)

                outputs = [image_display, base_result, bmo_result, speed_bar, status_box]
                btn_random.click(random_image, inputs=[class_state], outputs=outputs)
                btn_next.click(next_image,     inputs=[class_state], outputs=outputs)
                btn_prev.click(prev_image,     inputs=[class_state], outputs=outputs)
                app.load(lambda: random_image(None), outputs=outputs)

            # ── Tab 2: Results vs Paper ──────────────────────────────────────
            with gr.Tab("📊 Results vs Paper"):
                gr.HTML(value=build_comparison_table_html(output_dir))
                gr.HTML("""
                <div style="padding:16px 4px; font-family:'Segoe UI',sans-serif;">
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:4px;">
                    <div style="background:#1a1f2e; border:1px solid #2d3748;
                                border-radius:10px; padding:14px;">
                      <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                                  letter-spacing:1px; margin-bottom:8px;">ViT-Base Architecture</div>
                      <div style="font-size:0.82rem; color:#94a3b8; line-height:1.9;">
                        patch=16 &nbsp;·&nbsp; dim=768 &nbsp;·&nbsp; heads=12<br>
                        depth=12 &nbsp;·&nbsp; mlp=3072 &nbsp;·&nbsp; ~85.8M params
                      </div>
                    </div>
                    <div style="background:#1a1f2e; border:1px solid #2d3748;
                                border-radius:10px; padding:14px;">
                      <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                                  letter-spacing:1px; margin-bottom:8px;">BMO-ViT Architecture</div>
                      <div style="font-size:0.82rem; color:#94a3b8; line-height:1.9;">
                        patch=32 &nbsp;·&nbsp; dim=512 &nbsp;·&nbsp; heads=24<br>
                        depth=6 &nbsp;·&nbsp; mlp=256 &nbsp;·&nbsp; ~22.1M params
                      </div>
                    </div>
                  </div>
                </div>
                """)

    app.css = app_css
    return app


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Brain Tumor Classifier Demo")
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Path to BrainTumorMRI dataset root (must have Testing/ subfolder)")
    p.add_argument("--output_dir", type=str, default="./outputs",
                   help="Directory containing vit_base.pth and bmo_vit.pth")
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--share",      action="store_true",
                   help="Create a public Gradio link")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app  = build_app(args.data_dir, args.output_dir)
    app.launch(server_port=args.port, share=args.share, inbrowser=True, theme=gr.themes.Base())
