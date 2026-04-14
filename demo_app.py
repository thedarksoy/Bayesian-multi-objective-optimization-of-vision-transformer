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
    Scans <data_dir>/Testing/{class}/*.jpg|png and returns a dict:
        { "Glioma": [path, ...], "Meningioma": [...], ... }
    Also returns a flat list for random sampling.
    """
    root = Path(data_dir) / "Testing"
    # Map folder name → display name
    folder_map = {
        "glioma":      "Glioma",
        "meningioma":  "Meningioma",
        "notumor":     "No Tumor",
        "pituitary":   "Pituitary",
    }

    index   = {name: [] for name in CLASS_NAMES}
    flat    = []

    for folder, display in folder_map.items():
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in sorted(folder_path.glob(ext)):
                index[display].append(str(p))
                flat.append((str(p), display))

    total = sum(len(v) for v in index.values())
    print(f"✓ Found {total} test images across {len([k for k,v in index.items() if v])} classes")
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
    with gr.Blocks(title="Brain Tumor Classifier", theme=gr.themes.Base()) as app:

        # Title
        gr.HTML("""
        <div id="title-row">
          <h1>🧠 Brain Tumor Classification</h1>
          <p>ViT-Base vs BMO-ViT &nbsp;·&nbsp; Şahin et al., BSPC 2024</p>
        </div>
        """)

        with gr.Row():
            # ── Left column: image + controls ────────────────────────────────
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

            # ── Right column: stats + gallery ────────────────────────────────
            with gr.Column(scale=2):
                gallery_html = gr.HTML(
                    value=build_image_gallery_html(image_index)
                )

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

        # ── Prediction cards ──────────────────────────────────────────────────
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

        # ── Speed comparison bar ──────────────────────────────────────────────
        speed_bar = gr.HTML(visible=False)

        # ── Class filter buttons (one per class) ──────────────────────────────
        class_state = gr.State(value=None)   # None = all classes

        # ── Helpers ───────────────────────────────────────────────────────────

        def _pick(path: str, true_label: str):
            state["current_path"] = path
            state["current_true"] = true_label

            base_cls, base_conf, base_probs, base_ms = predict(vit_base, path, device)
            bmo_cls,  bmo_conf,  bmo_probs,  bmo_ms  = predict(bmo_vit,  path, device)

            base_html = build_result_html(
                base_cls, base_conf, base_probs, base_ms,
                "ViT-Base  (85.8 M)", base_params, true_label
            )
            bmo_html = build_result_html(
                bmo_cls, bmo_conf, bmo_probs, bmo_ms,
                "BMO-ViT  (22.1 M)", bmo_params, true_label
            )

            # Speed comparison
            faster = "ViT-Base" if base_ms < bmo_ms else "BMO-ViT"
            ratio  = max(base_ms, bmo_ms) / max(min(base_ms, bmo_ms), 0.01)
            spd_html = f"""
            <div style="background:#1a1f2e; border:1px solid #2d3748; border-radius:10px;
                        padding:12px 20px; display:flex; align-items:center; gap:16px;
                        font-family:'Segoe UI',sans-serif; font-size:0.82rem;">
              <span style="color:#94a3b8;">⚡ Speed</span>
              <span style="color:#e2e8f0;">
                ViT-Base <b style="color:#a78bfa">{base_ms:.1f}ms</b>
              </span>
              <span style="color:#64748b;">vs</span>
              <span style="color:#e2e8f0;">
                BMO-ViT <b style="color:#34d399">{bmo_ms:.1f}ms</b>
              </span>
              <span style="color:#64748b;">→</span>
              <span style="color:#fbbf24;">
                {faster} is {ratio:.1f}× faster this image
              </span>
              <span style="color:#64748b; margin-left:auto;">
                True label: <b style="color:#e2e8f0;">{true_label}</b>
              </span>
            </div>"""

            fname = Path(path).name
            status = f'<div style="color:#94a3b8; font-size:0.78rem; padding:4px 0; text-align:center;">{fname}</div>'

            return gr.update(value=path), base_html, bmo_html, \
                   gr.update(value=spd_html, visible=True), status

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
            pool = flat_list if not cls_filter else [
                (p, l) for p, l in flat_list if l == cls_filter
            ]
            if not pool:
                return [gr.update()] * 5
            state["idx"] = (state["idx"] + 1) % len(flat_list)
            path, label  = flat_list[state["idx"]]
            return _pick(path, label)

        def prev_image(cls_filter):
            pool = flat_list if not cls_filter else [
                (p, l) for p, l in flat_list if l == cls_filter
            ]
            if not pool:
                return [gr.update()] * 5
            state["idx"] = (state["idx"] - 1) % len(flat_list)
            path, label  = flat_list[state["idx"]]
            return _pick(path, label)

        outputs = [image_display, base_result, bmo_result, speed_bar, status_box]

        btn_random.click(random_image, inputs=[class_state], outputs=outputs)
        btn_next.click(next_image,   inputs=[class_state], outputs=outputs)
        btn_prev.click(prev_image,   inputs=[class_state], outputs=outputs)

        # Auto-load first image on startup
        app.load(lambda: random_image(None), outputs=outputs)

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
    app.launch(server_port=args.port, share=args.share, inbrowser=True)
