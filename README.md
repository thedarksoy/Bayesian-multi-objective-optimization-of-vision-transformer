#  Brain Tumor Classification — ViT-Base vs BMO-ViT

PyTorch implementation of **"Multi-objective optimization of ViT architecture for efficient brain tumor classification"** by Şahin et al., *Biomedical Signal Processing and Control* 91 (2024) 105938.

This repo reproduces the full pipeline from the paper: training a ViT-Base baseline, running a Bayesian Multi-Objective (BMO) architecture search with Optuna, and comparing both models on the Brain Tumor MRI dataset. A Gradio web UI is included for demo purposes.

---

##  Table of Contents

- [Results](#-results)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Demo UI](#-demo-ui)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Citation](#-citation)

---

##  Results

| Model | Accuracy | F1-Score | Precision | Params | Size |
|-------|----------|----------|-----------|--------|------|
| ViT-Base | 95.11% | 94.87% | 95.02% | 85.8 M | 343.1 MB |
| **BMO-ViT** | **96.59%** | **98.10%** | **98.38%** | **22.1 M** | **88.3 MB** |
| Δ (paper) | +1.48% | +3.23% | +3.36% | ~4× smaller | ~4× smaller |

BMO-ViT achieves higher accuracy with ~4× fewer parameters and ~2× faster inference than ViT-Base.

---

## Architecture

The paper searches over 5 ViT hyperparameters using Bayesian Multi-Objective Optimisation:

| Hyperparameter | ViT-Base | BMO-ViT (optimal) | Search Space |
|---|---|---|---|
| Patch size | 16 | **32** | {8, 16, 32} |
| Embedding dim | 768 | **512** | {64, 128, 256, 512, 768, 1536} |
| Attention heads | 12 | **24** | {8, 10, 12, 14, 16, 24} |
| Depth (layers) | 12 | **6** | {6, 8, 10, 12, 14, 16, 24} |
| MLP dim | 3072 | **256** | {256, 512, 768, 1024, 1536, 2048} |

The BMO search simultaneously **maximises accuracy** and **minimises parameter count** using Optuna's TPE sampler, running 200 trials with 5 low-epochs each.

---

## Installation
PLEASE DOWNLOAD THE OUTPUTS FOLDER FROM THIS LINK AND PASTE IT IN THE WORKING DIRECTORY -https://drive.google.com/drive/folders/1BgwhRSGw437EKuNMzE3ABbo6SBg462M_?usp=drive_link
```bash
git clone https://github.com/thedarksoy/Bayesian-multi-objective-optimization-of-vision-transformer
cd Bayesian-multi-objective-optimization-of-vision-transformer
pip install torch torchvision timm optuna scikit-learn gradio
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- timm
- optuna
- scikit-learn
- gradio (for demo UI only)

---

##  Dataset Setup

Download the Brain Tumor MRI dataset from Kaggle:
[masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Arrange the folders like this:

```
BrainTumorMRI/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The dataset contains **5,712 training** and **1,311 test** images across 4 classes.

---

## Usage

### Smoke test (no dataset needed)

Validates both model architectures, forward passes, and Optuna setup:

```bash
python main.py
```

### Train both models (full pipeline)

```bash
python main.py --data_dir /path/to/BrainTumorMRI --save_checkpoints
```

### Skip the BMO search, use paper's best params directly

```bash
python main.py --data_dir /path/to/BrainTumorMRI --save_checkpoints
```

### Run BMO architecture search + full training

```bash
python main.py --data_dir /path/to/BrainTumorMRI --run_bmo_search --save_checkpoints
```

### Load saved checkpoints (instant, no retraining)

```bash
python main.py --data_dir /path/to/BrainTumorMRI --load_checkpoints
```

### Run ablation study (Table 7 of paper)

```bash
python main.py --data_dir /path/to/BrainTumorMRI --run_ablation --load_checkpoints
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Path to BrainTumorMRI dataset root |
| `--epochs` | 100 | Training epochs (paper: 100) |
| `--lr` | 1e-4 | Adam learning rate |
| `--batch_size` | 64 | Batch size |
| `--run_bmo_search` | off | Run Optuna BMO search before training |
| `--bmo_trials` | 200 | Number of Optuna trials (paper: 200) |
| `--bmo_low_epochs` | 5 | Epochs per BMO trial (paper: 5) |
| `--subset_frac` | 0.3 | Fraction of data used during BMO search |
| `--run_ablation` | off | Run ablation study |
| `--save_checkpoints` | off | Save `.pth` files after training |
| `--load_checkpoints` | off | Load `.pth` files, skip training |
| `--skip_base` | off | Skip ViT-Base training |
| `--skip_bmo` | off | Skip BMO-ViT training |
| `--output_dir` | `./outputs` | Directory for checkpoints and logs |
| `--no_cuda` | off | Force CPU |

---

##  Demo UI

A Gradio web UI that auto-loads test images from the dataset and runs both models side-by-side.

**First train and save checkpoints:**
```bash
python main.py --data_dir /path/to/BrainTumorMRI --epochs 100 --save_checkpoints
```

**Then launch the demo:**
```bash
python demo_app.py --data_dir /path/to/BrainTumorMRI --output_dir ./outputs
```

Open `http://127.0.0.1:7860` in your browser.

**Features:**
- Auto-loads a random test image on startup
- Step through images with Next / Prev or jump to a random one
- Side-by-side confidence bars for both models
- ✓ / ✗ correct/wrong badge (ground truth is known from folder name)
- Inference time and parameter count comparison per image

> The demo works with random weights too (before training), but predictions will be meaningless. Train first for real results.

---

##  Project Structure

```
├── main.py           # Full pipeline: ViT-Base → BMO search → BMO-ViT → comparison
├── vit.py            # ViT-Base model, data loaders, training loop, evaluation
├── vit_bayesian.py   # BMO-ViT config, Optuna optimizer, ablation study
├── demo_app.py       # Gradio web UI
├── outputs/          # Checkpoints and logs (created at runtime)
│   ├── vit_base.pth
│   ├── bmo_vit.pth
│   ├── history_vit_base.json
│   ├── history_bmo_vit.json
│   ├── bmo_trials.json
│   ├── ablation.json
│   └── final_comparison.json
└── README.md
```

---

##  How It Works

### Loss Function

Multi-class Binary Cross-Entropy (Equation 5 of the paper) — each class is treated as an independent binary problem:

$$\mathcal{L} = -\frac{1}{N} \sum_i \sum_c \left[ y_{ic} \log(\hat{y}_{ic}) + (1 - y_{ic}) \log(1 - \hat{y}_{ic}) \right]$$

### BMO Search

The Bayesian Multi-Objective search uses Optuna's TPE sampler to explore the architecture space. Each trial:
1. Samples 5 hyperparameters from the search space
2. Trains the resulting ViT for 5 epochs on 30% of the data
3. Returns `(accuracy, n_params)` as the two objectives

Pareto-optimal trials satisfy both hard constraints: `accuracy > 80%` AND `params < 40M`. The best trial by accuracy from the Pareto front is used for full training.

### Data Augmentation

Training: random horizontal flip, ±15° rotation, colour jitter (brightness/contrast ±0.2), normalised with ImageNet mean/std.

---

##  Citation

```bibtex
@article{sahin2024bmo,
  title   = {Multi-objective optimization of ViT architecture for efficient brain tumor classification},
  author  = {Şahin and others},
  journal = {Biomedical Signal Processing and Control},
  volume  = {91},
  pages   = {105938},
  year    = {2024},
  doi     = {10.1016/j.bspc.2023.105938}
}
```
