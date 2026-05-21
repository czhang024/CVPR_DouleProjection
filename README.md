# 🧹 Closed-Form Concept Erasure with Double Projections

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-blue?style=flat-square" alt="CVPR 2026"/>
  <img src="https://img.shields.io/badge/Method-Training--Free-brightgreen?style=flat-square" alt="Training-Free"/>
  <img src="https://img.shields.io/badge/Backbone-SD%201.4%20%7C%20FLUX.1--schnell-orange?style=flat-square" alt="Backbone"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="License"/>
</p>

---

This repository contains the official implementation of **DP (Double Projections)**, a training-free, closed-form method for selectively erasing concepts from pre-trained text-to-image diffusion models. 

We evaluate DP on the **ImageNette** benchmark (10 object classes) across two diffusion backbones:
- 🖼️ **Stable Diffusion 1.4** — classical U-Net backbone, cross-attention weight edits
- ⚡ **FLUX.1-schnell** — modern flow-matching transformer, T5 + CLIP dual-encoder edits

For each backbone, runnable shell scripts, training logs, and evaluation results are provided for all 10 concepts under our DP method.

---

## 📁 Repository Structure

```
CVPR2026/
├── requirements.txt       # Shared Python dependencies
├── SD/                    # Stable Diffusion 1.4 experiments
│   ├── scripts_SD14/      # Shell scripts for all 10 concepts × 2 methods
│   └── training-logs/     # Pre-run logs with summary statistics
├── Flux/                  # FLUX.1-schnell experiments
│   ├── scripts/           # Shell scripts for all 10 concepts × 2 methods
│   └── logs/              # Pre-run logs with summary statistics
└── README.md              # This file
```

---

## ⚙️ Installation

```bash
conda create -n DP python=3.10
conda activate DP
pip install -r requirements.txt
```

---

## 🚀 Experiments

- **Stable Diffusion 1.4** — see [`SD/README.md`](SD/README.md)
- **FLUX.1-schnell** — see [`Flux/README.md`](Flux/README.md)

---

## 📄 Citation

```bibtex
@article{zhang2026closed,
  title={Closed-Form Concept Erasure via Double Projections},
  author={Zhang, Chi and Cheng, Jingpu and Wang, Zhixian and Liu, Ping},
  journal={arXiv preprint arXiv:2604.10032},
  year={2026}
}
```
