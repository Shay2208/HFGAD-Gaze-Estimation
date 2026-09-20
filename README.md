# HFGAD: Hierarchical Fine-Grained Attention Decoder for Gaze Estimation

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Paper](https://img.shields.io/badge/Algorithms-18%2C%20538-brightgreen)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fa18090538-blue)](https://doi.org/10.3390/a18090538)

Official PyTorch implementation of **HFGAD**, a lightweight hierarchical fine-grained
attention decoder for appearance-based gaze estimation.

> **论文开源仓库 / 中文说明**：本仓库为 *Algorithms* 2025 论文 *HFGAD: Hierarchical
> Fine-Grained Attention Decoder for Gaze Estimation* 的官方代码。正文为英文，关键处附中文注释。

| | |
|---|---|
| **Paper** | [Algorithms 2025, 18(9), 538](https://doi.org/10.3390/a18090538) (MDPI, open access, CC BY) |
| **Authors** | Shaojie Huang, Tianzhong Wang, Weiquan Liu, Yingchao Piao, Jinhe Su, Guorong Cai, Huilin Xu |
| **Affiliation** | School of Computer Engineering, Jimei University |
| **Task** | Appearance-based gaze estimation (predict pitch & yaw from a face image) |
| **Backbones** | ResNet-18 (default), PVT v2-B0 / B2, MobileNetV2 |

---

## 1. What is HFGAD?

Most appearance-based gaze estimators only consume the **coarse-grained** features from
the deepest stage of a visual encoder. Gaze, however, is determined by very fine cues —
tiny movements of the iris and pupil — which live in the **shallow** layers and are
gradually washed out by downsampling.

HFGAD is a **plug-and-play, ultra-lightweight decoder** that sits on top of any
hierarchical encoder (CNN or ViT) and progressively fuses fine-grained shallow features
into deep ones. It contains four parts:

| Module | Code | What it does |
|---|---|---|
| **MSCSA** — Multi-Scale Channel-Spatial Attention | `ELA` + `MSGC` in `models/GazeModules.py` | Orthogonally decouples spatial attention into height / width pathways, then re-weights three parallel grouped-conv branches (1×1, 3×3, 5×5) with a softmax gate. Directs focus onto gaze-relevant regions. |
| **SFM** — Selective Fusion Module | `SFM` | Fuses the current-stage feature with the previous-stage feature using a pair of *complementary* attention maps `a` and `1 - a`, so the network learns the optimal mixing instead of a naive sum. |
| **ECD** — Efficient Convolutional Downsample | `DPW` | `DW3×3(stride2) → attention → BN → SiLU → 1×1` aligns shallow features with the next stage (a bottom-up PANet-style path). |
| **EH** — Estimation Head | `AdaptiveAvgPool + Linear` | Regresses the 2-D gaze direction (pitch, yaw). |

The full decoder costs only **+0.838 M params / +0.126 G FLOPs** over a ResNet-18 encoder.

> **命名说明 / Note on naming:** the decoder class was called `SRAD` during development.
> It is now `HFGAD` to match the paper. `SRAD = HFGAD` is kept as an alias and the
> `state_dict` keys are unchanged, so all previously released checkpoints still load.

```python
import torch
from models.Gaze import GazeModel

model = GazeModel(backbone="resnet18")          # 12.04 M params, 1.95 G FLOPs
model.load_state_dict(torch.load("evaluation/gaze360/checkpoint/ours_10.46.pt"))
model.eval()

gaze = model(torch.randn(1, 3, 224, 224))       # -> [B, 2] = (pitch, yaw) in radians
```

---

## 2. Results

### 2.1 Comparison with state of the art (Table 1 of the paper)

Average angular error (°, **lower is better**). `HFGAD-Res18` = ResNet-18 encoder + HFGAD.

| Method | Params | FLOPs | Gaze360 | MPIIFaceGaze | IVGaze |
|---|---:|---:|---:|---:|---:|
| FullFace | 196.6 M | 2.99 G | 14.99° | 4.93° | 13.67° |
| RT-Gene | 82.0 M | 30.81 G | 12.26° | 4.66° | – |
| Gaze360 | 14.6 M | 12.78 G | 11.04° | 4.06° | 8.15° |
| CA-Net | 34.1 M | 15.6 G | 11.20° | 4.27° | – |
| GazeTR | 11.4 M | 1.82 G | 10.62° | 4.00° | 7.33° |
| GazePTR | 12.1 M | 3.75 G | 10.59° | 3.98° | 7.04° |
| DGE-GM | 87.7 M | 15.16 G | 10.62° | 3.76° | – |
| **HFGAD-Res18 (ours)** | **12.0 M** | **1.95 G** | **10.46°** | **3.88°** | **7.02°** |

### 2.2 Ablation (Table 3 of the paper)

| Model | Params (M) | FLOPs (G) | Gaze360 | MPII | IVGaze |
|---|---:|---:|---:|---:|---:|
| ResNet-18 baseline | 11.203 | 1.827 | 10.91° | 4.20° | 8.56° |
| + SFM | 11.409 | 1.945 | 10.74° | 4.07° | 7.67° |
| + MSCSA | 11.617 | 1.932 | 10.65° | 3.97° | 7.41° |
| **+ SFM + MSCSA (HFGAD)** | **12.041** | **1.953** | **10.46°** | **3.88°** | **7.02°** |

### 2.3 Generalization across backbones (Table 4 of the paper, Gaze360)

| Backbone | Params (M) | w/o HFGAD | w/ HFGAD |
|---|---:|---:|---:|
| ResNet-18 | 11.2 → 12.0 | 10.91° | **10.46°** |
| PVT v2-B0 | 3.4 → 3.7 | 11.32° | **11.00°** |
| PVT v2-B2 | 24.9 → 25.8 | 10.63° | **10.24°** |

---

## 3. Installation

```bash
git clone https://github.com/Shay2208/HFGAD.git
cd HFGAD

# CPU-only (enough for the demo and ONNX inference)
pip install -r requirements.txt
```

> 训练需要 GPU。请到 <https://pytorch.org/> 安装与你的 CUDA 版本匹配的 torch，例如：
> ```bash
> pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
>     --extra-index-url https://download.pytorch.org/whl/cu117
> ```
> The pinned `torch==1.13.1` / `torchvision==0.14.1` in `requirements.txt` are CPU builds.

Environment check (依赖自检 / sanity check):

```bash
python verify.py
```

---

## 4. Model weights

权重文件不存放在 git 中（合计约 580 MB），统一发布在 GitHub Release。
Weights are **not** stored in git; they are attached to a GitHub Release.

```bash
# Core weights only (Gaze360 .pt + .onnx) — enough for the demo
python scripts/download_weights.py

# Everything (MPIIFaceGaze folds + ResNet-18 backbone)
python scripts/download_weights.py --all

# Inspect what a release contains
python scripts/download_weights.py --list
```

| Release asset | Destination | Description |
|---|---|---|
| `hfgad_gaze360_resnet18.pt` | `evaluation/gaze360/checkpoint/ours_10.46.pt` | Gaze360, ResNet-18, **10.46°** |
| `hfgad_gaze360_resnet18.onnx` | `evaluation/gaze360/checkpoint/ours_10.46.onnx` | Same model, ONNX (opset 17, CPU-ready) |
| `hfgad_mpii_p00.pt` / `p04` / `p05` | `evaluation/mpii/checkpoint/pXX.label/hfgad.pt` | MPIIFaceGaze leave-one-out folds |
| `resnet18-5c106cde.pth` | `ckpts/resnet18-5c106cde.pth` | torchvision ResNet-18 ImageNet weights |

If you have no network access, download the assets manually from the Releases page and
place them at the paths above.

---

## 5. Quick start — real-time webcam demo (CPU, no dataset needed)

```bash
python scripts/download_weights.py      # fetch the ONNX model first
python webcam_gaze_demo.py
```

The demo opens your webcam, detects a face with a Haar cascade, runs the ONNX model,
draws the predicted gaze arrow, and smooths both the gaze vector and the face box with a
Kalman filter.

```bash
python webcam_gaze_demo.py --camera-id 1                       # pick another camera
python webcam_gaze_demo.py --show-profiler --detect-every 3 \
                           --infer-every 2 --detect-scale 0.6  # faster on weak CPUs
python webcam_gaze_demo.py --model path/to/your.onnx
```

Press `q` (or `Esc`) to quit.

---

## 6. Datasets

本仓库不分发任何数据集 / This repository does **not** redistribute any dataset.
Download them from the official sources and respect their licenses.

```
HFGAD/
└─ datasets/
   ├─ Gaze360/
   │  ├─ Image/          # images
   │  └─ Label/
   │     ├─ train.label
   │     └─ test.label
   ├─ MPIIFaceGaze/
   │  ├─ Image/          # p00/p01/... sub-folders
   │  └─ Label/          # p00.label/p01.label/...
   └─ RT-Gene/
      ├─ gaze_refine/
      └─ Label/train
```

| Dataset | Link |
|---|---|
| Gaze360 | <https://github.com/eth-avl/gaze360> |
| MPIIFaceGaze | <https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/> |
| RT-Gene | <https://github.com/Tobias-Fischer/rt_gene> |
| IVGaze | see the IVGaze project page (in-vehicle dataset used in the paper) |

Then point `data.image` / `data.label` in `config/**/*.yaml` at your local paths
(the defaults are `./datasets/...` placeholders).

---

## 7. Training

```bash
# Gaze360 (single run, 80 epochs, the main setting in the paper)
python trainer/total.py -s config/train/config_gaze360.yaml

# MPIIFaceGaze — leave-one-out, -p is the held-out subject id (0~14)
python trainer/leave.py -s config/train/config_mpii.yaml -p 0

# Train and evaluate in one shot
python train_test.py -s config/train/config_gaze360.yaml -t config/test/config_gaze360.yaml
```

Reference hyper-parameters (from the paper): Adam (β1 = 0.9, β2 = 0.999), 80 epochs,
5-epoch warmup, StepLR decaying ×0.5 every 15 epochs, batch size 64, single NVIDIA 3090.
The paper reports an initial LR of 5e-3 (1e-3 for IVGaze); the YAML files ship with
5e-4 — adjust to whichever matches your run.

Checkpoints are written to `evaluation/<dataset>/checkpoint/`.

---

## 8. Evaluation

```bash
python tester/total.py -t config/test/config_gaze360.yaml

# Evaluate a specific checkpoint
python tester/total.py -t config/test/config_gaze360.yaml \
                       -w evaluation/gaze360/checkpoint/ours_10.46.pt
```

The tester reports the mean angular error (°) over the test split.

---

## 9. ONNX export and CPU inference

```bash
python tools/export_onnx.py                     # .pt -> .onnx
python tools/cpu_infer.py                       # smoke test on CPU
python tools/benchmark_onnx.py --iters 30 --warmup 10 --intra-op-threads 2
python tools/quantize_onnx_int8.py              # optional INT8 quantization
python tools/compare_onnx_outputs.py            # compare ONNX vs PyTorch outputs
```

---

## 10. Repository structure

```
HFGAD/
├─ config/                  Training / test configs (YAML)
│  ├─ train/                config_gaze360 / mpii / rt / eth / diap
│  └─ test/
├─ models/
│  ├─ Gaze.py               Top-level model: encoder + HFGAD decoder
│  ├─ GazeModules.py        HFGAD decoder: MSCSA (ELA+MSGC), SFM, DPW/ECD
│  ├─ resnet.py             ResNet-18 backbone (default)
│  ├─ PVT.py                PVT v2 backbones
│  ├─ emcad.py              MobileNetV2 backbone
│  └─ gaze360.py / model.py / mtools.py   Experimental baselines & helpers
├─ reader/                  Dataset loading
├─ trainer/                 total.py (full run) / leave.py (leave-one-out)
├─ tester/                  total.py (full run) / leave.py
├─ tools/                   ONNX export / quantize / benchmark / CPU smoke test
├─ scripts/
│  └─ download_weights.py   Fetch weights from GitHub Releases
├─ ctools.py                Generic training utilities
├─ gtools.py                Gaze math & visualization (angle conversion, arrows)
├─ train_test.py            Train + test in one script
├─ webcam_gaze_demo.py      Real-time webcam demo (ONNX, CPU)
├─ verify.py                Environment sanity check
└─ requirements.txt
```

---

## 11. FAQ / 常见问题

- **Webcam does not open / black screen** — try `--camera-id 0/1/2`, and close other apps
  that may hold the camera.
- **`No module named 'xxx'`** — run `pip install -r requirements.txt` and make sure the
  interpreter matches the environment you installed into.
- **`Checkpoint not found`** — run `python scripts/download_weights.py` first.
- **Data path errors** — `config/*.yaml` uses `./datasets/...` placeholders; replace them
  with your real paths.
- **`emcad.py` complains about `mobilenetv2.pth`** — the MobileNetV2 pretrained weight is
  not shipped; download it yourself into `ckpts/` if you need that backbone.
- **Out of GPU memory** — lower `batch_size` in `config/train/*.yaml`.
- **ONNX slower than expected** — tune `--intra-op-threads` / `--inter-op-threads`.

---

## 12. Citation

If this code helps your research, please cite the paper:

```bibtex
@article{huang2025hfgad,
  title   = {HFGAD: Hierarchical Fine-Grained Attention Decoder for Gaze Estimation},
  author  = {Huang, Shaojie and Wang, Tianzhong and Liu, Weiquan and Piao, Yingchao
             and Su, Jinhe and Cai, Guorong and Xu, Huilin},
  journal = {Algorithms},
  volume  = {18},
  number  = {9},
  pages   = {538},
  year    = {2025},
  doi     = {10.3390/a18090538}
}
```

(`CITATION.bib` and `CITATION.cff` are included; GitHub renders the latter as
"Cite this repository".)

---

## 13. License and acknowledgements

- **Code**: MIT — see [LICENSE](LICENSE).
- **Paper**: open access under CC BY 4.0 (MDPI).
- Third-party code and pretrained weights (MobileNetV2, PVT v2, torchvision ResNet,
  optional GazeTR initialization) remain under their own licenses — see [NOTICE](NOTICE).
- We thank the authors of Gaze360, MPIIFaceGaze, RT-Gene and IVGaze for releasing their
  datasets, and the open-source gaze-estimation community whose code this project builds on.
