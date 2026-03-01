# 🧠 Intracranial Aneurysm Detection — RSNA Kaggle Competition

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA%20Aneurysm%20Detection-blue?logo=kaggle)](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-green)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end deep learning pipeline for detecting and localizing intracranial aneurysms across **multiple imaging modalities** (CTA, MRA, MRI T1post, MRI T2), featuring a two-stage hierarchical architecture: a **patch-level Hybrid SwinUNETR classifier** followed by a **scan-level Hierarchical Transformer** that aggregates patch embeddings across the full 3D volume.

---

## Table of Contents

- [Background & Competition](#background--competition)
- [Project Journey](#project-journey)
- [Dataset & Modalities](#dataset--modalities)
- [Pipeline Overview](#pipeline-overview)
- [Preprocessing](#preprocessing)
- [3D Patch Strategy](#3d-patch-strategy)
- [Model Architecture](#model-architecture)
  - [Phase 1 — Patch-Level Hybrid Classifier](#phase-1--patch-level-hybrid-classifier)
  - [Phase 2 — Scan-Level Hierarchical Transformer](#phase-2--scan-level-hierarchical-transformer)
- [Loss Function](#loss-function)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Technologies Used](#technologies-used)

---

## Background & Competition

The [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) competition challenged participants to automatically detect and localize intracranial aneurysms — balloon-like bulges in blood vessels in the brain — from medical imaging scans. Undetected aneurysms can rupture, causing life-threatening hemorrhagic stroke.

The task required:
- **Binary classification**: Is an aneurysm present?
- **Multi-label localization**: Which of 13 arterial locations is the aneurysm at? (e.g., Left MCA, Basilar Tip, Anterior Communicating Artery, etc.)
- Handling of a **highly imbalanced dataset** across heterogeneous imaging modalities.

---

## Project Journey

This competition was an intensive deep dive into 3D medical imaging. The focus was not on rushing to a model, but on truly understanding the data first.

**Month 1 — Deep Data Exploration.** The dataset contained multiple imaging modalities — CTA, MRA, MRI T1post, MRI T2 — each with fundamentally different acquisition physics and visual characteristics. Before writing a single line of model code, significant time was spent understanding what each modality *looks like*, how Hounsfield units behave in CT versus signal intensities in MRI, and how aneurysms manifest differently across these scan types.

A key insight came from the manufacturer metadata: even within a single modality, scans from different scanner manufacturers had vastly different intensity distributions, slice spacings, and resolutions. This drove a month-long investigation into clustering and normalization strategies, exploring multiple techniques before converging on a robust approach.

**Preprocessing Design.** The final preprocessing pipelines were designed separately for CT-based scans (CTA) and MRI-based scans (MRA, T1post, T2), each tuned to handle the specific physics and artifacts of that modality. Coordinate transformation logic was built to correctly map the radiologist's 2D DICOM annotations into the 3D resampled voxel space — a non-trivial challenge with multiframe DICOMs and varying frame-of-reference origins.

**Model Architecture.** The model strategy was inspired by the scale mismatch between aneurysms (small, millimeter-sized structures) and full brain scans (large 3D volumes). This led to a two-stage design: a patch-level model that learns local features, followed by a scan-level model that reasons globally.

---

## Dataset & Modalities

The competition dataset contains DICOM brain scans from multiple hospitals and scanner manufacturers. Each scan is annotated with:
- Aneurysm presence/absence
- 2D pixel coordinates on a specific DICOM slice (for positive cases)
- Arterial location label from 13 possible locations

| Modality | Description | Key Challenge |
|----------|-------------|---------------|
| **CTA** | CT Angiography — high-res vascular contrast | HU windowing, bone/vessel separation |
| **MRA** | MR Angiography — vascular-sensitive MRI | Highly variable intensity, manufacturer differences |
| **MRI T1post** | Post-contrast T1-weighted MRI | Lower vascular contrast than CTA |
| **MRI T2** | T2-weighted MRI | Fluid-bright, different tissue contrast |

The dataset is heavily imbalanced: the majority of scan patches do not contain an aneurysm.

---

## Pipeline Overview

```
Raw DICOM Scans
      │
      ▼
┌─────────────────────────────────────┐
│  Preprocessing (modality-specific)  │
│  • DICOM → 3D volume                │
│  • Resampling to isotropic spacing  │
│  • Intensity normalization          │
│  • 2D annotation → 3D voxel coords  │
└────────────────┬────────────────────┘
                 │
                 ▼
      HDF5 Storage (float16, gzip)
                 │
                 ▼
┌─────────────────────────────────────┐
│   3D Sliding Window Patching        │
│   • 96×96×96 voxel patches          │
│   • Stride-based full coverage      │
│   • Manifest CSVs with patch coords │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Phase 1: Patch-Level Model         │
│  HybridAneurysmClassifier           │
│  • Modality-specific CNN stems      │
│  • Shared SwinUNETR transformer     │
│  • Hierarchical dual-head output    │
│    ├── Aneurysm Present (binary)    │
│    └── Artery Location (13-class)   │
└────────────────┬────────────────────┘
                 │  (extract embeddings)
                 ▼
┌─────────────────────────────────────┐
│  Phase 2: Scan-Level Model          │
│  HierarchicalTransformer            │
│  • Local Transformer (patch→block)  │
│  • Global Transformer (block→scan)  │
│  • Final scan-level prediction      │
└─────────────────────────────────────┘
```

---

## Preprocessing

Two separate, modality-aware preprocessing pipelines were built:

### CT / CTA Pipeline (`preprocess_ct.py`)

- Reads DICOM series with SimpleITK, handles multiframe and single-frame formats
- Resamples to isotropic voxel spacing using B-spline interpolation
- Applies HU windowing specific to the vascular/brain window
- Maps 2D DICOM pixel annotations to 3D physical coordinates using the ImagePositionPatient and ImageOrientationPatient DICOM tags
- Handles edge cases: manufacturer-specific frame-of-reference fields, multiframe `PerFrameFunctionalGroupsSequence` extraction

### MRI / MRA Pipeline (`prep_mr.py`)

- Handles the diversity of MRI acquisition protocols and scanner manufacturers
- Applies modality-aware normalization (z-score or percentile clipping rather than HU windowing)
- Resolves the frame number `f` embedded in annotation coordinates for multiframe MRI DICOMs
- Implements peak-finding heuristics (`scipy.signal.find_peaks`) to identify the correct DICOM slice from the manufacturer-provided frame index

### Storage Format

All processed scans are stored in a single HDF5 file (`processed_scans.hdf5`) using:
- `float16` precision to reduce memory footprint
- Chunked storage `(32, 32, 32)` for efficient random patch reads
- GZIP compression
- A file-lock mechanism (`*.lock` files) for safe parallel multi-process writes

```
processed_data_unified/
├── processed_scans.hdf5        # All 3D volumes
├── preprocessing_log.csv       # Status per scan
└── localization_manifest.csv   # Coordinates + labels for all scans
```

---

## 3D Patch Strategy

Because aneurysms are small relative to the full brain volume, the scan is divided into overlapping 3D patches using a sliding window approach.

| Parameter | Value |
|-----------|-------|
| Patch size | 96 × 96 × 96 voxels |
| Stride | 64 voxels |
| Coverage | Full scan (last patch snapped to boundary) |
| Label assignment | Aneurysm present if center coordinate falls in patch |
| Artery labels | OR-aggregated across all aneurysms in patch |
| Relative coords | Aneurysm position relative to patch origin |

Patches are not saved to disk. Instead, **manifest CSVs** store the `(series_uid, start_z, start_y, start_x)` for each patch. At training time, patches are read on-the-fly from the HDF5 file via memory-mapped access.

```
aneurysm_dataset_manifests_hdf5/
├── train_manifest.csv
└── test_manifest.csv
```

The train/test split is done at the **scan (patient) level** to prevent data leakage.

---

## Model Architecture

### Phase 1 — Patch-Level Hybrid Classifier

The core design challenge: different modalities have completely different low-level features (HU ranges, vessel contrast, noise characteristics), but a shared high-level reasoning module should be able to recognize aneurysm morphology across all of them.

**Solution: modality-specific CNN stems + a shared Swin Transformer body.**

```
Input 96×96×96 Patch (single-channel)
        │
        ▼
┌────────────────────────────┐
│  Modality-Specific Stem    │  ← Pretrained weights extracted from:
│  CTA / MRA: SegResNet      │     • SegResNet (whole-body CT segmentation)
│  MRI T1post: BraTS T1c     │     • SwinUNETR (BTCV segmentation)
│  MRI T2:    BraTS T2w      │     • BraTS MRI segmentation model
│  Output: 128 channels      │
└────────────┬───────────────┘
             │
             ▼
     Linear Bridge (128 → 768)
     GELU + Dropout
             │
             ▼
┌────────────────────────────┐
│   SwinUNETR Transformer    │  ← Pretrained on BTCV
│   (shared across modalities│
│    SWIN_EMBED_DIM = 768)   │
│   4 Swin layers            │
│   Global average pooling   │
└────────────┬───────────────┘
             │
    ┌────────┴──────────┐
    ▼                   ▼
Aneurysm Head       Artery Head
LayerNorm → Linear  LayerNorm → Linear
768 → 384 → 1       768 → 384 → 13
(binary logit)      (multi-label logits)
```

**Key design choices:**
- CNN stems use pretrained weights from publicly available MONAI Model Zoo checkpoints, adapting single-channel BraTS weights by extracting the relevant channel slice (T1c channel for T1post, T2 channel for T2w)
- The bridge layer projects from the stem output dimensionality to the SwinUNETR embedding dimension, allowing the pretrained transformer to receive properly-shaped token sequences
- Two separate classification heads allow independent optimization of the coarse task (aneurysm present?) and the fine-grained task (which artery?)

### Phase 2 — Scan-Level Hierarchical Transformer

After Phase 1 training, embeddings are extracted from the penultimate layer of the patch-level model for every patch in every scan. These embeddings (dim=768) are stored in an HDF5 file.

The scan-level model then performs **two-stage spatial aggregation**:

```
Patch Embeddings (N patches × 768)
for a full 3D scan
        │
        ▼
┌─────────────────────────────┐
│   Local Transformer Stage   │  Patches grouped into spatial blocks
│   Depth=2, Heads=6          │  (128×128×128 voxel regions)
│   + Sinusoidal pos embed     │
│   + CLS token per block      │
│   → Block-level embedding    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Global Transformer Stage  │  Block CLS tokens from full scan
│   Depth=4, Heads=6          │
│   + Sinusoidal pos embed     │
│   → Scan-level embedding     │
└────────────┬────────────────┘
             │
    ┌────────┴──────────┐
    ▼                   ▼
Aneurysm Head       Artery Head
LayerNorm → Linear  LayerNorm → Linear
384 → 1             384 → 13
```

This two-stage design allows the model to first understand local spatial relationships between neighboring patches (local transformer), then reason about the full anatomical extent of the scan (global transformer) — analogous to how a radiologist scans a full volume and then focuses on suspicious regions.

Positional embeddings are **sinusoidal** (non-learnable), making the model robust to variable numbers of patches across scans of different sizes.

---

## Loss Function

A custom **Hierarchical Focal Loss** (`HierarchicalAneurysmLoss`) was designed to handle:

1. **Class imbalance** — aneurysms are rare, most patches are negative
2. **Task hierarchy** — detecting an aneurysm is the primary task; localizing the artery is conditional on that
3. **Label semantics** — artery location labels benefit from soft treatment (label smoothing), while presence/absence is a hard binary label

```python
Total Loss = W_aneurysm × FocalLoss(aneurysm_logits, aneurysm_labels)
           + W_artery   × BCE(artery_logits[positive_mask], artery_labels[positive_mask])
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `W_aneurysm` | 5.0 | Presence detection is the primary task |
| `W_artery` | 1.0 | Location is conditional/secondary |
| Focal α | 0.25 | Handles positive/negative imbalance |
| Focal γ | 2.0 | Down-weights easy negatives |
| Artery loss | BCE (not Focal) | Allows soft label smoothing for anatomical similarity |

**Masking**: the artery location loss is only computed on patches where an aneurysm is present (`aneurysm_present_mask`). This prevents the artery head from receiving gradients on negative patches, enforcing the hierarchical dependency.

For the scan-level model, **AUCMLoss** (`libauc`) replaces standard BCE to directly optimize the area under the ROC curve — a better surrogate for the competition metric on imbalanced data.

---

## Training Strategy

### Hardware
- Kaggle notebooks with **2× NVIDIA T4 GPUs**
- **Distributed Data Parallel (DDP)** via `torch.distributed` + NCCL backend
- **Mixed precision** training with `torch.cuda.amp`
- Training done in **multiple sessions** (checkpointing between Kaggle notebook runs)

### Phase 1 Training (Patch-Level)

| Setting | Value |
|---------|-------|
| Batch size | 10 (per GPU) |
| Gradient accumulation | 2 steps → effective batch 40 |
| Phase 1 LR | 1e-3 (stem + head unfrozen) |
| Phase 2 LR | 5e-5 (full fine-tuning) |
| Scheduler | CosineAnnealingLR |
| Early stopping | Patience = 15 epochs |
| Dropout | 0.3 |

**Dynamic Undersampling**: each epoch, a fresh subset of negative patches is drawn at a 30:1 negative-to-positive ratio, ensuring the model sees every positive patch while preventing it from memorizing specific negatives.

**Augmentations (GPU-accelerated via MONAI):**
- Random flip along all 3 axes
- Random 90° rotations
- Random small-angle rotations (±25°)
- Random intensity scaling
- Random contrast adjustment (gamma correction)

### Phase 2 Training (Scan-Level)

| Setting | Value |
|---------|-------|
| Batch size | 120 embeddings |
| Epochs | up to 100 (early stopping) |
| LR | 5e-5 |
| Optimizer | AUCMLoss (libauc) |
| Early stopping | Patience = 6 epochs |
| Dropout | 0.2 |

---

## Results

> _Graphs and detailed metrics to be added. Placeholder sections below._

### Training Curves

<!-- Insert training loss and PR-AUC curves here -->
| Metric | Value |
|--------|-------|
| Patch-level PR-AUC (test) | *(add result)* |
| Scan-level ROC-AUC (test) | *(add result)* |

### Confusion Matrix & ROC Curves

<!-- Insert confusion matrix image here -->
<!-- Insert ROC and Precision-Recall curves here -->

### Per-Artery Location Performance

<!-- Insert per-class performance table or bar chart here -->

---

## Repository Structure

```
Aneurysm_detection/
│
├── preprocessing/
│   ├── preprocess_ct.py          # CTA preprocessing pipeline
│   ├── prep_mr.py                # MRA/MRI preprocessing pipeline
│   ├── save_cta.py               # Parallel CTA processing → HDF5
│   ├── save_mr.py                # Parallel MRA processing → HDF5
│   └── save_all.py               # Unified pipeline (all modalities)
│
├── patching/
│   ├── patching.py               # Patch manifest creation (NPY backend)
│   └── patching_hdf5.py          # Patch manifest creation (HDF5 backend)
│
├── models/
│   └── (saved model checkpoints)
│
├── training/
│   ├── training-aneurysm-kaggle.ipynb   # Phase 1: Patch-level training (Kaggle)
│   └── scan-level-model-rsna.ipynb      # Phase 2: Scan-level training (Kaggle)
│
├── clustering/
│   └── (data exploration & clustering notebooks)
│
├── utils/
│   ├── view3d_data.py            # 3D volume viewer with crosshair visualization
│   ├── verify_hdf5.py            # HDF5 integrity verification & slice extraction
│   └── savee_2d_images.py        # 2D slice export for visual inspection
│
├── mra.ipynb                     # MRA data exploration notebook
├── train_data.ipynb              # Training data analysis
├── test.ipynb                    # Inference / evaluation notebook
│
├── processed_data_unified/
│   ├── localization_manifest.csv # Final training labels with 3D coordinates
│   └── preprocessing_log.csv     # Per-scan processing status
│
└── aneurysm_dataset_manifests_hdf5_ho/
    ├── train_manifest.csv         # Patch-level training set
    └── test_manifest.csv          # Patch-level test set
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision monai[all] h5py SimpleITK pydicom \
            numpy pandas scikit-learn scipy tqdm libauc matplotlib seaborn
```

### Step 1 — Preprocess Scans

```bash
# Preprocess all modalities (CTA + MRA/MRI) in parallel → HDF5
python save_all.py
```

Configure paths in the `CONFIGURATION` block at the top of `save_all.py`:
- `BASE_PATH`: path to raw DICOM series
- `ORIGINAL_LOCALIZATION_CSV`: competition annotation CSV
- `OUTPUT_DIR`: output directory for HDF5 and manifests

### Step 2 — Generate Patch Manifests

```bash
# Create train/test patch manifests from the HDF5 file
python patching_hdf5.py
```

Configurable: `PATCH_SIZE`, `STRIDE`, `TRAIN_SIZE`, `HDF5_DATA_PATH`.

### Step 3 — Train Phase 1 (Patch-Level)

Open `training/training-aneurysm-kaggle.ipynb` in Kaggle (T4×2 recommended).  
The notebook writes `train_ddp.py` and runs it via `!python train_ddp.py`.

Key configuration in `Config` class:
- Set `HDF5_DATA_PATH`, `TRAIN_MANIFEST_HDF5`, `TEST_MANIFEST_HDF5`
- Set `SEGRESNET_WEIGHTS`, `SWIN_UNETR_WEIGHTS`, `BRATS_MRI_WEIGHTS` (MONAI Model Zoo)

### Step 4 — Extract Embeddings & Train Phase 2

After Phase 1 training, run the embedding extraction cell, then open `training/scan-level-model-rsna.ipynb` to train the scan-level `HierarchicalTransformer`.

### Visualize a Scan

```python
from view3d_data import view_3d_volume
import h5py, numpy as np

with h5py.File("processed_data_unified/processed_scans.hdf5", "r") as f:
    volume = f["<series_uid>"][()].astype(np.float32)

# Optional: pass known aneurysm coordinates as (z, y, x) tuples
view_3d_volume(volume, crosshair_coords=[(42, 128, 96)])
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, MONAI, SwinUNETR, SegResNet |
| Medical Imaging | SimpleITK, pydicom |
| Data Processing | NumPy, Pandas, SciPy, HDF5 (h5py) |
| Training Optimization | DDP (torch.distributed), AMP (torch.cuda.amp), libauc |
| Visualization | Matplotlib, Seaborn, ipywidgets |
| Infrastructure | Kaggle (T4×2 GPU), multiprocessing.Pool, file-lock concurrency |

---

## Key Engineering Highlights

- **Multi-modal transfer learning**: extracted modality-specific channel slices from BraTS pretrained weights to initialize single-channel MRI stems — allowing transfer from a 4-channel model without retraining from scratch.
- **Coordinate pipeline**: end-to-end DICOM annotation → 3D voxel coordinate transformation, handling both single-frame and multiframe DICOM edge cases including `PerFrameFunctionalGroupsSequence`.
- **Memory-efficient HDF5 pipeline**: float16 storage with chunking and gzip, combined with on-the-fly patch extraction — enabling training on a dataset too large to fit in RAM.
- **File-lock based safe parallel writes**: atomic file-based locking (`os.O_CREAT | os.O_EXCL`) for race-condition-free multi-process HDF5 writes without relying on a shared memory manager.
- **Hierarchical loss with conditional masking**: artery location loss computed only on positive samples, enforcing the medical prior that location is meaningless without presence.

---

## Acknowledgements

- [RSNA](https://www.rsna.org/) and [Kaggle](https://www.kaggle.com/) for organizing the competition and providing the dataset
- [MONAI](https://monai.io/) for pretrained medical imaging models (SegResNet, SwinUNETR, BraTS)
- [libauc](https://libauc.org/) for AUC-optimized loss functions

---

*This project was developed as part of the RSNA Intracranial Aneurysm Detection Kaggle Competition.*
