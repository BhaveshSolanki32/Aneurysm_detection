# 🧠 Intracranial Aneurysm Detection — RSNA Kaggle Competition

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA%20Aneurysm%20Detection-blue?logo=kaggle)](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange?logo=pytorch)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-green)](https://monai.io/)

> End-to-end deep learning pipeline for detecting and localizing intracranial aneurysms across **four imaging modalities** (CTA, MRA, MRI T1post, MRI T2). Two-stage architecture: a **patch-level Hybrid SwinUNETR classifier** trains on 96³ voxel patches, then a separate **scan-level Hierarchical Transformer** aggregates per-patch embeddings to make a final whole-scan prediction.

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
  - [Phase 2 — Embedding Extraction](#phase-2--embedding-extraction)
  - [Phase 3 — Scan-Level Hierarchical Transformer](#phase-3--scan-level-hierarchical-transformer)
- [Loss Function](#loss-function)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Technologies Used](#technologies-used)

---

## Background & Competition

The [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) competition challenged participants to automatically detect and localize intracranial aneurysms — balloon-like bulges in cerebral blood vessels — from medical brain scans. Undetected aneurysms can rupture, causing life-threatening hemorrhagic stroke.

The task required:
- **Binary detection**: Is an aneurysm present in this scan?
- **Multi-label localization**: Which of 13 arterial locations is the aneurysm at? (e.g., Left MCA, Basilar Tip, Anterior Communicating Artery, etc.)
- Handling a **heavily imbalanced dataset** across heterogeneous imaging modalities and scanner manufacturers.

---

## Project Journey

This competition was an intensive deep dive into 3D medical imaging. The focus was deliberately on understanding the data before modeling.

**Month 1 — Deep Data Exploration.** The dataset contained four imaging modalities — CTA, MRA, MRI T1post, MRI T2 — each with fundamentally different acquisition physics and visual characteristics. Before writing any model code, significant time was spent understanding what each modality looks like, how Hounsfield units behave in CT versus signal intensities in MRI, and how aneurysms manifest differently across scan types.

A key insight came from the scanner metadata: even within a single modality, scans from different manufacturers had vastly different intensity distributions, slice spacings, and resolutions. This drove an extended investigation into clustering and normalization strategies before converging on a robust approach.

**Preprocessing Design.** Separate pipelines were designed for CT-based (CTA) and MRI-based (MRA, T1post, T2) scans, each handling the specific physics and artifacts of that modality. Particular care went into the coordinate transformation logic — correctly mapping a radiologist's 2D DICOM pixel annotation into the 3D resampled voxel space, including handling multiframe DICOMs and `PerFrameFunctionalGroupsSequence` edge cases.

**Modeling.** The scale mismatch between aneurysms (millimeter-sized) and full brain scans (large 3D volumes) drove a two-stage design: a patch-level model learning local features, and a scan-level model that reasons globally across all patches.

---

## Dataset & Modalities

| Modality | Description | Key Challenge |
|----------|-------------|---------------|
| **CTA** | CT Angiography — high-res vascular contrast | HU windowing, bone/vessel separation |
| **MRA** | MR Angiography — vascular-sensitive MRI | Variable intensity, manufacturer differences |
| **MRI T1post** | Post-contrast T1-weighted MRI | Lower vascular contrast than CTA |
| **MRI T2** | T2-weighted MRI | Fluid-bright, different tissue contrast |

Each positive scan includes 2D pixel coordinates on a specific DICOM slice and a location label from 13 possible arterial sites. The dataset is heavily imbalanced — the vast majority of 96³ patches contain no aneurysm.

---

## Pipeline Overview

```
Raw DICOM Scans (CTA / MRA / MRI T1post / MRI T2)
         │
         ▼
┌──────────────────────────────────────┐
│   Modality-Specific Preprocessing    │
│   • DICOM series → 3D numpy volume   │
│   • Isotropic resampling             │
│   • Intensity normalization          │
│   • 2D annotation → 3D voxel coords  │
└─────────────────┬────────────────────┘
                  │
                  ▼
     HDF5 Storage (float16, gzip, chunked)
                  │
                  ▼
┌──────────────────────────────────────┐
│   3D Sliding Window Patching         │
│   96×96×96 patches, stride 64        │
│   → train/test manifest CSVs         │
│     (319,550 total patches)          │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Phase 1: Patch-Level Training       │   training-aneurysm-kaggle.ipynb
│  HybridAneurysmClassifier            │
│  • Modality-specific CNN stems       │
│  • Shared SwinUNETR transformer body │
│  • Dual output heads:                │
│    ├── Aneurysm Present (binary)     │
│    └── Artery Location (13-class)    │
└─────────────────┬────────────────────┘
                  │  best checkpoint saved
                  ▼
┌──────────────────────────────────────┐
│  Phase 2: Embedding Extraction       │   model-data-collection-patch.ipynb
│  HybridAneurysmEmbedder              │
│  • Same stems + SwinUNETR body       │
│  • No classification heads           │
│  • Returns 768-dim pooled embedding  │
│    per patch → extracted_embeddings  │
└─────────────────┬────────────────────┘
                  │  768-dim embeddings for all patches
                  ▼
┌──────────────────────────────────────┐
│  Phase 3: Scan-Level Training        │   scan-level-model-rsna.ipynb
│  HierarchicalTransformer             │
│  • Local Transformer (patches→block) │
│  • Global Transformer (blocks→scan)  │
│  • Final scan-level prediction       │
└──────────────────────────────────────┘
```

---

## Preprocessing

### CTA Pipeline (`preprocess_ct.py`)

- Reads DICOM series via SimpleITK, handles single-frame and multiframe formats
- Resamples to isotropic voxel spacing using B-spline interpolation
- Applies HU windowing for the vascular/brain window
- Maps 2D DICOM pixel annotations to 3D physical coordinates using `ImagePositionPatient` and `ImageOrientationPatient` DICOM tags

### MRA / MRI Pipeline (`prep_mr.py`)

- Handles diversity of MRI protocols and scanner manufacturers
- Modality-aware intensity normalization (rather than HU windowing)
- Resolves the frame number `f` embedded inside annotation `coords_xy` dictionaries for multiframe MRI DICOMs
- Uses `PerFrameFunctionalGroupsSequence` to correctly locate the annotated frame in multiframe acquisitions
- Peak-finding heuristics via `scipy.signal.find_peaks` to identify the correct slice from manufacturer-provided frame indices

### Storage

All processed scans stored in a single HDF5 file:
- `float16` precision to halve storage size
- Chunked `(32, 32, 32)` for efficient random patch reads during training
- GZIP compression
- File-lock mechanism (`*.lock` via `os.O_CREAT | os.O_EXCL`) for safe concurrent multi-process writes

---

## 3D Patch Strategy

| Parameter | Value |
|-----------|-------|
| Patch size | 96 × 96 × 96 voxels |
| Stride | 64 voxels |
| Total patches (train + test) | 319,550 |
| Label assignment | Aneurysm present if coordinate falls inside patch bounds |
| Artery labels | Bitwise OR across all aneurysms within patch |
| Train / Test split | 80% / 20% at the scan (patient) level |

Patches are not stored on disk. **Manifest CSVs** record `(series_uid, start_z, start_y, start_x)` per patch and patches are extracted on-the-fly from HDF5 at training time.

---

## Model Architecture

### Phase 1 — Patch-Level Hybrid Classifier

**The problem**: different modalities have completely different low-level features (HU ranges, vessel contrast, noise patterns), but aneurysm morphology is consistent across them. The solution is modality-specific CNN stems feeding into a single shared Swin Transformer body.

```
Input: 96×96×96 single-channel patch
             │
             ▼
┌────────────────────────────────────┐
│    Modality-Specific CNN Stem      │  Pretrained weights from MONAI Model Zoo:
│                                    │
│  CTA  → SegResNet first 4 layers   │  wholeBody_ct_segmentation
│  MRA  → (same as CTA)              │
│  T1c  → BraTS T1c channel slice    │  brats_mri_segmentation (channel 1)
│  T2w  → BraTS T2w channel slice    │  brats_mri_segmentation (channel 2)
│         + 1×1×1 conv proj → 128ch  │
│  Output: 128 channels              │
└─────────────────┬──────────────────┘
                  │
        Linear Bridge: 128 → 768
        GELU + Dropout(0.3)
                  │
                  ▼
┌────────────────────────────────────┐
│    SwinUNETR Transformer Body      │  Pretrained: swin_unetr_btcv_segmentation
│    Shared across all modalities    │
│    4 Swin layers → Global AvgPool  │
│    Output: 768-dim vector          │
└───────────────┬────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
 Aneurysm Head       Artery Head
 LayerNorm           LayerNorm
 768 → 384 → 1       768 → 384 → 13
 (binary logit)      (multi-label logits)
```

The MRI stems extract a single relevant input channel from BraTS pretrained weights (channel 1 for T1c, channel 2 for T2w), then project the 32-channel output to 128 via a 1×1×1 conv — enabling partial weight reuse from a 4-channel segmentation model.

### Phase 2 — Embedding Extraction

After Phase 1 training, a stripped version `HybridAneurysmEmbedder` removes both classification heads and returns the **768-dim global average pool** of the SwinUNETR output for every patch. Run across 2 GPUs via `DataParallel`.

- **319,550 patches** processed → `extracted_embeddings.hdf5`
- Extraction runtime: ~68 minutes on 2× T4 GPUs
- Output: 768-dim embedding per patch, plus coordinates in `embeddings_manifest.csv`

### Phase 3 — Scan-Level Hierarchical Transformer

The scan-level model performs two-stage spatial aggregation over all patch embeddings for a given scan:

```
Per-patch embeddings (768-dim) for all N patches in a scan
             │
             ▼
┌────────────────────────────────────┐
│    Local Transformer Stage         │
│    Depth=2, Heads=6, MLP dim=768   │
│    Patches grouped into spatial    │
│    blocks (128×128×128 voxel region│
│    + Sinusoidal positional embed   │
│    + learnable CLS token per block │
│    → One 768-dim CLS per block     │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│    Global Transformer Stage        │
│    Depth=4, Heads=6, MLP dim=768   │
│    All block CLS tokens as input   │
│    + Sinusoidal positional embed   │
│    → Scan-level 768-dim vector     │
└───────────────┬────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
 Aneurysm Head       Artery Head
 LayerNorm           LayerNorm
 768 → 384 → 1       768 → 384 → 13
```

Sinusoidal (non-learnable) positional embeddings with max length 256 make the model robust to variable numbers of patches across scans of different sizes. Training begins with a 1-epoch **alignment phase** (frozen backbone, eval mode) before full fine-tuning — post-alignment baseline was **0.4742**.

---

## Loss Function

### Phase 1 — Hierarchical Focal Loss

Custom `HierarchicalAneurysmLoss` handles class imbalance and the hierarchical dependency between tasks:

```
Total Loss = W_aneurysm × FocalLoss(aneurysm_logits, aneurysm_labels)
           + W_artery   × BCE(artery_logits[positive_mask], artery_labels[positive_mask])
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `W_aneurysm` | 5.0 | Detection is the primary task |
| `W_artery` | 1.0 | Localization is conditional/secondary |
| Focal α | 0.25 | Positive/negative class weight |
| Focal γ | 2.0 | Down-weights easy negatives |
| Artery loss fn | BCE (not Focal) | Supports soft label treatment for anatomically similar locations |

Artery location loss is only computed on positive-mask samples — no gradient flows to the artery head from negative patches, enforcing the medical prior that location is undefined without presence.

### Phase 3 — AUCMLoss

The scan-level model uses `AUCMLoss` (from `libauc`) for both aneurysm and artery tasks, directly optimizing the AUC metric rather than a proxy, which is better suited to the severe class imbalance at scan level.

---

## Training Strategy

### Hardware

- Kaggle notebooks, **2× NVIDIA T4 GPUs**
- Phase 1: **Distributed Data Parallel** via `torch.distributed` + NCCL
- Phase 2 extraction: `torch.nn.DataParallel`
- **Mixed precision** (`torch.cuda.amp`) throughout
- Training across **multiple Kaggle sessions** with checkpoint resumption

### Phase 1 — Patch-Level

| Parameter | Value |
|-----------|-------|
| Batch size | 10 per GPU |
| Gradient accumulation | 2 steps (effective batch: 40) |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 5e-5 |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 15 epochs |
| Dropout | 0.3 |
| Dynamic negative sampling | 30:1 (negatives : positives per epoch) |

**Dynamic Undersampling**: each epoch draws a fresh random subset of negatives at a 30:1 ratio. Every positive patch is always seen; negatives are rotated across sessions to prevent memorization of specific negative patches.

**GPU Augmentations (MONAI):** random flips (all 3 axes), random 90° rotations, random small-angle rotations (±25°), random intensity scaling, random gamma contrast adjustment.

### Phase 3 — Scan-Level

| Parameter | Value |
|-----------|-------|
| Train / Test scans | 3,456 / 864 |
| Batch size | 120 embeddings |
| Max epochs | 100 |
| Fine-tuning LR | 5e-6 |
| Early stopping patience | 6 epochs |
| Dropout | 0.2 |

---

## Results

### Phase 1 — Patch-Level Model (64,784 test patches)

| Metric | Value |
|--------|-------|
| **Aneurysm Present — ROC AUC** | **0.9107** |
| Aneurysm Present — PR AUC | 0.2495 |
| Artery Location — ROC AUC (Micro) | 0.7630 |

> The low patch-level PR AUC reflects extreme class imbalance — the vast majority of patches contain no aneurysm. ROC AUC is the more meaningful patch-level metric.

---

### Phase 3 — Scan-Level Model (864 test scans)

| Metric | Value |
|--------|-------|
| **Aneurysm Present — ROC AUC** | **0.8540** |
| **Aneurysm Present — PR AUC** | **0.8221** |
| Mean Artery Location — ROC AUC | 0.6109 |
| **Final Combined Score** | **0.7325** |

**Aneurysm detection at 0.5 threshold:** Precision 0.92 · Recall 0.50 · F1 0.65 (320 positive scans in test set)

> The scan-level model shows a large improvement in PR AUC (0.25 → 0.82) by aggregating global context across the full volume. The 0.92 precision at 0.5 threshold indicates that when the model predicts positive, it is very likely correct. Recall can be improved with threshold tuning. Artery localization (mean AUC 0.61) is the harder sub-task, inherently dependent on the detection step being correct first.

---

### Visualizations

<!-- Add training history plot here: visualizations/training_history.png -->
**Training Loss & Score Curves**

<!-- Add ROC and PR curves here: visualizations/roc_pr_curves.png -->
**ROC and Precision-Recall Curves**

<!-- Add confusion matrix here: visualizations/confusion_matrix_and_distribution.png -->
**Confusion Matrix & Score Distribution**

---

## Repository Structure

```
Aneurysm_detection/
│
├── preprocess_ct.py                   # CTA preprocessing pipeline
├── prep_mr.py                         # MRA / MRI preprocessing pipeline
├── save_cta.py                        # Parallel CTA processing → HDF5
├── save_mr.py                         # Parallel MRA processing → HDF5
├── save_all.py                        # Unified pipeline (all modalities)
│
├── patching.py                        # Patch manifest creation (NPY backend)
├── patching_hdf5.py                   # Patch manifest creation (HDF5 backend)
├── nii_to_npy.py                      # NIfTI to NumPy conversion utility
│
├── training-aneurysm-kaggle.ipynb     # Phase 1: Patch-level DDP training
├── model-data-collection-patch.ipynb  # Phase 2: Embedding extraction
├── scan-level-model-rsna.ipynb        # Phase 3: Scan-level training
│
├── mra.ipynb                          # MRA data exploration
├── train_data.ipynb                   # Training data analysis
├── test.ipynb                         # Evaluation / inference notebook
│
├── view3d_data.py                     # Interactive 3D volume viewer with crosshairs
├── verify_hdf5.py                     # HDF5 integrity check + slice export
├── savee_2d_images.py                 # 2D slice export for visual inspection
│
├── clustering/                        # Data exploration & clustering notebooks
├── models/                            # Saved model checkpoints
│
├── processed_data_unified/
│   ├── localization_manifest.csv      # 3D coordinates + labels for all scans
│   └── preprocessing_log.csv          # Per-scan processing status
│
└── aneurysm_dataset_manifests_hdf5_ho/
    ├── train_manifest.csv             # Patch-level training set
    └── test_manifest.csv              # Patch-level test set
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
python save_all.py
```

Configure paths at the top of `save_all.py`: `BASE_PATH` (raw DICOMs), `ORIGINAL_LOCALIZATION_CSV`, `OUTPUT_DIR`.

### Step 2 — Generate Patch Manifests

```bash
python patching_hdf5.py
```

Key settings: `PATCH_SIZE=96`, `STRIDE=64`, `TRAIN_SIZE=0.8`, `HDF5_DATA_PATH`.

### Step 3 — Phase 1: Train Patch-Level Model

Open `training-aneurysm-kaggle.ipynb` on Kaggle (2× T4 recommended). Requires MONAI Model Zoo pretrained weights for `wholeBody_ct_segmentation`, `swin_unetr_btcv_segmentation`, and `brats_mri_segmentation`.

### Step 4 — Phase 2: Extract Embeddings

Open `model-data-collection-patch.ipynb` on Kaggle. Loads the best Phase 1 checkpoint and extracts 768-dim embeddings for all 319,550 patches into `extracted_embeddings.hdf5`.

### Step 5 — Phase 3: Train Scan-Level Model

Open `scan-level-model-rsna.ipynb` on Kaggle. Trains `HierarchicalTransformer` over 3,456 training scans using the extracted embeddings.

### Visualize a Scan

```python
from view3d_data import view_3d_volume
import h5py, numpy as np

with h5py.File("processed_data_unified/processed_scans.hdf5", "r") as f:
    volume = f["<series_uid>"][()].astype(np.float32)

# Pass known aneurysm coordinates as (z, y, x) tuples to see crosshair views
view_3d_volume(volume, crosshair_coords=[(42, 128, 96)])
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch 2.6, MONAI, SwinUNETR, SegResNet |
| Medical Imaging | SimpleITK, pydicom |
| Data Processing | NumPy, Pandas, SciPy, HDF5 (h5py) |
| Training | DDP (`torch.distributed`), AMP (`torch.cuda.amp`), libauc (AUCMLoss) |
| Visualization | Matplotlib, Seaborn, ipywidgets |
| Infrastructure | Kaggle (2× T4), `multiprocessing.Pool`, file-lock concurrency |

---

## Key Engineering Highlights

- **Multi-modal transfer learning via channel slicing**: extracted single-channel slices from a 4-channel BraTS pretrained model to initialize MRI stems, reusing learned feature representations without retraining from scratch.
- **Three-stage decoupled pipeline**: patch classification → embedding extraction → scan-level aggregation, each stage producing artifacts consumed by the next, coordinated across multiple Kaggle sessions via checkpointing.
- **End-to-end DICOM coordinate pipeline**: 2D pixel annotation → physical 3D point → resampled voxel coordinate, handling single-frame and multiframe DICOMs, `PerFrameFunctionalGroupsSequence`, and varying `ImagePositionPatient` origins across manufacturers.
- **Memory-efficient on-the-fly HDF5 patching**: float16 + chunked gzip storage with patch extraction at training time, enabling training on a dataset too large to fit in RAM.
- **Atomic file-lock for parallel HDF5 writes**: `os.O_CREAT | os.O_EXCL` provides kernel-level atomic lock acquisition, preventing race conditions across 8 concurrent workers without a shared memory manager.
- **Hierarchical conditional loss**: artery location loss backpropagates only through positive-mask samples, enforcing the medical prior that location is meaningless without presence.

---

## Acknowledgements

- [RSNA](https://www.rsna.org/) and [Kaggle](https://www.kaggle.com/) for the competition and dataset
- [MONAI](https://monai.io/) for pretrained medical imaging models (SegResNet, SwinUNETR, BraTS)
- [libauc](https://libauc.org/) for AUC-optimized loss functions
