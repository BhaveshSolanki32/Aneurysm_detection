# 🧠 Intracranial Aneurysm Detection — RSNA Kaggle Competition

> A full end-to-end deep learning pipeline for detecting and localizing intracranial aneurysms from multi-modal 3D medical scans (CTA and MRA), built for the [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) Kaggle competition.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [The Challenge](#the-challenge)
3. [Dataset & Data Exploration](#dataset--data-exploration)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Patch-Level Model — Stage 1](#patch-level-model--stage-1)
6. [Scan-Level Model — Stage 2](#scan-level-model--stage-2)
7. [Training Strategy](#training-strategy)
8. [Results](#results)
9. [Repository Structure](#repository-structure)
10. [Tech Stack](#tech-stack)
11. [How to Run](#how-to-run)
12. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

Intracranial aneurysms are life-threatening vascular abnormalities — a rupture is fatal in roughly 40% of cases. Yet detection from 3D brain imaging is time-consuming and depends heavily on radiologist experience. This project builds a **two-stage deep learning detection system** that:

1. **Stage 1 (Patch-Level)** — Scans 3D brain volumes in fixed 96³ voxel patches and classifies each patch for aneurysm presence and artery location using a **Hybrid CNN-SwinUNETR** model.
2. **Stage 2 (Scan-Level)** — Aggregates patch-level embeddings across the full scan via a **hierarchical transformer** (local + global attention) to produce a final scan-level diagnosis.

The pipeline handles both **CTA** and **MRA** modalities with modality-specific preprocessing branches, runs distributed training across dual GPUs using PyTorch DDP, and was fully developed and trained on Kaggle (T4×2).

---

## The Challenge

The RSNA competition presented several hard technical problems:

- **Multi-modal data**: CTA and MRA have fundamentally different intensity characteristics, voxel spacings, and acquisition protocols — and cannot be treated identically.
- **Scanner heterogeneity**: Scans came from different manufacturers and machines, causing significant distribution shift even within the same modality.
- **Extreme class imbalance**: The ratio of negative patches (no aneurysm) to positive patches reached 1000:1 in raw data.
- **Small lesions in large volumes**: Aneurysms are only a few millimeters — tiny relative to the full 3D volume.
- **Multi-label output**: The model predicts not just *whether* an aneurysm is present, but *which artery* it belongs to (13 possible anatomical locations).
- **Compute constraints**: Full 3D volumes exceeded GPU memory, requiring a careful patch-based approach with efficient HDF5 I/O.

---

## Dataset & Data Exploration

The dataset contains thousands of 3D brain scans across two imaging modalities:

| Modality | Description |
|----------|-------------|
| **CTA** | CT Angiography — contrast-enhanced CT highlighting blood vessels, with standardized Hounsfield Units |
| **MRA** | MR Angiography — MRI-based vascular imaging; includes TOF-MRA and contrast-enhanced variants with arbitrary intensity scales |

### Deep Dive into the Data

I spent the first month of the project on data exploration before touching any model. The key discoveries:

- **CTA scans** have a standardized HU scale (~-1000 to +3000), and a vascular window (roughly 400–600 HU) is optimal for highlighting aneurysms.
- **MRA scans** have no HU scale — intensities are manufacturer-dependent and scanner-specific. TOF-MRA and CE-MRA require completely different normalization strategies.
- Even within MRA, scans from different scanners had different orientation conventions and wildly different slice spacings.
- **Clustering analysis** (see `clustering/`) was used to identify distinct data subgroups and inform preprocessing strategy before writing a single line of model code.
- The localization labels (`train_localizers.csv`) provided 2D pixel coordinates per DICOM slice, which required transformation through the DICOM geometry matrices into 3D voxel coordinates — a step that had to be done *after* resampling.

This month of exploration was the single biggest factor in building a reliable preprocessing pipeline.

---

## Data Preprocessing Pipeline

All scans are preprocessed into a unified HDF5 file (`processed_scans.hdf5`) stored as float16 to minimize disk and memory footprint. The pipeline runs in parallel using Python `multiprocessing` with a file-based lock for safe concurrent HDF5 writes.

### CTA Preprocessing (`preprocess_ct.py`)

1. Load DICOM series using SimpleITK
2. Resample to isotropic voxel spacing
3. Apply vascular HU windowing (clip + normalize)
4. Transform per-slice 2D localization coordinates into 3D voxel space using DICOM geometry
5. Save as float16 to HDF5

### MRA Preprocessing (`prep_mr.py`)

1. Load DICOM series using SimpleITK
2. Detect acquisition sub-type from DICOM metadata (TOF-MRA, CE-MRA, etc.)
3. Apply sub-type-specific normalization (percentile clipping, z-score, or min-max)
4. Resample to target isotropic spacing
5. Transform localization coordinates to 3D space
6. Save as float16 to HDF5

### Patch Manifest Generation (`patching_hdf5.py`)

After preprocessing, every scan is sliced into overlapping 3D patches and a manifest CSV is generated:

| Setting | Value |
|---------|-------|
| Patch size | 96 × 96 × 96 voxels |
| Stride | 64 voxels (overlapping) |
| Train/Test split | 80/20 patient-level (no data leakage) |
| Manifest columns | `series_uid`, `start_z/y/x`, `Aneurysm Present`, `relative_coords`, 13× artery labels |

At runtime, patches are read directly from HDF5 using coordinate offsets — no patch files are ever written to disk.

---

## Patch-Level Model — Stage 1

📓 **Kaggle Notebook**: [`training-aneurysm-kaggle`](https://www.kaggle.com/code/bhaveshsolanki32/training-aneurysm-kaggle)

### Architecture: Hybrid CNN-SwinUNETR Classifier

The model (`HybridAneurysmClassifier`) has three components:

#### 1. Modality-Specific CNN Stems

Each modality (CTA, MRA) gets its own lightweight CNN stem that learns modality-specific low-level features and projects them to a shared embedding space (`128 → 768` via a GELU linear bridge). This lets the shared backbone focus on task-relevant features while the stems absorb imaging-specific variance.

#### 2. SwinUNETR Backbone

A pretrained `SwinUNETR` from MONAI serves as the shared 3D feature extractor. It processes 96³ patches and outputs a 768-dimensional patch embedding.

- **Embed dim**: 768
- **Pretrained on**: BTCV whole-body CT segmentation + BraTS MRI segmentation
- **Architecture**: Hierarchical Swin Transformer with shifted-window attention

#### 3. Multi-Label Classification Head

The 768-dim embedding feeds into a multi-label head producing **14 outputs**:

- 1 binary output: *Aneurysm Present* (yes/no)
- 13 binary outputs: *which artery* (Left/Right ICA, MCA, ACA, AComA, PComA, Basilar Tip, etc.)

### Loss: Hierarchical Focal Loss

A custom loss function combines two objectives with separate weighting:

```
L = W_aneurysm × FocalLoss(detection_pred, detection_label)
  + W_artery   × FocalLoss(artery_preds,   artery_labels)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `W_aneurysm` | 5.0 | Detection is the primary task |
| `W_artery` | 1.0 | Location is secondary |
| Focal `alpha` | 0.25 | Handles positive/negative imbalance |
| Focal `gamma` | 2.0 | Down-weights easy negatives |

Focal Loss was essential — standard cross-entropy was dominated by the flood of easy negatives and failed to learn meaningful signal.

---

## Scan-Level Model — Stage 2

📓 **Embedding Extraction**: [`model-data-collection-patch`](https://www.kaggle.com/code/bhaveshsolanki32/model-data-collection-patch)  
📓 **Scan-Level Training**: [`scan-level-model-rsna`](https://www.kaggle.com/code/bhaveshsolanki32/scan-level-model-rsna/notebook)

### From Patches to Scans

The patch model classifies each 96³ sub-volume, but a clinical diagnosis requires a **scan-level prediction**: is there an aneurysm anywhere in this patient's brain?

This is a classic **Multiple Instance Learning (MIL)** problem: given a bag of patch embeddings, predict the bag label.

### Pipeline

**Step 1 — Embedding Extraction**  
The trained patch-level model runs in inference mode over all patches of every scan. The 768-dim feature vector from a penultimate layer is extracted and stored by scan in HDF5 (`extracted_embeddings.hdf5`).

**Step 2 — Hierarchical Scan Transformer**  
The scan-level model processes the full bag of patch embeddings for a scan:

- **Local Transformer** (depth=2, heads=6): Attends within spatial blocks of `128³` voxels, capturing neighborhood context.
- **Global Transformer** (depth=4, heads=6): Attends across all blocks, integrating whole-brain evidence.
- **Classification Head**: 14 predictions (same label space as patch model).

**Step 3 — AUC-Maximizing Loss**  
Training uses `AUCMLoss` from the `libauc` library, which directly maximizes AUROC. This aligns the training objective with the evaluation metric, unlike cross-entropy which optimizes accuracy.

---

## Training Strategy

Training was done in **multiple sequential sessions** due to Kaggle's GPU time limit (~9 hours per session on T4×2). Checkpoints were saved after each session and resumed in the next.

### Patch Model — Two-Phase Training

| Phase | Epochs | LR | Frozen? |
|-------|--------|-----|---------|
| Phase 1 — Warmup | 6 | `1e-3` | SwinUNETR frozen (train stems + head only) |
| Phase 2 — Fine-tune | 8 | `5e-5` | Full model unfrozen |

### Optimization Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Gradient Accumulation | 2 steps |
| Early Stopping | Patience = 15 epochs |
| Mixed Precision | float16 |
| Distributed Training | PyTorch DDP — 2× T4 GPUs |
| Dynamic Undersampling | 30 negatives per positive per epoch |
| Batch Size | 10 (patch model), 120 (scan model) |

### Data Augmentation

Applied to patches during training:
- Random axis-aligned flips (3 axes)
- Random 90° rotations
- Small random rotations (±15°)
- Random contrast adjustment
- Random intensity scaling
- Spatial padding to enforce fixed 96³ size

---

## Results

> 📊 *Add metric values and plots from training_history.json after training runs complete. Placeholder structure below.*

### Patch-Level Model

| Metric | Train | Validation |
|--------|-------|------------|
| Aneurysm Detection AUROC | — | — |
| Mean Artery Location AUROC | — | — |
| Average Precision (AP) | — | — |

**Training Curves**  
<!-- Insert training_loss_curve.png here -->

**ROC Curve — Aneurysm Detection**  
<!-- Insert roc_curve_aneurysm.png here -->

**Per-Artery AUROC Breakdown**  
<!-- Insert per_class_auroc_bar.png here -->

### Scan-Level Model

| Metric | Value |
|--------|-------|
| Scan-Level AUROC | — |
| Scan-Level Average Precision | — |

---

## Repository Structure

```
Aneurysm_detection/
│
├── 📄 preprocess_ct.py               # CTA preprocessing: HU windowing, resampling, coord transform
├── 📄 prep_mr.py                     # MRA preprocessing: modality-aware normalization
├── 📄 save_cta.py                    # Parallel CTA processing → NIfTI
├── 📄 save_mr.py                     # Parallel MRA processing → HDF5
├── 📄 save_all.py                    # Unified CTA+MRA dispatcher → single HDF5 (batched, fault-tolerant)
│
├── 📄 patching.py                    # Patch manifest generator (NPY-based)
├── 📄 patching_hdf5.py               # Patch manifest generator (HDF5-based — used in final pipeline)
│
├── 📄 view3d_data.py                 # Interactive 3D volume + aneurysm viewer (ipywidgets)
├── 📄 verify_hdf5.py                 # QA: extract JPEG slices from HDF5 for visual inspection
├── 📄 savee_2d_images.py             # Extract 2D slices from NIfTI files
├── 📄 nii_to_npy.py                  # NIfTI → NumPy conversion utility
│
├── 📁 clustering/                    # Data exploration: modality clustering analysis
├── 📁 models/                        # Saved model checkpoints (not tracked by git LFS)
│
├── 📁 aneurysm_dataset_manifests_hdf5_ho/
│   ├── train_manifest.csv            # Patch-level training manifest
│   └── test_manifest.csv             # Patch-level test manifest
│
├── 📁 processed_data_unified/
│   ├── localization_manifest.csv     # Scan-level labels with 3D coordinates
│   └── preprocessing_log.csv        # Per-scan preprocessing status
│
├── 📄 mra.ipynb                      # MRA data exploration
├── 📄 train_data.ipynb               # Training data analysis
└── 📄 test.ipynb                     # Model evaluation
```

---

## Tech Stack

| Category | Library / Tool |
|----------|---------------|
| **Deep Learning** | PyTorch, MONAI |
| **Architecture** | SwinUNETR, SegResNet (MONAI pretrained) |
| **Medical Imaging I/O** | SimpleITK, h5py, NiBabel |
| **Training Infra** | PyTorch DDP, CosineAnnealingLR, gradient accumulation |
| **Loss Functions** | Custom Hierarchical Focal Loss, AUCMLoss (libauc) |
| **Augmentation** | MONAI Transforms |
| **Data** | HDF5, NumPy, Pandas |
| **Metrics & Eval** | scikit-learn (ROC AUC, AP, confusion matrix, classification report) |
| **Visualization** | Matplotlib, Seaborn, ipywidgets, PIL |
| **Parallelism** | Python `multiprocessing`, `concurrent.futures` |

---

## Kaggle Notebooks

| Notebook | Purpose |
|----------|---------|
| [training-aneurysm-kaggle](https://www.kaggle.com/code/bhaveshsolanki32/training-aneurysm-kaggle) | Patch-level model training (DDP, multi-phase, full evaluation) |
| [model-data-collection-patch](https://www.kaggle.com/code/bhaveshsolanki32/model-data-collection-patch) | Patch embedding extraction for scan-level model input |
| [scan-level-model-rsna](https://www.kaggle.com/code/bhaveshsolanki32/scan-level-model-rsna/notebook) | Scan-level hierarchical transformer training and evaluation |

---

## How to Run

### Prerequisites

```bash
pip install torch monai[all] SimpleITK h5py pandas scikit-learn tqdm libauc
```

### 1. Preprocess All Scans

```bash
# Edit BASE_PATH and OUTPUT_HDF5_PATH in save_all.py, then:
python save_all.py
```

Processes all CTA and MRA scans in parallel, dispatching to modality-specific pipelines. Saves everything to a single HDF5 file. Fault-tolerant: already-processed scans are skipped on re-run.

### 2. Generate Patch Manifests

```bash
# Edit HDF5_DATA_PATH and LOCALIZATION_CSV_PATH in patching_hdf5.py, then:
python patching_hdf5.py
```

Outputs `train_manifest.csv` and `test_manifest.csv` with all patch coordinates and labels.

### 3. Train Patch-Level Model

Upload the HDF5 file and manifests to Kaggle, then run:  
[`training-aneurysm-kaggle`](https://www.kaggle.com/code/bhaveshsolanki32/training-aneurysm-kaggle)

### 4. Extract Patch Embeddings

With the trained patch model checkpoint, run:  
[`model-data-collection-patch`](https://www.kaggle.com/code/bhaveshsolanki32/model-data-collection-patch)

### 5. Train Scan-Level Model

Upload `extracted_embeddings.hdf5`, then run:  
[`scan-level-model-rsna`](https://www.kaggle.com/code/bhaveshsolanki32/scan-level-model-rsna/notebook)

### Visualize Data

```python
# Interactive 3D viewer with aneurysm crosshair overlay (in Jupyter)
from view3d_data import view_3d_volume
import h5py

with h5py.File("processed_scans.hdf5", "r") as f:
    vol = f["<series_uid>"][()]

view_3d_volume(vol, crosshair_coords=[(z, y, x)])
```

```bash
# Quick visual QA — extract JPEG slices from all scans in HDF5
python verify_hdf5.py
```

---

## Key Design Decisions

**Why HDF5 over individual NIfTI files?**  
HDF5 enables random-access patch reads — during training, only the 96³ patch being loaded is read from disk, not the whole scan. With millions of patches across thousands of scans, this is the difference between a practical training loop and an I/O bottleneck.

**Why modality-specific CNN stems?**  
CTA intensities are in Hounsfield Units with physical meaning; MRA intensities are arbitrary and scanner-dependent. A shared first layer trained on raw voxels would produce incompatible representations. The stems act as learned normalization layers, projecting each modality into a shared feature space before the SwinUNETR sees them.

**Why a two-stage pipeline rather than end-to-end 3D?**  
Full brain volumes (512×512×200+ voxels) do not fit in T4 GPU memory for 3D convolutions at useful batch sizes. More importantly, the two-stage design mirrors radiologist workflow: *find suspicious patches*, then *make a global judgment across the scan*. The MIL framing is also better suited for the training signal available (scan-level labels with noisy 3D localization).

**Why dynamic undersampling rather than class weighting?**  
At 30:1 negative-to-positive sampling per epoch, the model sees a realistic but manageable number of negatives. Class weighting alone inflated gradients from negatives in early training, causing instability. Dynamic undersampling proved more stable and gave the model a better implicit curriculum.

**Why AUCMLoss for the scan-level model?**  
Clinical deployment cares about ranking (is this scan more likely to have an aneurysm than that one?) not about a fixed threshold. AUROC directly measures ranking quality. AUCMLoss optimizes AUROC end-to-end, aligning training and evaluation — cross-entropy was a proxy at best.

---

## Author

**Bhavesh Solanki**  
B.Tech Student — NSUT (Netaji Subhas University of Technology), Delhi  
Published IEEE Researcher — *ECG Anomaly Detection Using Machine Learning: Comparative Analysis* (IEEE DELCON 2025)

[Kaggle](https://www.kaggle.com/bhaveshsolanki32) · [GitHub](https://github.com/BhaveshSolanki32)

---

## Acknowledgements

- [RSNA](https://www.rsna.org/) for organizing the competition and providing the dataset
- [MONAI](https://monai.io/) for the SwinUNETR implementation and medical imaging pretrained weights
- [libauc](https://libauc.org/) for the AUCMLoss implementation
- The Kaggle community for open discussion and shared insights on medical imaging competitions
