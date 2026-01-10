# Medical Image Segmentation – Liver & Tumor (CNN)

## Project overview

This project focuses on **multiclass semantic segmentation** of liver and liver tumors
from abdominal CT scans.  
The objective is to build a **robust, efficient and clinically realistic pipeline**
covering data preprocessing, model training, evaluation, and error analysis.

Classes:
- 0: background
- 1: liver
- 2: tumor

---

## Dataset

**Source**: LiTS (Liver Tumor Segmentation Challenge)

- 3D CT volumes (`.nii`)
- Pixel-wise annotations for liver and tumor
- High class imbalance (tumor ≪ liver ≪ background)

---

## Data preprocessing & optimization

### Motivation

Training directly from `.nii` files is slow due to:
- heavy disk I/O
- repeated decompression
- large 3D volumes

To improve training speed and stability, a dedicated preprocessing pipeline was built.

---

### Conversion to `.npy` (2D slice-based dataset)

Each 3D volume is converted into **2D axial slices** and saved as NumPy arrays.

Steps:
1. Patient-level split into **train / val / test** (no data leakage)
2. Slice selection:
   - keep slices containing tumor or liver
   - optionally add a small ratio of empty slices
3. CT normalization:
   - Hounsfield Unit clipping
   - min-max normalization to `[0,1]`
4. Multiclass masks preserved as integers `{0,1,2}`

Benefits:
- Much faster dataloading
- Reduced memory footprint
- Reproducible patient-level splits

---

## Training pipeline

### Frameworks & tools

- **PyTorch Lightning** – structured and reproducible training
- **segmentation-models-pytorch**
- **Weights & Biases (wandb)** – experiment tracking
- **Hydra** – configuration management

---

### DataLoader & Lightning DataModule

- Custom `LightningDataModule`
- Efficient batching
- Multiprocessing enabled
- Persistent workers & pinned memory

This avoids GPU under-utilization and stabilizes throughput.

---

### Model architecture

**U-Net** with pretrained encoder:

- Encoder: `ResNet34` (ImageNet weights)
- Decoder: standard U-Net
- Input: 3-channel (duplicated CT slice)
- Output: 3-class logits (background / liver / tumor)

The pretrained encoder significantly improves convergence and stability.

---

### Loss function

A **combined loss** is used to handle class imbalance:

- Cross-entropy: global pixel-wise supervision
- Dice loss: overlap-focused, crucial for small tumors

---

### Optimization

- Optimizer: `AdamW`
- Mixed precision training (bf16 / fp16)
- Early stopping on validation loss
- Learning rate scheduling

---

## Evaluation metrics

Evaluation is performed on the **test set only**, with no patient overlap.

Metrics:
- Dice score (macro & per class)
- Intersection over Union (IoU)

---

## Quantitative results (summary)

| Metric | Value |
|------|------|
| Mean Dice (all classes) | high |
| Dice – Liver | very high (stable anatomy) |
| Dice – Tumor | highly variable |
| IoU | consistent with Dice trends |

---

## Tumor segmentation analysis

### Dice score distribution

The Dice score distribution for tumor segmentation is **bimodal**:

- Many slices with Dice ≈ 0
- A second peak with Dice ≈ 0.7–0.9

This behavior is **expected and realistic** in medical imaging.

#### Interpretation:
- Dice is extremely sensitive to **small objects**
- Small tumors or borderline slices lead to near-zero Dice
- Large, well-defined tumors are segmented accurately

---

### Qualitative analysis

Findings:
- Excellent liver segmentation across most samples
- Tumor segmentation succeeds on large lesions
- Failures mostly occur on:
  - very small tumors
  - ambiguous boundaries
  - low contrast regions


## Error analysis & limitations

Main challenges:
- Severe class imbalance
- Dice instability on very small tumors
- Slice-based (2D) context limitation

These limitations are common in real-world clinical segmentation tasks.

---

## Possible improvements (future work)

- Focal or Focal-Tversky loss
- Tumor-focused slice sampling
- 2.5D or 3D context modeling
- Post-processing to remove small false positives
- Tumor-size–aware evaluation

---

## Conclusion

This project demonstrates:
- A **realistic end-to-end medical segmentation pipeline**
- Efficient data engineering for large medical images
- Proper evaluation and error analysis
- Awareness of clinical and methodological limitations

The results are consistent with expectations in liver tumor segmentation
and provide a strong baseline for further improvements.

---

**Keywords**: Medical imaging, CT, segmentation, U-Net, PyTorch Lightning, Dice loss, liver tumor


