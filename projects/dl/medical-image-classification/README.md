# Results — Chest X-Ray Pneumonia (DL Image Classification)

## Dataset
- Task: Binary classification (NORMAL vs PNEUMONIA)
- Source: Kaggle “Chest X-Ray Images (Pneumonia)”
- Classes:
  - 0: NORMAL
  - 1: PNEUMONIA
- Class imbalance handled with `BCEWithLogitsLoss(pos_weight=neg/pos)`.

---

## Model
- Backbone: EfficientNet-B7 (pretrained)
- Head: Dropout(p=0.2) + Linear → 1 logit
- Input: 224×224, grayscale images converted to 3 channels
- Data augmentation (train):
  - Resize(224×224)
  - RandomHorizontalFlip
  - RandomRotation(±10°)
  - Normalize (ImageNet mean/std)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Training:
  - Early stopping on validation loss
  - Best model checkpoint selected on validation loss

---

## Training dynamics
- Training and validation losses decrease smoothly.
- A small but stable generalization gap is observed, indicating limited overfitting.
- Validation loss plateaus after ~15–20 epochs, justifying early stopping.


---

## Metrics (TEST)

### ROC & PR performance
- ROC-AUC: **≈ 0.96**
- PR-AUC (Average Precision): **≈ 0.93**

These results indicate strong discriminative performance despite class imbalance.


---

## Threshold-based evaluation (TEST)

### Threshold = 0.50
Confusion matrix:

|              | Pred NORMAL | Pred PNEUMONIA |
|--------------|-------------|----------------|
| **True NORMAL**     | 187         | 47             |
| **True PNEUMONIA**  | 18          | 372            |

Derived metrics:
- Precision: **0.89**
- Recall (Sensitivity): **0.95**
- F1-score: **≈ 0.92**
- False negatives remain limited (18 cases), which is critical in a medical screening context.

---

### Clinical interpretation
- The default threshold already achieves high recall (>95%), which is desirable for pneumonia screening.
- A lower threshold could further reduce false negatives at the cost of increased false positives, depending on clinical requirements.

---

## Calibration
- Brier score (TEST): **0.0786**
- The calibration curve shows reasonable alignment with the diagonal, indicating moderately well-calibrated probabilities.
- Some miscalibration is observed in the mid-probability range, which could be improved with post-hoc calibration.


---

## Error analysis (qualitative)

### False Negatives (FN)
- Typically correspond to subtle or low-contrast opacities.
- Some cases exhibit partial lung visibility or atypical patterns.

### False Positives (FP)
- Often associated with image artifacts, medical devices, or strong textures near lung borders.
- Some NORMAL cases show ambiguous patterns resembling mild infiltrates.


---

## Conclusions
- The model achieves strong performance on pneumonia detection from chest X-rays:
  - High ROC-AUC and PR-AUC
  - High sensitivity at the default threshold
- The approach is suitable for screening-oriented use cases where false negatives must be minimized.
- Limitations:
  - Single dataset evaluation
  - Potential dataset bias (acquisition conditions, population)
- Future work:
  - External validation on a separate dataset
  - Explicit threshold optimization based on clinical constraints
  - Probability calibration (e.g., temperature scaling)
  - Comparison with alternative backbones (e.g., DenseNet121)

---
