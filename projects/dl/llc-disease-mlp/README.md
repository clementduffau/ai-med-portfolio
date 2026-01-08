# Results — Spectral Disease Classification (MLP)

## Dataset
- One row = one spectrum (single-cell)
- Features: `lambda_1 … lambda_999` + `cell_type` (B vs TNK)
- Target: `patient_state` (malade vs sain)
- Key constraint: multiple spectra per patient → **group split by `patient_name`** to avoid leakage.

## Protocol
- Preprocessing: StandardScaler on spectral features + OneHotEncoder on `cell_type`
- Split: GroupShuffleSplit (train/val/test by patient)
- Model: MLP (PyTorch)
- Hyperparameter search: Optuna on grouped CV (patients kept in a single fold)
- Evaluation: ROC-AUC, PR-AUC + threshold tuned on validation for high sensitivity.

## Test Performance (final model)
- Chosen threshold (target recall≈0.80 on validation): **0.35**

- Confusion matrix: TN=589, FP=541, FN=119, TP=413

## Interpretation
The model reaches **moderate discrimination** (AUC ~0.71).  
When prioritizing sensitivity (recall), performance comes with many false positives, which decreases precision.

## Limitations / Next steps
- High-dimensional spectra (999 features) may require stronger regularization or dimensionality reduction (e.g., PCA).
- Probability calibration (Brier score + calibration curve) could improve decision threshold stability.
- Consider TabNet comparison and/or patient-level aggregation (e.g., averaging spectra per patient) to reduce noise.
