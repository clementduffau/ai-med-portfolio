# Results — Clinical Multi-Label Classification of Medical Notes

## Overview

This project addresses **multi-label classification of clinical notes** using a transformer-based language model adapted to the medical domain.  
The goal is to automatically assign multiple clinical conditions (phenotypes) to a single medical note, reflecting real-world clinical documentation where multiple diagnoses can co-occur.

The final evaluation is performed **at the note level**, not at the chunk level, to ensure clinically meaningful metrics.

---

## Model

- **Architecture**: ClinicalBERT
- **Checkpoint**: `emilyalsentzer/Bio_ClinicalBERT`
- **Task**: Multi-label classification
- **Loss**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Number of labels**: 11 clinical conditions

---

## Dataset & Preprocessing

- **Dataset**: Synthetic clinical notes (Asclepius)
- **Input**: Long unstructured clinical notes
- **Labels**: Multi-hot vectors representing clinical conditions
- **Challenge**: Notes often exceed BERT’s maximum context length

### Sliding Window Strategy

To handle long notes, a **sliding window** approach was used:

- `max_length = 256`
- `stride = 128`
- Each note is split into overlapping chunks
- Each chunk is independently processed by the model

To recover a **note-level prediction**, chunk-level predictions are aggregated using **max pooling**:
> If any chunk strongly indicates a condition, the whole note is considered positive for that label.

---

## Evaluation Protocol

### Note-Level Aggregation

For each note:
1. Compute probabilities for all chunks
2. Aggregate chunk probabilities with **max pooling**
3. Apply a threshold of `0.5` to obtain binary predictions

This approach prioritizes **recall** for clinically relevant conditions that may appear only once in a long note.

---

## Test Results (Note-Level)

| Metric        | Score |
|--------------|-------|
| **F1 Micro** | **0.9933** |
| **F1 Macro** | **0.9934** |
| **Subset Accuracy** | **0.9907** |

### Interpretation

- **F1 Micro (0.9933)**  
  Indicates excellent overall performance, dominated by frequent labels.

- **F1 Macro (0.9934)**  
  Shows that performance is consistently strong across all labels, including rarer conditions.

- **Subset Accuracy (0.9907)**  
  Nearly all notes have **all labels correctly predicted simultaneously**, which is a very strict metric in multi-label settings.

These results demonstrate that the model:
- Successfully captures clinically meaningful patterns
- Handles label imbalance effectively
- Scales well from chunk-level inference to note-level decisions

---

## Qualitative Evaluation

Beyond metrics, qualitative analysis was performed:

- Inspection of predicted labels per note
- Identification of false positives and false negatives
- Verification that rare conditions are detected when mentioned in a single section of a note

The max-pooling aggregation strategy significantly reduced false negatives compared to chunk-level evaluation alone.

---

## Key Technical Contributions

- End-to-end **multi-label clinical NLP pipeline**
- Robust **sliding window tokenization**
- Correct **note-level aggregation of predictions**
- Evaluation aligned with **real clinical use cases**
- Clean separation between data processing, modeling, and evaluation

---

## Limitations & Future Work

- The dataset is synthetic; performance on real-world clinical notes (e.g. MIMIC-III/IV) remains to be evaluated.
- Thresholds are fixed at 0.5; label-specific calibration could further improve clinical reliability.
- Future work could include:
  - Attention or attribution analysis
  - Calibration curves per pathology
  - Evaluation on temporally structured clinical notes

---

## Conclusion

This project demonstrates that **ClinicalBERT combined with a sliding window and note-level aggregation** is a highly effective approach for multi-label classification of long clinical notes.

The methodology is directly applicable to real-world clinical NLP tasks such as:
- Automated phenotyping
- Clinical decision support
- Medical coding assistance

The strong quantitative results, combined with a clinically grounded evaluation protocol, make this approach suitable for both research and applied healthcare settings.
