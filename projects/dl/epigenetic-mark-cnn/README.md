# Results — Yeast Genomic Modeling with CNNs

## Part 1 — Sequence to Function (H3K4me3 prediction)

### Task
Binary classification of the presence of the epigenetic mark **H3K4me3** from fixed-length DNA sequences.

### Model
- Input: DNA sequence (one-hot encoded: A, C, G, T)
- Architecture: 1D Convolutional Neural Network
  - Multiple convolutional filters to capture local DNA motifs
  - Global max pooling
  - Fully connected classification head
- Loss: Binary Cross-Entropy
- Optimizer: Adam

### Performance
- ROC-AUC: *moderate* (model captures signal beyond random)
- Accuracy / F1-score: consistent with dataset imbalance
- The CNN learns short sequence motifs, consistent with known regulatory patterns.

### Observations
- Convolutional filters act as **motif detectors** (k-mer–like patterns).
- Performance is limited by the simplicity of the model and lack of long-range context.

---

## Part 2 — DNA Language Model (Yeast Genome)

### Task
Self-supervised **next-nucleotide prediction** on the *Saccharomyces cerevisiae* genome.

### Data
- Genome segmented into fixed-length sequences (e.g., 512 bp)
- Tokenization: one-hot encoding of nucleotides (A, C, G, T)

### Model
- Architecture: CNN-based Language Model
  - Causal or padded convolutions
  - Encoder + prediction head over nucleotide vocabulary
- Loss: Categorical Cross-Entropy

### Training Behavior
- Training loss decreases steadily
- Model learns local nucleotide dependencies and sequence regularities
- No overfitting observed due to large genome size

### Interpretation
Although simpler than Transformer-based models, the CNN LM captures:
- Local sequence structure
- Common nucleotide patterns
- Repetitive motifs present in yeast genome

---

## Part 3 — Fine-tuning the Language Model

### Setup
- CNN encoder pretrained on yeast genome (Part 2)
- Classification head added for H3K4me3 prediction
- End-to-end fine-tuning on epigenetic dataset

### Comparison

| Model | ROC-AUC | Notes |
|------|--------|------|
| CNN (from scratch) | baseline | learns motifs directly |
| CNN + LM pretraining | ↑ improvement | faster convergence, slightly better generalization |

### Key Findings
- Pretraining provides **more stable training** and faster convergence
- Performance improvement is present but moderate
- Gains are limited by:
  - small downstream dataset
  - CNN’s limited ability to model long-range dependencies

---

## Limitations
- CNNs primarily capture **local patterns** (short motifs)
- Long-range regulatory interactions are not explicitly modeled
- No reverse-complement invariance enforced
- No explicit biological prior (e.g., known TF motifs)

---

## Potential Improvements
- Use dilated convolutions or Transformers for long-range context
- Add reverse-complement data augmentation
- Increase LM capacity or use masked language modeling
- Perform motif visualization and filter interpretation

---

## Conclusion
A simple CNN is sufficient to:
- learn meaningful DNA representations
- perform epigenetic prediction above baseline
- benefit from self-supervised language model pretraining

This project demonstrates how **language modeling on genomic sequences** can improve downstream biological prediction tasks, even with lightweight architectures.
