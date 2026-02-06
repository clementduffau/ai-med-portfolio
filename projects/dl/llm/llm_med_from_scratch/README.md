# Medical Language Model from Scratch

This project implements a **medical-domain Large Language Model (LLM) trained from scratch**, using **PubMed biomedical literature** as training data.

The objective is to reproduce the **entire classical LLM training pipeline** — from raw medical text to a trained Transformer — without relying on pretrained language models.

---

## Project Overview

This repository covers the full **LLM-from-scratch pipeline**:

1. Large-scale medical data ingestion (PubMed)
2. Custom tokenizer training (Hugging Face BPE)
3. Offline tokenization and binary dataset creation
4. Implementation of a **GPT-style decoder-only Transformer from scratch**
5. Classical **Causal Language Model (CLM)** training

The project intentionally **stops after base model training**, before instruction tuning or chatbot alignment.

---

## Dataset

### Source
- **PubMed Baseline**
- Open-access biomedical abstracts
- Millions of peer-reviewed scientific articles

### Processing
- XML parsing (streaming, memory-safe)
- Extraction of:
  - Article titles
  - Abstract text
- Text normalization and filtering
- Sharding into manageable files


## Tokenizer

A **custom Byte-Level BPE tokenizer** is trained from scratch using the Hugging Face `tokenizers` library.

### Motivation
- Medical vocabulary (diseases, drugs, acronyms)
- No out-of-vocabulary tokens
- GPT-compatible tokenization

### Configuration
- Type: **Byte-Level BPE**
- Vocabulary size: **32,000**
- Special tokens:
  - `[PAD]`
  - `[UNK]`
  - `[BOS]`
  - `[EOS]`

## Tokenized Dataset (Binary Format)
Text data is tokenized offline and stored as binary files for high-performance training.

### Why binary files?
- No tokenization during training
- Fast sequential reads
- Efficient memory usage
- Scales to large datasets

### Dataset splits
- Training: train_0000.bin, train_0001.bin, ...
- Validation: val.bin
- Test: test.bin
- Each file contains a continuous stream of token IDs, with [EOS] tokens separating documents.

## Model Architecture
The model is a GPT-style decoder-only Transformer, fully implemented from scratch in PyTorch.

### Architecture overview
- Token embeddings
- Learned positional embeddings
- Stack of Transformer blocks:
    - LayerNorm
    - Causal Multi-Head Self-Attention
    - Feed-Forward Network (GELU)
    - Residual connections
- Final LayerNorm
- Language modeling head (weight tying)

### Example configuration
```
Number of layers:        12
Hidden dimension:        768
Attention heads:         12
Context length:          1024 tokens
Approx. parameters:     ~120M
```

## Training Pipeline

### Framework
- PyTorch
- PyTorch Lightning
- AdamW optimizer
- Linear warmup + learning rate decay

### Objective
The model is trained using Causal Language Modeling (CLM):
- Predict token(t+1) given tokens(0..t)

### Data loading strategy
- Memory-mapped .bin files
- Random sliding windows of fixed length
- No padding or attention masks required

## Evaluation

During training, the following metrics are monitored:
- Training loss
- Validation loss
- Perplexity (PPL)

Final evaluation is performed on a held-out test set, never seen during training.

## Current Status
- PubMed data ingestion and preprocessing
- Tokenizer training
- Binary dataset creation
- GPT model implemented from scratch
- Classical CLM training