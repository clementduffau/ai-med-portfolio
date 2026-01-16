# Multimodal Medical Image Captioning (ROCOv2 + BLIP)

This project implements a **multimodal image-to-text system** for **medical image captioning**.  
The goal is to generate clinically meaningful textual descriptions from medical images using a **Vision–Language Model (BLIP)** fine-tuned on the **ROCOv2** dataset.

---

## Project Overview

- **Task**: Medical image captioning (image → text)
- **Domain**: Radiology / Medical imaging
- **Model**: BLIP (Bootstrapping Language-Image Pretraining)
- **Dataset**: ROCOv2 (Radiology Objects in Context)
- **Frameworks**: PyTorch, PyTorch Lightning, Hugging Face Transformers & Datasets

---

## Motivation

Medical images are often accompanied by textual descriptions (reports, captions, annotations).  
Automatically generating these descriptions can help with:
- clinical documentation,
- dataset annotation,
- multimodal medical AI research.

This project focuses on **learning the alignment between medical images and textual descriptions**.

---

## Dataset: ROCOv2

**ROCOv2 (Radiology Objects in Context)** contains medical images paired with textual captions extracted from radiology-related scientific articles.

Each sample contains:
- `image`: medical image (PIL)
- `caption`: associated textual description
- `image_id`: image identifier
- `cui`: UMLS medical concepts (optional, not used in training)

The dataset is loaded via Hugging Face and stored locally using `save_to_disk`.

---

## Project Structure

```
multimodal-med-project/
││
├── src/
│ ├── datamodule.py # Lightning DataModule (image + text)
│ ├── model.py # LightningModule (BLIP training & evaluation)
│ ├── main.py # Training / evaluation entry point
│
├── notebooks/
│ └── project_analyse/ # Test of the model
│
├── README.md
└── requirements.txt / pyproject.toml
```
---

##  Data Pipeline

### DataModule (`DataModule_multi`)

- Loads ROCOv2 from disk
- Applies **on-the-fly preprocessing**:
  - image normalization & resizing
  - text tokenization
- Uses `AutoProcessor` (image + tokenizer)
- Pads sequences to fixed length
- Creates `labels` with `-100` on padding tokens (ignored by loss)

**Batch structure**:
- `pixel_values`: `[B, 3, H, W]`
- `input_ids`: `[B, L]`
- `attention_mask`: `[B, L]`
- `labels`: `[B, L]`

---

## Model

### BLIP – Image Captioning

- Model: `BlipForConditionalGeneration`
- Pretrained checkpoint:

BLIP consists of:
- a vision encoder (image understanding),
- a text decoder (caption generation).

The model is fine-tuned end-to-end on ROCOv2.

---

## Training

### Loss
- Cross-entropy loss on generated captions
- Padding tokens are ignored (`-100` labels)

### Optimizer & Scheduler
- Optimizer: `AdamW`
- Learning rate warmup with linear decay
- Hyperparameters are configurable via `cfg_model`

### Training Framework
- PyTorch Lightning
- Automatic logging and checkpointing

---

## Evaluation

### Quantitative Metrics

- **Validation Loss**
- **ROUGE-L**  
Measures the longest common subsequence between generated captions and ground truth captions.

ROUGE-L is well-suited for medical captions because it captures:
- correct ordering of medical concepts,
- missing or omitted clinical information.

### Qualitative Evaluation

During validation and testing:
- captions are generated using **beam search**
- example outputs (image + GT caption + predicted caption) are inspected manually

---

## Caption Generation Details

Generation uses **beam search**:

```python
num_beams = 3
max_new_tokens = 64
```
Beam search improves **global** sentence coherence compared to greedy decoding.

## Testing

During test time:
- token-level loss is aggregated
- ROUGE-L is computed on generated captions qualitative examples can be saved or visualized

## Limitations

- ROCO captions are often long and technical, making generation challenging
- ROUGE-L does not fully capture clinical correctness
- BLIP base may produce generic captions without sufficient fine-tuning

## Possible Improvements

- Increase maximum caption length
- Add decoding constraints (e.g. no-repeat n-grams)
- Use BLIP-2 or LLaVA-style models
- Incorporate medical concept supervision (CUIs)
- Add clinical evaluation metrics (e.g. label extraction)