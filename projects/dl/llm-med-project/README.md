# Results — Medical LLM Fine-Tuning for QA and Clinical Summarization

## Overview

This project focuses on the **fine-tuning of a Large Language Model (LLM) for medical applications** using **parameter-efficient fine-tuning (LoRA)**.  
The goal is to adapt a single generative model to perform multiple clinically relevant NLP tasks while remaining computationally efficient.

The final evaluation emphasizes **answer-level and document-level quality**, ensuring that generated outputs are meaningful in real-world medical settings.

---

## Model

- **Architecture**: LLaMA-style causal language model  
- **Checkpoint**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
- **Task type**: Instruction-following causal language modeling  
- **Fine-tuning method**: LoRA (Low-Rank Adaptation)

The base model is fully frozen, and only a small subset of trainable parameters is introduced via LoRA adapters.

---

## Datasets & Tasks

### Medical Question Answering

The model is trained on multiple medical QA datasets to improve robustness and coverage of medical knowledge:

- **PubMedQA**
- **MedMCQA**
- **MEDIQA**

Each example contains a medical question, optional supporting context, and a reference answer written in a clinical style.

### Clinical & Biomedical Summarization

To enable summarization capabilities, biomedical text datasets are included:

- **PubMed Summarization**
- **Scientific Papers (PubMed)**

These datasets contain long biomedical articles paired with concise abstracts, encouraging the model to learn information compression and relevance selection.

---

## Instruction-Tuning Format

All datasets are unified into a single **instruction-based format**:

System: You are a medical assistant.
Task: <medical_qa | summarization>
User: <question or document>
Assistant: <target answer or summary>


For training:
- The prompt and answer are concatenated
- Tokens corresponding to the prompt are masked with `-100`
- Loss is computed **only on the answer tokens**

This ensures the model learns how to generate medically relevant responses rather than memorizing prompts.

---

## Fine-Tuning Strategy

### LoRA (Low-Rank Adaptation)

Instead of full fine-tuning, **LoRA adapters** are injected into the attention mechanism of the model:

- Base model parameters are frozen
- Only low-rank update matrices are trained

**Targeted attention modules**:
- `q_proj`
- `v_proj`


## Evaluation Protocol

### Quantitative Evaluation

Since the model is generative, evaluation relies on the following metrics:

- **Validation loss**
- **Test loss**
- **Perplexity**

Loss is computed **only on answer tokens**, providing a faithful and task-aligned measure of generation quality.

---

### Qualitative Evaluation

In addition to numerical metrics, qualitative inspection is performed to assess real-world usability:

- Comparison between generated outputs and gold reference answers
- Assessment of medical correctness and clinical coherence
- Analysis of hallucinations and verbosity
- Evaluation of summary structure and informativeness

---

## Results

### Validation Behavior

- Rapid decrease in validation loss during early training
- Stable convergence without divergence
- No signs of overfitting

This behavior indicates that the model quickly learns task structure and that final performance is primarily limited by **LoRA capacity** rather than optimization issues.

---

### Qualitative Observations

- Medical QA responses are concise and context-aware
- The model appropriately avoids answering when context is insufficient
- Summaries capture key biomedical information with limited redundancy
- LoRA adapters successfully inject medical domain knowledge without degrading language fluency

---

## Key Technical Contributions

- Unified **medical instruction-tuning pipeline**
- Correct masking strategy for causal language modeling
- Custom dynamic padding and collation
- Efficient **LoRA-based fine-tuning**
- Evaluation aligned with real-world medical NLP usage

---

## Limitations & Future Work

- Automatic metrics (e.g. ROUGE, QA F1) are not yet integrated
- Medical factuality is currently assessed manually
- Future improvements include:
  - Integrating automatic evaluation metrics
  - Evaluation on real-world clinical datasets (e.g. MIMIC-style notes)

---

## Conclusion

This project demonstrates that **LoRA-based fine-tuning of LLaMA-style models** is an effective and resource-efficient approach for adapting LLMs to medical tasks such as **question answering** and **clinical summarization**.

The proposed methodology is **scalable**, **reproducible**, and directly applicable to both **research** and **applied medical NLP systems**.
