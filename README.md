# AI Engineer (Medical) — ML / DL

This repository is my **AI & Medical Machine Learning portfolio**, showcasing end-to-end projects in **machine learning, deep learning**, with a strong focus on **production-ready pipelines**, **medical relevance**, and **clear experimental reporting**.

Each project is **self-contained**, well-documented, and reproducible.

---

## Repository Structure
```
ai-med-portfolio/
├── projects/
│   ├── ml/
│   │   ├── diabetes-regression/
│   │   ├── heart-failure-classification/
│   │   └── breast-cancer-svm/
│   └── dl/
│       ├── imaging/
│       │   ├── medical-image-classification/
│       │   ├── medical-image-segmentation/
│       │   └── medical-image-diffusion/
│       ├── genomics/
│       │   └── epigenetic-mark-cnn/
│       ├── tabular/
│       │   └── llc-disease-mlp/
│       ├── clinical-nlp/
│       │   └── medical-notes-multilabel-classification/
│       └── llm/
│           └── llm-med-project/
└── README.md
```
## Project Organization (Standard)

Each project follows the **same structure** for clarity and reproducibility:
```
project-name/
├── README.md
│ └── Project overview: objective, medical context, dataset, methods, and results

├── src/
│ └── Source code: model definitions, training loops, evaluation, utilities
│
├── notebooks/
│ └── Experiments and analysis notebooks
│ (model testing, inference, or training when not implemented in src/)
│
├── results/
│ └── Outputs: metrics, plots, figures, qualitative predictions
│ (present only when relevant)
│
└── pyproject.toml
└── Project dependencies and configuration
```

### Notes
- Training code may live in **`src/` or `notebooks/`**, depending on project complexity
- Not all projects include a `results/` folder (only when outputs are meaningful)

---

## Projects Overview

### Machine Learning (`projects/ml/`)

**3 projects — focus on fundamentals, interpretability, and baselines**

- **Diabetes Regression**
  - Linear & regularized regression models
  - Feature scaling, evaluation with RMSE / R²

- **Heart Failure Classification**
  - Decision Trees, Random Forest, XGBoost
  - Feature importance & clinical interpretability

- **Breast Cancer SVM**
  - Support Vector Machines (linear & kernel)
  - Margin analysis and model comparison

---

### Deep Learning (`projects/dl/`)

**7 projects — structured data, genomics, medical imaging, NLP, LLM**

- **LLC Disease MLP**
  - Classical MLP for medical tabular classification
  - Regularization & optimization strategies

- **Epigenetic Mark CNN**
  - 1D CNN for genomic sequence modeling
  - Multiclass prediction on epigenetic signals

- **Medical Image Classification**
  - CNNs for medical imaging (X-ray / scans)
  - Data augmentation & transfer learning

- **Medical Image Segmentation**
  - U-Net-based architectures
  - Multi-label segmentation (Dice, IoU metrics)

- **Medical Image Diffusion**
  - Conditional DDPM with U-Net backbone for Chest X-ray generation
  - Class-conditional image synthesis using pathology embeddings
  - Training via noise prediction (MSE loss) and qualitative visual evaluation

- **Medical Notes Multi-Label Classification**
  - Transformer-based model (ClinicalBERT)
  - Multi-label disease coding from clinical notes

- **LLM Project (Medical QA & Summarization)**
  - Fine-tuning **TinyLLaMA** with LoRA
  - Medical question answering
  - Medical paragraph summarization
  - Quantitative (loss, perplexity) & qualitative evaluation

---

## Technical Stack

- **Languages**: Python
- **ML/DL**: PyTorch, PyTorch Lightning, scikit-learn
- **NLP / LLM**: Hugging Face Transformers, LoRA, PEFT
- **Medical Imaging**: segmentation-models-pytorch, diffusers
- **Genomics**: Custom 1D CNNs
- **Experiment Tracking**: Weights & Biases
- **Evaluation**: Dice, IoU, ROC-AUC, F1, RMSE, Perplexity, MSE

---

## Philosophy

- End-to-end pipelines (data → model → evaluation)
- Medical relevance & correctness
- Reproducibility and clean code
- Clear documentation for **technical and non-technical reviewers**
- Focus on **real-world deployment constraints**

---

## Contact

- **LinkedIn**: *https://www.linkedin.com/in/cl%C3%A9ment-duffau-7aa983238/*
- **Email**: *duffauclem@gmail.com*

Feel free to explore each project — feedback and discussions are welcome.
