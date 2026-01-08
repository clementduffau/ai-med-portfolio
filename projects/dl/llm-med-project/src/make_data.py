from __future__ import annotations
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a medical assistant. Answer based on the provided context. "
    "If the context is insufficient, say you don't know."
)

def fmt_chat(task: str, user: str) -> str:
    return f"System: {SYSTEM}\nTask: {task}\nUser: {user}\nAssistant:"

# -------- PubMedQA (scorable: yes/no/maybe) --------
def map_pubmedqa(ex):
    ctx_list = ex.get("context", {}).get("contexts", [])
    context = " ".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list)
    question = ex.get("question", "")
    decision = (ex.get("final_decision", "") or "").strip().lower()

    user = f"Question: {question}\nContext: {context}\nAnswer with: Final: yes|no|maybe"
    return {
        "task": "qa_pubmedqa",
        "input": fmt_chat("medical_qa_yesno", user),
        "output": f"Final: {decision}"
    }

# -------- MedMCQA (scorable: A/B/C/D) --------
def map_medmcqa(ex):
    q = ex.get("question", "")
    A, B, C, D = ex.get("opa", ""), ex.get("opb", ""), ex.get("opc", ""), ex.get("opd", "")
    options = f"A. {A}\nB. {B}\nC. {C}\nD. {D}"
    cop = ex.get("cop", None)  # 0..3
    letter = ["A", "B", "C", "D"][int(cop)] if cop is not None else ""

    user = f"{q}\n\nOptions:\n{options}\nReply with: Final: <A|B|C|D>"
    return {
        "task": "qa_medmcqa",
        "input": fmt_chat("medical_mcq", user),
        "output": f"Final: {letter}"
    }

# -------- Summarization (Option A HF, no scripts) --------
def map_pubmed_summarization(ex):
    article = ex.get("article", "")
    abstract = ex.get("abstract", "")
    user = f"Summarize the following biomedical article into a concise abstract:\n\n{article}"
    return {
        "task": "summarization_pubmed",
        "input": fmt_chat("summarization", user),
        "output": (abstract or "").strip()
    }

def main():
    parts = []

    # PubMedQA
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    if "train" in pubmedqa:
        parts.append(pubmedqa["train"].map(map_pubmedqa, remove_columns=pubmedqa["train"].column_names))

    # MedMCQA
    medmcqa = load_dataset("openlifescienceai/medmcqa")
    if "train" in medmcqa:
        parts.append(medmcqa["train"].map(map_medmcqa, remove_columns=medmcqa["train"].column_names))

    # PubMed summarization (Option A)
    pubmed_sum = load_dataset("ccdv/pubmed-summarization")
    if "train" in pubmed_sum:
        # évite que le résumé domine tout le dataset
        summ_train = pubmed_sum["train"].shuffle(seed=42).select(range(min(50_000, len(pubmed_sum["train"]))))
        parts.append(summ_train.map(map_pubmed_summarization, remove_columns=summ_train.column_names))

    ds = concatenate_datasets(parts).shuffle(seed=42)

    # Splits
    ds = ds.train_test_split(test_size=0.02, seed=42)
    train = ds["train"]
    test = ds["test"]
    tmp = train.train_test_split(test_size=0.02, seed=42)
    train, val = tmp["train"], tmp["test"]

    train.to_json(OUT_DIR / "train.jsonl")
    val.to_json(OUT_DIR / "val.jsonl")
    test.to_json(OUT_DIR / "test.jsonl")

    print("Saved:", OUT_DIR / "train.jsonl", OUT_DIR / "val.jsonl", OUT_DIR / "test.jsonl")
    print("Sizes:", len(train), len(val), len(test))

if __name__ == "__main__":
    main()
