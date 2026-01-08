from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset, DatasetDict, concatenate_datasets


@dataclass
class Label_spec:
    label_names : List[str]
    patterns : Dict[str, List[str]]
    

def default_label_spec():
    patterns = {
    "diabetes": [r"\bdiabetes\b", r"\bt1dm\b", r"\bt2dm\b", r"\bhba1c\b"],
    "hypertension": [r"\bhypertension\b", r"\bhtn\b", r"\bhigh blood pressure\b"],
    "asthma_copd": [r"\basthma\b", r"\bcopd\b", r"\bchronic obstructive\b"],
    "cad": [r"\bcoronary artery disease\b", r"\bcad\b", r"\bischemic heart\b", r"\bmyocardial infarction\b", r"\bmi\b"],
    "heart_failure": [r"\bheart failure\b", r"\bchf\b", r"\bhfr?ef\b"],
    "ckd": [r"\bchronic kidney disease\b", r"\bckd\b", r"\besrd\b", r"\bdialysis\b"],
    "depression_anxiety": [r"\bdepression\b", r"\banxiety\b", r"\bssri\b"],
    "obesity": [r"\bobesity\b", r"\bbmi\b"],
    "cancer": [r"\bcancer\b", r"\bmalignanc(y|ies)\b", r"\bcarcinoma\b", r"\btumou?r\b"],
    "stroke": [r"\bstroke\b", r"\bcva\b", r"\btia\b"],
    "infection": [r"\bsepsis\b", r"\bpneumonia\b", r"\buti\b", r"\binfection\b"],
    }
    return Label_spec(label_names = list(patterns.keys()), patterns = patterns)

def build_dataset(
    dataset_id : str,
    text_col : str = "note",
    seed : int = 42,
    neg_ratio : float = 1.0,
    test_size : float = 0.2,
):
    spec = default_label_spec()
    label_names = spec.label_names
    compiled = {k: [re.compile(p, re.I) for p in pats] for k, pats in spec.patterns.items()}
    raw = load_dataset(dataset_id)["train"]
    
    def add_labels(ex):
        text = ex.get(text_col) or ""
        y = [int(any(p.search(text) for p in compiled[lab])) for lab in label_names]
        ex["labels"] = y
        return ex
    
    ds = raw.map(add_labels, num_proc = 4)
    
    ds_pos = ds.filter(lambda ex : sum(ex["labels"]) > 0)
    ds_neg = ds.filter(lambda ex : sum(ex["labels"]) == 0)
    
    n_neg = int(len(ds_pos) * neg_ratio)
    ds_neg_sampled = ds_neg.shuffle(seed=seed).select(range(min(n_neg, len(ds_neg))))

    ds_balanced = concatenate_datasets([ds_pos, ds_neg_sampled]).shuffle(seed=seed)

    split = ds_balanced.train_test_split(test_size=test_size, seed=seed)
    tmp = split["test"].train_test_split(test_size=0.5, seed=seed)

    out = DatasetDict(train=split["train"], validation=tmp["train"], test=tmp["test"])
    return out, label_names

def add_note_id(example, idx):
    example["note_id"] = idx
    return example
    
