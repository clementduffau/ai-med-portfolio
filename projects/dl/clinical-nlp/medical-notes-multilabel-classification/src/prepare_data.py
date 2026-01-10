from make_dataset import *

from pathlib import Path



out_dir = Path("data")
def main():
    ds, label_names = build_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes",neg_ratio = 1.0)
    out_dir.mkdir(parents = True, exist_ok=True)
    ds = ds.map(add_note_id, with_indices=True)
    ds.save_to_disk(str(out_dir))
    (out_dir / "label_names.txt").write_text("\n".join(label_names), encoding="utf-8")
    print("Saved to:", out_dir, "num_labels=", len(label_names))
    
    
if __name__ == "__main__":
    main()
