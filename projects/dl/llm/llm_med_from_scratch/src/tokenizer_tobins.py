from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def iter_text_lines(shards_dir: Path) -> Iterable[str]:
    shard_files = sorted(shards_dir.glob("*.txt"))
    assert shard_files, f"No .txt shards found in {shards_dir}"

    for fp in shard_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def write_tokens(buffer_ids: list[int], fp, dtype: np.dtype) -> int:
    if not buffer_ids:
        return 0
    arr = np.asarray(buffer_ids, dtype=dtype)
    arr.tofile(fp)
    return int(arr.size)


def main(
    shards_dir: str = "data/pubmed/shards",
    tokenizer_path: str = "tokenizer/hf_bpe_med_32k/tokenizer.json",
    out_dir: str = "data/tokens",
    shard_tokens: int = 50_000_000,  # tokens per train shard
    val_ratio: float = 0.01,         # 1% lines
    test_ratio: float = 0.001,       # 0.1% lines (adjust as you like)
    seed: int = 42,
    max_lines: Optional[int] = None, # quick test
):
    assert 0 <= val_ratio < 1
    assert 0 <= test_ratio < 1
    assert (val_ratio + test_ratio) < 1, "val_ratio + test_ratio must be < 1"

    random.seed(seed)

    shards_dir = Path(shards_dir)
    assert shards_dir.exists() and shards_dir.is_dir(), f"Missing shards_dir: {shards_dir}"

    tok_path = Path(tokenizer_path)
    assert tok_path.exists(), f"Missing tokenizer.json: {tok_path}"
    tokenizer = Tokenizer.from_file(str(tok_path))

    eos_id = tokenizer.token_to_id("[EOS]")
    bos_id = tokenizer.token_to_id("[BOS]")
    assert eos_id is not None, "Tokenizer missing [EOS]"
    assert bos_id is not None, "Tokenizer missing [BOS]"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    val_path = out_dir / "val.bin"
    test_path = out_dir / "test.bin"

    # Train shard state
    train_shard_idx = 0
    train_tok_count_in_shard = 0

    # Counters
    total_lines = train_lines = val_lines = test_lines = 0
    train_total_tokens = val_total_tokens = test_total_tokens = 0

    # Files
    train_fp = (out_dir / f"train_{train_shard_idx:04d}.bin").open("wb")
    val_fp = val_path.open("wb")
    test_fp = test_path.open("wb")

    train_buf: list[int] = []
    val_buf: list[int] = []
    test_buf: list[int] = []

    flush_every = 1_000_000  # tokens

    def rotate_train_shard():
        nonlocal train_shard_idx, train_tok_count_in_shard, train_fp
        train_fp.close()
        train_shard_idx += 1
        train_tok_count_in_shard = 0
        train_fp = (out_dir / f"train_{train_shard_idx:04d}.bin").open("wb")

    try:
        for line in tqdm(iter_text_lines(shards_dir), desc="Tokenizing lines"):
            total_lines += 1
            if max_lines is not None and total_lines > max_lines:
                break

            enc = tokenizer.encode(line)
            ids = enc.ids

            # Ensure EOS boundary
            if not ids or ids[-1] != eos_id:
                ids = ids + [eos_id]

            r = random.random()

            # Split decision
            if r < test_ratio:
                # TEST
                test_lines += 1
                test_buf.extend(ids)
                test_total_tokens += len(ids)

                if len(test_buf) >= flush_every:
                    write_tokens(test_buf, test_fp, dtype)
                    test_buf.clear()

            elif r < (test_ratio + val_ratio):
                # VAL
                val_lines += 1
                val_buf.extend(ids)
                val_total_tokens += len(ids)

                if len(val_buf) >= flush_every:
                    write_tokens(val_buf, val_fp, dtype)
                    val_buf.clear()

            else:
                # TRAIN
                train_lines += 1
                train_buf.extend(ids)
                train_total_tokens += len(ids)
                train_tok_count_in_shard += len(ids)

                if train_tok_count_in_shard >= shard_tokens:
                    write_tokens(train_buf, train_fp, dtype)
                    train_buf.clear()
                    rotate_train_shard()

                if len(train_buf) >= flush_every:
                    write_tokens(train_buf, train_fp, dtype)
                    train_buf.clear()

        # Final flush
        write_tokens(train_buf, train_fp, dtype); train_buf.clear()
        write_tokens(val_buf, val_fp, dtype); val_buf.clear()
        write_tokens(test_buf, test_fp, dtype); test_buf.clear()

    finally:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    print("\n[OK] Done.")
    print(f"[OK] Vocab size: {vocab_size} -> dtype: {dtype}")
    print(f"[OK] Ratios: val={val_ratio}, test={test_ratio}")
    print(f"[OK] Lines: total={total_lines:,} train={train_lines:,} val={val_lines:,} test={test_lines:,}")
    print(f"[OK] Tokens: train={train_total_tokens:,} val={val_total_tokens:,} test={test_total_tokens:,}")
    print(f"[OK] Train shards: 0..{train_shard_idx:04d} in {out_dir}")
    print(f"[OK] Val file:  {val_path}")
    print(f"[OK] Test file: {test_path}")


if __name__ == "__main__":
    main()
