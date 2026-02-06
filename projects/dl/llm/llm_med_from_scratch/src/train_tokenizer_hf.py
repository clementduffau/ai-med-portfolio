from pathlib import Path
from typing import Iterator, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

def iter_lines_from_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
  
def iter_lines_from_shards(shards_dir: Path, limit_lines: Optional[int] = None) -> Iterator[str]:
    shard_files = sorted(shards_dir.glob("*.txt"))
    assert shard_files, f"No .txt shards found in {shards_dir}"

    n = 0
    for fp in shard_files:
        # tqdm on files only (cheap). You can also remove it.
        for line in iter_lines_from_file(fp):
            yield line
            n += 1
            if limit_lines is not None and n >= limit_lines:
                return
            
                          
def main(
    input_path : str = "data/pubmed/shards",
    out_dir: str = "tokenizer/hf_bpe_med_32k",
    vocab_size : int = 32000,
    min_frequency : int = 2,
    limit_lines : int | None = None,
):
    input_path = Path(input_path)
    assert input_path.exists(), f"Missing {input_path}"
    assert input_path.is_dir(), f"input_path must be a directory of shards, got: {input_path}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True,exist_ok = True)
    
    
    tokenizer = Tokenizer(BPE(unk_token= "[UNK]"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space = True)
    tokenizer.decode = ByteLevelDecoder()
    
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    
    trainer = BpeTrainer(
        vocab_size = vocab_size,
        min_frequency = min_frequency,
        special_tokens = special_tokens,
        show_progress = True,
    )
    
    lines = iter_lines_from_shards(input_path, limit_lines=limit_lines)
    tokenizer.train_from_iterator(lines, trainer = trainer)
    
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    
    tokenizer.save(str(out_dir / "tokenizer.json"))
    print(f"[OK] Saved tokenizer to {out_dir/'tokenizer.json'}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
        
    
    