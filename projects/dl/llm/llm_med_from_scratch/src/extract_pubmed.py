import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_pubmed_docs(xml_gz_path: Path) -> Iterator[str]:
    # Streaming parse: memory-safe
    with gzip.open(xml_gz_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("PubmedArticle"):
                title = elem.find(".//ArticleTitle")
                title_text = clean("".join(title.itertext())) if title is not None else ""

                abs_elems = elem.findall(".//AbstractText")
                parts = []
                for a in abs_elems:
                    t = clean("".join(a.itertext()))
                    if t:
                        parts.append(t)
                abstract_text = clean(" ".join(parts)) if parts else ""

                doc = clean(" ".join([t for t in (title_text, abstract_text) if t]))
                if doc:
                    yield doc

                # Free memory
                elem.clear()


def main(
    in_dir: str = "data/pubmed/raw",
    out_dir: str = "data/pubmed/shards",
    shard_size: int = 1_000_000,   # docs per shard
    min_chars: int = 200,          # filter very short docs
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.xml.gz"))
    assert files, f"No .xml.gz found in {in_dir}"

    shard_idx = 0
    doc_in_shard = 0
    total_docs = 0
    kept_docs = 0
    skipped_short = 0

    shard_path = out_dir / f"pubmed_corpus_{shard_idx:04d}.txt"
    out = shard_path.open("w", encoding="utf-8")

    def rotate_shard():
        nonlocal shard_idx, doc_in_shard, out, shard_path
        out.close()
        shard_idx += 1
        doc_in_shard = 0
        shard_path = out_dir / f"pubmed_corpus_{shard_idx:04d}.txt"
        out = shard_path.open("w", encoding="utf-8")

    try:
        for fp in files:
            for doc in iter_pubmed_docs(fp):
                total_docs += 1

                # Filter
                if len(doc) < min_chars:
                    skipped_short += 1
                    continue

                out.write(doc + "\n")
                kept_docs += 1
                doc_in_shard += 1

                if doc_in_shard >= shard_size:
                    rotate_shard()

    finally:
        out.close()

    print(f"[OK] Parsed docs: {total_docs}")
    print(f"[OK] Kept docs:   {kept_docs}")
    print(f"[OK] Skipped (<{min_chars} chars): {skipped_short}")
    print(f"[OK] Shards written to: {out_dir}")
    print(f"[OK] Last shard index: {shard_idx:04d}")


if __name__ == "__main__":
    main()
