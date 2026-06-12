from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
ANCHORS: Dict[str, str] = {
    "chunk_0223": "intermediate layers having many possible activation functions",
    "chunk_0046": "Let's see a general equation for this n-gram approximation",
    "chunk_0554": "make one slight change to turn this language model with autoregressive generation",
    "chunk_0492": "Greedy search chooses yes followed by yes",
    "chunk_0451": "the dot product between CLS token outputs for the query and document",
    "chunk_0281": "Temperature sampling can help with this situation",
    "chunk_0206": "Mikolov et al. (2011) showed that recurrent neural nets could be used as language models",
    "chunk_0175": "The dot product acts as a similarity metric",
    "chunk_0561": "Attention thus replaces the static context vector",
    "chunk_0564": "13.9 Summary",
    "chunk_0080": "Shannon-McMillan-Breiman theorem",
    "chunk_0150": "if logically negative words",
    "chunk_0300": "Gehman et al. (2020) show that even completely non-toxic prompts",
    "chunk_0400": "Instruction tuning is a form of supervised learning where the training data consists of instructions",
    "chunk_0625": "useful for general transcription, for example for automatically generating captions",
}


def _normalize(text: str) -> str:
    
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")  # right/left single quote
    text = text.replace("“", '"').replace("”", '"')  # right/left double quote
    text = text.replace("–", "-").replace("—", "-")  
    text = text.replace("?", "")
    text = re.sub(r"\s+", " ", text)
    return text


def find_chunk_by_anchor(chunks: list, anchor: str) -> Optional[str]:
    needle = _normalize(anchor)
    candidates = []
    for c in chunks:
        haystack = _normalize(c["content"])
        if needle in haystack:
            candidates.append(c["id"])

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  [warn] anchor matched {len(candidates)} chunks; picking first: {candidates[0]}")
        return candidates[0]
    return None


def remap_eval_file(
    chunks_path: str | Path = "data/processed/jurafsky_chunks.json",
    eval_path: str | Path = "data/evaluation_set.csv",
    out_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Rewrite the eval CSV with current-canonical chunk_ids.

    Args:
        chunks_path: Path to the canonical chunks JSON.
        eval_path:   Path to the eval CSV to read.
        out_path:    Where to write the fixed CSV. Defaults to overwriting *eval_path*.

    Returns:
        The fixed DataFrame.
    """
    out_path = Path(out_path or eval_path)

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    df = pd.read_csv(eval_path)
    print(f"Loaded {len(chunks)} chunks and {len(df)} eval queries.")

    print("\nResolving anchors:")
    v1_to_new: Dict[str, str] = {}
    for v1_id, anchor in ANCHORS.items():
        new_id = find_chunk_by_anchor(chunks, anchor)
        v1_to_new[v1_id] = new_id
        safe_anchor = anchor.encode("ascii", "replace").decode()[:60]
        status = new_id if new_id else "NOT FOUND"
        print(f"  {v1_id} -> {status}    (anchor: '{safe_anchor}...')")

    new_ids = []
    for _, row in df.iterrows():
        old_id = row["expected_chunk_id"]
        new_id = v1_to_new.get(old_id)
        if new_id is None:
            print(f"  [warn] no anchor mapping for {old_id} — keeping as-is")
            new_id = old_id
        new_ids.append(new_id)

    df["expected_chunk_id"] = new_ids
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} remapped queries -> {out_path}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remap eval set by content anchors")
    parser.add_argument("--chunks", default="data/processed/jurafsky_chunks.json")
    parser.add_argument("--eval",   default="data/evaluation_set.csv")
    parser.add_argument("--out",    default=None,
                        help="Output path; defaults to overwriting --eval")
    args = parser.parse_args()

    remap_eval_file(args.chunks, args.eval, args.out)
