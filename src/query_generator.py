from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence


class ManualBatchGenerator:
    

    BATCH_PROMPT_TEMPLATE = """You are creating training data for a semantic search engine.

Below are {n} passages from the NLP textbook "Speech and Language Processing".
For EACH passage, generate {n_queries} short natural-language QUESTIONS the passage \
would answer. Questions should be 8-20 words, end with "?", and avoid copying textbook \
phrasing. Use synonyms.

Return ONE valid JSON object in this exact shape (no prose, no markdown):
{{
  "results": [
    {{"id": "chunk_0001", "queries": ["...?", "...?"]}},
    {{"id": "chunk_0002", "queries": ["...?", "...?"]}}
  ]
}}

PASSAGES:
{passages}
"""

    def __init__(self, n_queries: int = 2) -> None:
        self.n_queries = n_queries

    def batch_prompts(
        self,
        chunks: Sequence[Dict],
        out_dir: str | Path,
        batch_size: int = 20,
    ) -> int:
        """Write one prompt file per batch.

        Returns the number of batches created.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_batches = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            passages = "\n\n".join(
                f"--- {c['id']} ---\n{c['content']}" for c in batch
            )
            prompt = self.BATCH_PROMPT_TEMPLATE.format(
                n=len(batch),
                n_queries=self.n_queries,
                passages=passages,
            )
            n_batches += 1
            out_path = out_dir / f"prompt_{n_batches:03d}.txt"
            out_path.write_text(prompt, encoding="utf-8")

        print(f"Wrote {n_batches} prompt files to {out_dir}/")
        print("Paste each prompt into a chat UI. Save responses as response_NNN.json.")
        return n_batches

    def ingest_responses(
        self,
        responses_dir: str | Path,
        chunks: Sequence[Dict],
        output_path: str | Path,
    ) -> List[Dict]:
        """Parse response files and build a pairs JSON.

        Each response file should contain valid JSON of the form:
            {"results": [{"id": "chunk_0001", "queries": ["...?", "...?"]}, ...]}
        """
        responses_dir = Path(responses_dir)
        id_to_text = {c["id"]: c["content"] for c in chunks}
        pairs: List[Dict] = []

        for resp_path in sorted(responses_dir.glob("response_*.json")):
            try:
                data = json.loads(resp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"  [skip] {resp_path.name}: bad JSON ({e})")
                continue

            for item in data.get("results", []):
                cid = item.get("id")
                queries = item.get("queries", [])
                if cid not in id_to_text:
                    print(f"  [skip] {cid} not in corpus")
                    continue
                for q in queries:
                    pairs.append({
                        "query": q,
                        "positive_id": cid,
                        "positive_text": id_to_text[cid],
                    })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(pairs)} pairs -> {output_path}")
        return pairs

def add_random_negatives(
    pairs: List[Dict],
    chunks: Sequence[Dict],
    min_distance: int = 20,
    seed: int = 42,
) -> List[Dict]:
    
    import random
    rng = random.Random(seed)
    id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

    enriched: List[Dict] = []
    for p in pairs:
        pos_idx = id_to_idx.get(p["positive_id"])
        if pos_idx is None:
            continue
        neg_pool = [j for j in range(len(chunks)) if abs(j - pos_idx) > min_distance]
        if not neg_pool:
            continue
        neg = chunks[rng.choice(neg_pool)]
        enriched.append({
            **p,
            "negative_id": neg["id"],
            "negative_text": neg["content"],
        })
    return enriched

