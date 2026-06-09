"""
src/evaluate.py — Retrieval evaluation metrics and model comparison.

This module answers: "How good is our model compared to BM25?"

Metrics implemented:
────────────────────────────────────────────────────────────────────────────
MRR@K  (Mean Reciprocal Rank)
    The gold-standard metric for ranked retrieval evaluation.

    For each query we ask: "At what rank does the correct chunk appear?"
    If the correct chunk is at rank r, we get a score of 1/r.
    MRR = average of (1/r) across all queries.

    Examples:
        Correct chunk at rank 1 → score = 1.0   (perfect)
        Correct chunk at rank 2 → score = 0.5
        Correct chunk at rank 5 → score = 0.2
        Correct chunk not in top-K → score = 0.0

    MRR@10 means we only look at the top-10 results (K=10).

Recall@K
    "In what fraction of queries was the correct chunk found in the top K?"

    Recall@1 = accuracy (is the #1 result correct?)
    Recall@5 = was the correct answer anywhere in the top 5?
    Recall@10 = was the correct answer anywhere in the top 10?

    For a search engine this is more user-friendly:
        Recall@5 ≈ "will the user see the answer on the first page?"

Why these metrics?
    • They're standard in information retrieval (BEIR benchmark, MS MARCO, etc.)
    • They work naturally with our setup: one correct chunk per query
    • They capture both precision (MRR) and coverage (Recall)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked_ids: List[str], correct_id: str, k: int = 10) -> float:
    """Compute 1/rank for the first occurrence of *correct_id* in *ranked_ids*.

    Args:
        ranked_ids: List of chunk IDs sorted by descending relevance score.
        correct_id: The ground-truth chunk ID.
        k:          Only consider the top-k results.

    Returns:
        1/rank if found within top-k, else 0.0.
    """
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        if chunk_id == correct_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: List[str], correct_id: str, k: int) -> float:
    """Return 1.0 if *correct_id* appears in the top-k results, else 0.0.

    Args:
        ranked_ids: List of chunk IDs sorted by descending relevance.
        correct_id: The ground-truth chunk ID.
        k:          Cutoff rank.
    """
    return 1.0 if correct_id in ranked_ids[:k] else 0.0


def compute_metrics(
    ranked_id_lists: List[List[str]],
    correct_ids: List[str],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute MRR@10 and Recall@K for a set of queries.

    Args:
        ranked_id_lists: One ranked list of chunk IDs per query.
        correct_ids:     One ground-truth chunk ID per query.
        k_values:        Recall cutoffs to compute.

    Returns:
        Dict of metric name → mean score over all queries.
        Example: {"MRR@10": 0.65, "R@1": 0.4, "R@5": 0.7, "R@10": 0.8}
    """
    assert len(ranked_id_lists) == len(correct_ids), "Lists must be same length"
    n = len(correct_ids)

    mrr_scores = [
        reciprocal_rank(ids, cid, k=10)
        for ids, cid in zip(ranked_id_lists, correct_ids)
    ]

    results: Dict[str, float] = {
        "MRR@10": round(sum(mrr_scores) / n, 4),
    }

    for k in k_values:
        recall_scores = [
            recall_at_k(ids, cid, k)
            for ids, cid in zip(ranked_id_lists, correct_ids)
        ]
        results[f"R@{k}"] = round(sum(recall_scores) / n, 4)

    return results


# ---------------------------------------------------------------------------
# BiEncoder evaluation
# ---------------------------------------------------------------------------

def evaluate_biencoder(
    store: VectorStore,
    eval_df: pd.DataFrame,
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluate a VectorStore on the hand-crafted evaluation set.

    Args:
        store:   A built VectorStore (build() must have been called).
        eval_df: DataFrame with columns: query, expected_chunk_id.
        top_k:   Number of results to retrieve per query.

    Returns:
        Metrics dict (MRR@10, R@1, R@5, R@10).
    """
    ranked_id_lists: List[List[str]] = []

    for _, row in eval_df.iterrows():
        results = store.search(row["query"], top_k=top_k)
        ranked_id_lists.append([r["id"] for r in results])

    return compute_metrics(ranked_id_lists, eval_df["expected_chunk_id"].tolist())


# ---------------------------------------------------------------------------
# BM25 evaluation
# ---------------------------------------------------------------------------

def _tokenize_bm25(text: str) -> List[str]:
    """Lowercase + strip punctuation for BM25 tokenization."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def evaluate_bm25(
    chunks: List[Dict],
    eval_df: pd.DataFrame,
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluate BM25 retrieval on the same evaluation set.

    Args:
        chunks:  All corpus chunks (id + content).
        eval_df: DataFrame with columns: query, expected_chunk_id.
        top_k:   Number of results to retrieve per query.

    Returns:
        Metrics dict (MRR@10, R@1, R@5, R@10).
    """
    corpus = [c["content"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]

    tokenized_corpus = [_tokenize_bm25(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    ranked_id_lists: List[List[str]] = []
    for _, row in eval_df.iterrows():
        tokenized_query = _tokenize_bm25(row["query"])
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        ranked_id_lists.append([chunk_ids[i] for i in top_indices])

    return compute_metrics(ranked_id_lists, eval_df["expected_chunk_id"].tolist())


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison_table(
    bm25_metrics: Dict[str, float],
    biencoder_metrics: Dict[str, float],
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame.

    Args:
        bm25_metrics:      Metrics from evaluate_bm25().
        biencoder_metrics: Metrics from evaluate_biencoder().

    Returns:
        DataFrame with one row per metric and columns for BM25 / BiEncoder
        and a Δ (delta) column.
    """
    all_keys = sorted(set(bm25_metrics) | set(biencoder_metrics))
    rows = []
    for key in all_keys:
        b = bm25_metrics.get(key, float("nan"))
        e = biencoder_metrics.get(key, float("nan"))
        delta = round(e - b, 4) if not (e != e or b != b) else float("nan")
        rows.append({"Metric": key, "BM25": b, "BiEncoder": e, "Δ (BiEnc − BM25)": delta})
    return pd.DataFrame(rows)


def qualitative_comparison(
    query: str,
    store: VectorStore,
    chunks: List[Dict],
    top_k: int = 3,
) -> None:
    """Print side-by-side results from BM25 and BiEncoder for a single query.

    Useful for the demo notebook and the report.

    Args:
        query:   Raw query string.
        store:   Built VectorStore.
        chunks:  Full list of corpus chunks.
        top_k:   Number of results to show per system.
    """
    # BiEncoder results
    bi_results = store.search(query, top_k=top_k)

    # BM25 results
    corpus = [c["content"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    tokenized_corpus = [_tokenize_bm25(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize_bm25(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Print
    print(f"Query: {query}\n")
    print(f"{'BM25':^70}  {'BiEncoder':^70}")
    print("-" * 145)

    for rank in range(top_k):
        bm_idx = top_indices[rank]
        bm_id = chunk_ids[bm_idx]
        bm_score = round(scores[bm_idx], 3)
        bm_text = corpus[bm_idx][:120]

        bi = bi_results[rank]
        bi_id = bi["id"]
        bi_score = bi["score"]
        bi_text = bi["content"][:120]

        print(f"[{rank+1}] {bm_id} (score={bm_score})")
        print(f"    {bm_text}...")
        print()
        print(f"    {' '*40}[{rank+1}] {bi_id} (score={bi_score})")
        print(f"    {' '*40}{bi_text}...")
        print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from src.model import BiEncoder

    root = Path(__file__).resolve().parent.parent

    with open(root / "data" / "processed" / "jurafsky_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    eval_df = pd.read_csv(root / "data" / "evaluation_set.csv")
    print(f"Eval set: {len(eval_df)} queries")

    # BM25 baseline
    bm25_metrics = evaluate_bm25(chunks, eval_df)
    print("\nBM25 metrics:")
    for k, v in bm25_metrics.items():
        print(f"  {k}: {v}")

    # BiEncoder (untrained — just to check pipeline runs)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BiEncoder()
    store = VectorStore(model, tokenizer, batch_size=64)
    store.build(chunks)

    bi_metrics = evaluate_biencoder(store, eval_df)
    print("\nBiEncoder (pre-trained, no fine-tuning) metrics:")
    for k, v in bi_metrics.items():
        print(f"  {k}: {v}")

    print("\nComparison table:")
    print(comparison_table(bm25_metrics, bi_metrics).to_string(index=False))
