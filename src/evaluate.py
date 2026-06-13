from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore


def reciprocal_rank(ranked_ids: List[str], correct_id: str, k: int = 10) -> float:
    
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        if chunk_id == correct_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: List[str], correct_id: str, k: int) -> float:
    
    return 1.0 if correct_id in ranked_ids[:k] else 0.0


def compute_metrics(
    ranked_id_lists: List[List[str]],
    correct_ids: List[str],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    
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


def evaluate_biencoder(
    store: VectorStore,
    eval_df: pd.DataFrame,
    top_k: int = 10,
) -> Dict[str, float]:
   
    ranked_id_lists: List[List[str]] = []

    for _, row in eval_df.iterrows():
        results = store.search(row["query"], top_k=top_k)
        ranked_id_lists.append([r["id"] for r in results])

    return compute_metrics(ranked_id_lists, eval_df["expected_chunk_id"].tolist())


def _tokenize_bm25(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def evaluate_bm25(
    chunks: List[Dict],
    eval_df: pd.DataFrame,
    top_k: int = 10,
) -> Dict[str, float]:
    
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

def metrics_by_query_type(
    chunks: List[Dict],
    store,
    eval_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    
    if "query_type" not in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["query_type"] = "all"

    rows: List[Dict] = []
    for qtype, sub in eval_df.groupby("query_type"):
        bm25_m = evaluate_bm25(chunks, sub, top_k=top_k)
        bi_m   = evaluate_biencoder(store, sub, top_k=top_k)

        for metric_name, score in bm25_m.items():
            rows.append({"query_type": qtype, "system": "BM25",
                         "metric": metric_name, "score": score,
                         "n_queries": len(sub)})
        for metric_name, score in bi_m.items():
            rows.append({"query_type": qtype, "system": "BiEncoder",
                         "metric": metric_name, "score": score,
                         "n_queries": len(sub)})

    return pd.DataFrame(rows)


def comparison_table(
    bm25_metrics: Dict[str, float],
    biencoder_metrics: Dict[str, float],
) -> pd.DataFrame:

    all_keys = sorted(set(bm25_metrics) | set(biencoder_metrics))
    rows = []
    for key in all_keys:
        b = bm25_metrics.get(key, float("nan"))
        e = biencoder_metrics.get(key, float("nan"))
        delta = round(e - b, 4) if not (e != e or b != b) else float("nan")
        rows.append({"Metric": key, "BM25": b, "BiEncoder": e, "Delta (BiEnc - BM25)": delta})
    return pd.DataFrame(rows)


def qualitative_comparison(
    query: str,
    store: VectorStore,
    chunks: List[Dict],
    top_k: int = 3,
) -> None:

    bi_results = store.search(query, top_k=top_k)

    corpus = [c["content"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    tokenized_corpus = [_tokenize_bm25(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize_bm25(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

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
