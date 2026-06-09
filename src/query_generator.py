"""
src/query_generator.py — Generate training queries from chunks using an LLM.

Why this exists:
    The "first sentence as query" trick produces low-quality training data when
    chunks start mid-sentence. Even with the new sentence-aware chunker, the
    first sentence is just a re-statement of the chunk topic — too easy for
    the model to memorize, too dissimilar from the natural-language questions
    users actually ask.

    LLM-generated queries solve both problems: each chunk gets 2-3 plausible
    questions written in natural language with different vocabulary than the
    chunk itself. This produces a much harder, more realistic training signal.

Three operating modes are provided:

    1. AnthropicQueryGenerator — uses the Claude API (best quality).
       Set ANTHROPIC_API_KEY in your environment.

    2. OpenAIQueryGenerator — uses the OpenAI API.
       Set OPENAI_API_KEY in your environment.

    3. ManualBatchGenerator — generates a single prompt you paste into
       Claude.ai / ChatGPT / Gemini web UI, then ingests the JSON response.
       Free, takes ~30 minutes for the whole corpus, useful when no API key
       is available.

Cost estimate (999 chunks, 2 queries each):
    Claude Haiku 3.5     ~ $0.50    (recommended)
    OpenAI gpt-4o-mini   ~ $0.40
    Manual (Claude.ai)   ~ free, batches of 20 chunks
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Prompt template — used by all three modes
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are creating training data for a semantic search engine over the textbook \
"Speech and Language Processing" by Jurafsky and Martin.

Below is a passage from the textbook. Generate exactly {n_queries} short, natural-language \
QUESTIONS that this passage would answer. The questions should:

  - Sound like real student questions (not copies of textbook sentences).
  - Use synonyms and rephrasing where possible (don't just lift words from the passage).
  - Cover different aspects of the passage (concept, mechanism, definition, example).
  - Be 8-20 words long, ending in a question mark.

Return ONLY valid JSON in this exact format (no prose, no markdown fences):
{{"queries": ["question 1?", "question 2?"{example_extra}]}}

PASSAGE:
{passage}
"""


def _build_prompt(passage: str, n_queries: int = 2) -> str:
    """Render the prompt template for a single passage."""
    example_extra = ', "question 3?"' if n_queries >= 3 else ""
    return PROMPT_TEMPLATE.format(
        n_queries=n_queries,
        example_extra=example_extra,
        passage=passage,
    )


def _parse_response(text: str) -> List[str]:
    """Extract a list of queries from a (potentially noisy) LLM response.

    Args:
        text: Raw response from the LLM.

    Returns:
        List of cleaned query strings.

    Raises:
        ValueError: If no valid JSON with a 'queries' key is found.
    """
    # Strip markdown code fences if the model wrapped its answer
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Find the first {...} block (handles surrounding chatter)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    data = json.loads(match.group(0))
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError(f"'queries' is not a list: {queries}")

    # Light cleanup
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()]


# ---------------------------------------------------------------------------
# Base class — defines the common contract
# ---------------------------------------------------------------------------

class _BaseQueryGenerator:
    """Abstract base — subclasses implement ``_call_llm``."""

    def __init__(self, n_queries: int = 2) -> None:
        self.n_queries = n_queries

    def _call_llm(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_for_chunk(self, chunk: Dict) -> List[str]:
        """Generate queries for a single chunk dict."""
        prompt = _build_prompt(chunk["content"], n_queries=self.n_queries)
        response = self._call_llm(prompt)
        return _parse_response(response)

    def generate_pairs(
        self,
        chunks: Sequence[Dict],
        output_path: str | Path,
        resume: bool = True,
        sleep_seconds: float = 0.0,
    ) -> List[Dict]:
        """Generate queries for every chunk and write a pairs JSON.

        Output schema:
            [{"query": "...", "positive_id": "chunk_0001", "positive_text": "..."}, ...]

        Args:
            chunks:        List of chunk dicts.
            output_path:   Where to write the resulting pairs JSON.
            resume:        If True and the output file exists, skip chunk_ids
                           already present (lets you re-run after a crash).
            sleep_seconds: Optional pause between API calls (rate limiting).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume support — read what's already done
        done_ids: set[str] = set()
        existing: List[Dict] = []
        if resume and output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            done_ids = {p["positive_id"] for p in existing}
            print(f"[resume] {len(done_ids)} chunks already processed in {output_path}")

        pairs = list(existing)
        for i, chunk in enumerate(chunks, 1):
            if chunk["id"] in done_ids:
                continue
            try:
                queries = self.generate_for_chunk(chunk)
            except Exception as e:
                print(f"  [error] chunk {chunk['id']}: {e}")
                continue

            for q in queries:
                pairs.append({
                    "query": q,
                    "positive_id": chunk["id"],
                    "positive_text": chunk["content"],
                })

            if i % 10 == 0 or i == len(chunks):
                # Persist incrementally so a crash doesn't lose work
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(pairs, f, indent=2, ensure_ascii=False)
                print(f"  [{i}/{len(chunks)}] {len(pairs)} pairs total")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        # Final write
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(pairs)} pairs from {len(chunks)} chunks -> {output_path}")
        return pairs


# ---------------------------------------------------------------------------
# Mode 1 — Anthropic Claude API
# ---------------------------------------------------------------------------

class AnthropicQueryGenerator(_BaseQueryGenerator):
    """Use the Claude API to generate queries.

    Install:    uv add anthropic
    Env var:    ANTHROPIC_API_KEY

    Args:
        model:      Claude model id. Haiku is cheapest and fast; Sonnet is
                    higher quality. Default: claude-haiku-4-5.
        n_queries:  Number of queries to generate per chunk.
        api_key:    Override the env var if needed.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        n_queries: int = 2,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(n_queries=n_queries)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Install anthropic: `uv add anthropic`") from e

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        # resp.content is a list of TextBlock objects
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


# ---------------------------------------------------------------------------
# Mode 2 — OpenAI API
# ---------------------------------------------------------------------------

class OpenAIQueryGenerator(_BaseQueryGenerator):
    """Use the OpenAI API to generate queries.

    Install:   uv add openai
    Env var:   OPENAI_API_KEY

    Args:
        model:     OpenAI model id. gpt-4o-mini is cheapest. Default.
        n_queries: Queries per chunk.
        api_key:   Override env var if needed.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_queries: int = 2,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(n_queries=n_queries)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai: `uv add openai`") from e

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Mode 3 — Manual / batch mode (no API key)
# ---------------------------------------------------------------------------

class ManualBatchGenerator:
    """No-API-key fallback: emit prompts to paste into a chat UI.

    Workflow:
        1. ``batch_prompts(chunks, batch_size=20)`` writes one prompt per file
           in ``out_dir/``. Each prompt asks for queries for 20 chunks at once.
        2. You paste each prompt into Claude.ai / ChatGPT / Gemini.
        3. Save each response as ``response_001.json`` etc. in ``out_dir/``.
        4. ``ingest_responses(out_dir, chunks)`` parses the responses and
           emits a pairs JSON.

    With ~50 batches of 20 chunks, this takes ~30 minutes of clicking.
    """

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


# ---------------------------------------------------------------------------
# Negative sampling — same logic as src/dataset.py, lifted so query
# generators can produce triplet-format output directly.
# ---------------------------------------------------------------------------

def add_random_negatives(
    pairs: List[Dict],
    chunks: Sequence[Dict],
    min_distance: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """Augment query/positive pairs with a randomly sampled negative chunk.

    Args:
        pairs:        List of {"query","positive_id","positive_text"} dicts.
        chunks:       The full corpus (used to look up negatives).
        min_distance: Negative must be at least this many positions away from positive.
        seed:         Random seed.

    Returns:
        New list with negative_id and negative_text added to each pair.
    """
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


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: print the prompt that would be sent for a single chunk
    sample_chunk = {
        "id": "chunk_0010",
        "content": (
            "A bigram language model assigns a probability to each word in a "
            "sequence based on the previous word. The model is estimated from "
            "corpus counts using maximum likelihood estimation. P(w_n | w_{n-1}) "
            "is the conditional probability that w_n follows w_{n-1}."
        ),
    }

    prompt = _build_prompt(sample_chunk["content"], n_queries=2)
    print("=" * 60)
    print("SAMPLE PROMPT (what gets sent to the LLM):")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    # Parse a fake response to verify the parser works
    fake_response = '''{"queries": ["How do bigram language models work?", "What method estimates bigram probabilities?"]}'''
    parsed = _parse_response(fake_response)
    print(f"\nParsed {len(parsed)} queries:")
    for q in parsed:
        print(f"  - {q}")
