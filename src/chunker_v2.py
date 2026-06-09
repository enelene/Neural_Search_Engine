"""
src/chunker_v2.py — Sentence-aware chunker (replacement for pipeline.py logic).

Why this exists:
    The original ``DataPipeline.preprocess_and_chunk`` in pipeline.py slices the
    text every N words. That produces chunks that start and end MID-SENTENCE:

        chunk_0223: "templates. So a neural network is like multinomial logistic..."
                     ^^^^^^^^^^ — orphan word from the previous sentence

    This breaks two things:
      1. The "first sentence as query" trick yields half-sentence garbage.
      2. The chunk's last sentence is also truncated, so the chunk doesn't read
         like a self-contained paragraph.

How this fixes it:
    1. Split the full text into sentences using a robust regex.
    2. Greedily pack sentences into chunks until the word budget is reached.
    3. Overlap by carrying over the last N sentences (not N words) into the
       next chunk so context isn't lost.

Result:
    - Every chunk starts at a sentence boundary.
    - Every chunk ends at a sentence boundary.
    - The first sentence is a real, complete sentence — ready to use as a query
      (though we'll prefer LLM-generated queries; see src/query_generator.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


# A robust sentence splitter for English prose. Handles:
#   - Standard sentence ends:  .  !  ?
#   - Common abbreviations that LOOK like sentence ends but aren't
#     (Mr.  Dr.  Fig.  e.g.  i.e.  vs.  Eq.  Sec.  Ch.  etc.)
# We protect abbreviations by replacing their period with a placeholder, splitting,
# then putting the period back.
_ABBREVIATIONS = {
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "Fig", "Eq", "Sec", "Ch", "App", "Vol",
    "etc", "vs", "cf", "pp", "p", "no", "St",
    "i.e", "e.g", "et al",
}

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> List[str]:
    """Split *text* into sentences, protecting common abbreviations.

    Args:
        text: A long string of prose (cleaned).

    Returns:
        List of sentence strings, each ending with its terminal punctuation.
    """
    # Step 1: protect abbreviation periods by replacing them with a placeholder
    protected = text
    for abbr in _ABBREVIATIONS:
        # Word-boundary match of "Abbr." → "Abbr<DOT>"
        protected = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}<DOT>", protected)

    # Step 2: split on sentence-ending punctuation followed by capital letter
    raw_sentences = _SENT_END.split(protected)

    # Step 3: restore the abbreviation periods
    sentences = [s.replace("<DOT>", ".").strip() for s in raw_sentences]

    # Step 4: drop empties and trim
    return [s for s in sentences if s]


class SentenceAwareChunker:
    """Pack sentences into ~target_words chunks, preserving sentence boundaries.

    Args:
        target_words:    Soft target for chunk size in words. The chunker may
                         overshoot slightly because it never breaks a sentence.
        max_words:       Hard ceiling. If adding the next sentence exceeds this,
                         the current chunk closes.
        overlap_sentences: How many trailing sentences to carry into the next
                         chunk for context continuity.
        min_words:       Chunks shorter than this are dropped (usually trailing
                         partial chunks at the end of a document).
        min_letter_ratio: Minimum fraction of alpha characters to keep a chunk
                         (filters out tables / math-only chunks).
    """

    def __init__(
        self,
        target_words: int = 220,
        max_words: int = 300,
        overlap_sentences: int = 2,
        min_words: int = 80,
        min_letter_ratio: float = 0.60,
    ) -> None:
        self.target_words = target_words
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences
        self.min_words = min_words
        self.min_letter_ratio = min_letter_ratio

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    def _is_high_quality(self, chunk_text: str) -> bool:
        """Drop chunks dominated by numbers/symbols (tables, code blocks, etc.)."""
        if not chunk_text:
            return False
        letters = sum(c.isalpha() for c in chunk_text)
        return (letters / len(chunk_text)) >= self.min_letter_ratio

    def chunk(self, text: str) -> List[Dict]:
        """Split *text* into well-formed chunks.

        Args:
            text: Cleaned full document text.

        Returns:
            List of chunk dicts with keys: id, content, word_count, num_sentences.
        """
        sentences = split_into_sentences(text)

        chunks: List[Dict] = []
        buffer: List[str] = []
        buffer_words = 0

        def flush() -> None:
            """Materialize the current buffer as a chunk."""
            nonlocal buffer, buffer_words
            if buffer_words < self.min_words:
                buffer, buffer_words = [], 0
                return
            content = " ".join(buffer)
            if not self._is_high_quality(content):
                buffer, buffer_words = [], 0
                return
            chunks.append({
                "id": f"chunk_{len(chunks) + 1:04d}",
                "content": content,
                "word_count": self._word_count(content),
                "num_sentences": len(buffer),
            })
            # Carry the last K sentences forward as overlap
            keep = buffer[-self.overlap_sentences:] if self.overlap_sentences else []
            buffer = list(keep)
            buffer_words = sum(self._word_count(s) for s in buffer)

        for sent in sentences:
            sent_words = self._word_count(sent)

            # If adding this sentence would exceed the hard ceiling, flush first
            if buffer_words + sent_words > self.max_words and buffer:
                flush()

            buffer.append(sent)
            buffer_words += sent_words

            # If we've crossed the soft target, flush at the next sentence boundary
            if buffer_words >= self.target_words:
                flush()

        # Final flush
        if buffer_words >= self.min_words:
            flush()

        return chunks

    def chunk_file(self, input_path: str | Path, output_path: str | Path) -> List[Dict]:
        """Read a raw-text file, chunk it, write JSON output.

        Args:
            input_path:  Path to a .txt file (or .json with a "text" field).
            output_path: Where to write the chunk JSON.

        Returns:
            The list of chunk dicts.
        """
        input_path = Path(input_path)
        text = input_path.read_text(encoding="utf-8")
        # If the input is JSON wrapping the text, unwrap it
        if input_path.suffix == ".json":
            text = json.loads(text).get("text", text)
        chunks = self.chunk(text)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunks)} chunks -> {output_path}")
        return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Re-chunk the Jurafsky PDF using sentence-aware logic.

    Run from project root:
        uv run python -m src.chunker_v2
    """
    import argparse
    from src.utils import extract_clean_text_from_pdf

    parser = argparse.ArgumentParser(description="Sentence-aware re-chunker")
    parser.add_argument("--pdf", default="data/jurafsky_martin.pdf")
    parser.add_argument("--out", default="data/processed/jurafsky_chunks_v2.json")
    parser.add_argument("--target", type=int, default=220)
    parser.add_argument("--max", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=2)
    args = parser.parse_args()

    raw_text = extract_clean_text_from_pdf(args.pdf)

    chunker = SentenceAwareChunker(
        target_words=args.target,
        max_words=args.max,
        overlap_sentences=args.overlap,
    )
    chunks = chunker.chunk(raw_text)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Quick stats
    wcs = [c["word_count"] for c in chunks]
    sents = [c["num_sentences"] for c in chunks]
    print(f"\nWrote {len(chunks)} chunks -> {args.out}")
    print(f"Word counts   - min:{min(wcs)} max:{max(wcs)} avg:{sum(wcs)/len(wcs):.0f}")
    print(f"Sentence count - min:{min(sents)} max:{max(sents)} avg:{sum(sents)/len(sents):.1f}")

    # Show first sentence of first 5 chunks to verify boundaries
    print("\nFirst sentence of first 5 chunks (should look like real sentences):")
    for c in chunks[:5]:
        first_sent = split_into_sentences(c["content"])[0]
        # Strip non-ASCII for Windows console
        safe = first_sent.encode("ascii", "replace").decode()
        print(f"  [{c['id']}] {safe[:120]}")
