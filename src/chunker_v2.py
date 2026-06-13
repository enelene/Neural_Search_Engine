from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

_ABBREVIATIONS = {
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "Fig", "Eq", "Sec", "Ch", "App", "Vol",
    "etc", "vs", "cf", "pp", "p", "no", "St",
    "i.e", "e.g", "et al",
}

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> List[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}<DOT>", protected)

    raw_sentences = _SENT_END.split(protected)

    sentences = [s.replace("<DOT>", ".").strip() for s in raw_sentences]
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
        if not chunk_text:
            return False
        letters = sum(c.isalpha() for c in chunk_text)
        return (letters / len(chunk_text)) >= self.min_letter_ratio

    def chunk(self, text: str) -> List[Dict]:
        sentences = split_into_sentences(text)

        chunks: List[Dict] = []
        buffer: List[str] = []
        buffer_words = 0

        def flush() -> None:
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
            keep = buffer[-self.overlap_sentences:] if self.overlap_sentences else []
            buffer = list(keep)
            buffer_words = sum(self._word_count(s) for s in buffer)

        for sent in sentences:
            sent_words = self._word_count(sent)

            if buffer_words + sent_words > self.max_words and buffer:
                flush()

            buffer.append(sent)
            buffer_words += sent_words

            if buffer_words >= self.target_words:
                flush()

        if buffer_words >= self.min_words:
            flush()

        return chunks

    def chunk_file(self, input_path: str | Path, output_path: str | Path) -> List[Dict]:
        input_path = Path(input_path)
        text = input_path.read_text(encoding="utf-8")
        if input_path.suffix == ".json":
            text = json.loads(text).get("text", text)
        chunks = self.chunk(text)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunks)} chunks -> {output_path}")
        return chunks
