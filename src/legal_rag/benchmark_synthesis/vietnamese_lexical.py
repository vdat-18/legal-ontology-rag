"""Vietnamese word segmentation for synthesis lexical retrieval."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

VNCORENLP_VERSION = "1.2"
VNCORENLP_JAR_NAME = f"VnCoreNLP-{VNCORENLP_VERSION}.jar"
VIETNAMESE_LEXICAL_TOKENIZER_VERSION = "vncorenlp_1_2_words_bigrams_v3"

_BOUNDARY_PREFIX = "LEGALRAGVNCOREBOUNDARY"


class VietnameseWordSegmenter(Protocol):
    """Word-segmentation boundary required by synthesis lexical retrieval."""

    def segment(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Return VnCoreNLP-style tokens for every input text in order."""


@dataclass(frozen=True, slots=True)
class VnCoreNlpWordSegmenter:
    """Run the pinned VnCoreNLP word segmenter once for a batch of texts."""

    java_path: str
    model_dir: Path
    max_heap_megabytes: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Reject runtime settings that VnCoreNLP cannot execute safely."""
        if not self.java_path:
            msg = "VnCoreNLP java_path must not be empty"
            raise ValueError(msg)
        if self.max_heap_megabytes < 1:
            msg = f"VnCoreNLP max_heap_megabytes must be positive, got {self.max_heap_megabytes}"
            raise ValueError(msg)
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            msg = f"VnCoreNLP timeout_seconds must be positive, got {self.timeout_seconds}"
            raise ValueError(msg)
        if any(character.isspace() for character in str(self.model_dir.resolve())):
            msg = "VnCoreNLP model_dir must not contain whitespace because VnCoreNLP 1.2 cannot load such paths"
            raise ValueError(msg)

    def segment(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Segment all texts in one JVM invocation and preserve input ordering."""
        if not texts:
            return ()

        model_dir = self.model_dir.resolve()
        with tempfile.TemporaryDirectory(prefix="legal-rag-vncorenlp-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input.txt"
            output_path = temporary_path / "output.txt"
            input_path.write_text(_build_vncorenlp_input(texts), encoding="utf-8")
            subprocess.run(  # noqa: S603
                [
                    self.java_path,
                    "-Dfile.encoding=UTF-8",
                    f"-Xmx{self.max_heap_megabytes}m",
                    "-jar",
                    str(model_dir / VNCORENLP_JAR_NAME),
                    "-fin",
                    str(input_path),
                    "-fout",
                    str(output_path),
                    "-annotators",
                    "wseg",
                ],
                cwd=model_dir,
                check=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            return _parse_vncorenlp_output(output_path.read_text(encoding="utf-8"), len(texts))


def _boundary(index: int) -> str:
    """Return one token VnCoreNLP preserves as an input boundary."""
    return f"{_BOUNDARY_PREFIX}{index:08d}"


def _build_vncorenlp_input(texts: Sequence[str]) -> str:
    """Append explicit boundary sentences so multi-sentence inputs can be reconstructed."""
    lines: list[str] = []
    for index, text in enumerate(texts):
        lines.extend((text, _boundary(index)))
    return "\n".join(lines) + "\n"


def _parse_vncorenlp_output(output: str, expected_text_count: int) -> tuple[tuple[str, ...], ...]:
    """Parse VnCoreNLP's tabular output into one token sequence per input text."""
    segmented: list[tuple[str, ...]] = []
    current_tokens: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        columns = line.split("\t")
        if len(columns) < 2:
            msg = f"Invalid VnCoreNLP output line: {line}"
            raise ValueError(msg)
        token = columns[1]
        if len(segmented) < expected_text_count and token == _boundary(len(segmented)):
            segmented.append(tuple(current_tokens))
            current_tokens = []
        else:
            current_tokens.append(token)

    if len(segmented) != expected_text_count or current_tokens:
        msg = f"VnCoreNLP returned {len(segmented)} completed texts for {expected_text_count} inputs"
        raise ValueError(msg)
    return tuple(segmented)
