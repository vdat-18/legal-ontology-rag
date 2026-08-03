"""Tests for VnCoreNLP-backed Vietnamese lexical segmentation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis import vietnamese_lexical
from legal_rag.benchmark_synthesis.vietnamese_lexical import VnCoreNlpWordSegmenter


def test_vncorenlp_segments_multiple_sentences_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One JVM batch is reconstructed into exactly one token sequence per source text."""
    calls: list[tuple[list[str], Path]] = []

    def _run(command: list[str], *, cwd: Path, **options: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, cwd))
        output_path = Path(command[command.index("-fout") + 1])
        output_path.write_text(
            "\n".join(
                (
                    "1\tKhông\t_\t_\t_\t\t_",
                    "2\tchấp_hành\t_\t_\t_\t\t_",
                    "",
                    f"1\t{vietnamese_lexical._boundary(0)}\t_\t_\t_\t\t_",
                    "",
                    "1\tÔ_tô\t_\t_\t_\t\t_",
                    "2\t.\t_\t_\t_\t\t_",
                    "",
                    f"1\t{vietnamese_lexical._boundary(1)}\t_\t_\t_\t\t_",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vietnamese_lexical.subprocess, "run", _run)
    segmenter = VnCoreNlpWordSegmenter(
        java_path="java",
        model_dir=Path("/vncorenlp"),
        max_heap_megabytes=512,
        timeout_seconds=120,
    )

    segmented = segmenter.segment(("Không chấp hành", "Ô tô."))

    assert segmented == (("Không", "chấp_hành"), ("Ô_tô", "."))
    assert len(calls) == 1
    assert calls[0][1] == Path("/vncorenlp").resolve()
    assert calls[0][0][:4] == ["java", "-Dfile.encoding=UTF-8", "-Xmx512m", "-jar"]


def test_vncorenlp_rejects_model_directories_with_spaces() -> None:
    """VnCoreNLP 1.2 fails to locate its own models when the JAR path contains spaces."""
    with pytest.raises(ValueError, match="must not contain whitespace"):
        VnCoreNlpWordSegmenter(
            java_path="java",
            model_dir=Path("vncorenlp models"),
            max_heap_megabytes=512,
            timeout_seconds=120,
        )


def test_vncorenlp_rejects_incomplete_batch_output() -> None:
    """Missing boundary records cannot silently shift tokens between facts."""
    with pytest.raises(ValueError, match="returned 0 completed texts for 1 inputs"):
        vietnamese_lexical._parse_vncorenlp_output("1\tKhông\t_\t_\t_\t\t_\n", 1)
