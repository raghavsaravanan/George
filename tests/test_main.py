"""Unit tests for George voice formatting helpers (no Qdrant required)."""

from __future__ import annotations

from main import (
    format_number_for_speech,
    format_voice_answer,
    query_normalizer,
    to_single_line,
)

_CAUTION_CHUNK = """| Application | Torque | Notes |
| --- | --- | --- |
| Intake Bolts | 15-18 foot-pounds | |
Caution: Serious damage can occur when ignition is not installed correctly.
"""


def test_part_number_226_020_speech_spaced() -> None:
    spoken = format_number_for_speech(to_single_line("226.020"))
    assert spoken == "2 2 6 0 2 0"


def test_part_number_12_13_speech_spaced() -> None:
    spoken = format_number_for_speech(to_single_line("12.13"))
    assert spoken == "1 2 1 3"


def test_query_normalizer_collapses_digit_gaps() -> None:
    assert query_normalizer("20. 19") == "2019"
    assert "2019" in query_normalizer("20. 19 Silverado")


def test_format_voice_answer_suppresses_caution_on_torque_query() -> None:
    spoken = format_voice_answer(
        _CAUTION_CHUNK,
        "What's the intake manifold bolt torque?",
    )
    assert "15-18 foot-pounds" in spoken
    assert "ignition" not in spoken.lower()
    assert "caution" not in spoken.lower()


def test_format_voice_answer_includes_caution_when_asked() -> None:
    spoken = format_voice_answer(
        _CAUTION_CHUNK,
        "Any safety caution for intake bolts?",
    )
    assert "15-18 foot-pounds" in spoken
    assert "ignition" in spoken.lower() or "caution" in spoken.lower()
