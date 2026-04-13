from __future__ import annotations

from benchmarks.beam.dataset import BEAMQuestion
from benchmarks.beam.judge import judge_answer
from benchmarks.beam.run_beam import _parse_args


def _event_ordering_question() -> BEAMQuestion:
    return BEAMQuestion(
        question_id="10M_1_event_ordering_0",
        question="Reconstruct the ordered sequence of milestones.",
        category="event_ordering",
        rubric=[
            "token limits and segmentation",
            "reranking and feedback parsing",
            "language detection and vector alignment",
        ],
        ideal_response="",
        difficulty="medium",
    )


def test_fast_event_ordering_judge_matches_paraphrases_without_llm(monkeypatch) -> None:
    def _should_not_call(*args, **kwargs):
        raise AssertionError("fast event ordering judge should not call the judge LLM")

    monkeypatch.setattr("benchmarks.beam.judge._call_judge_llm", _should_not_call)

    result = judge_answer(
        _event_ordering_question(),
        "\n".join([
            "segmentation and token limit handling",
            "feedback parsing plus reranking problems",
            "vector alignment with language detection",
        ]),
        event_ordering_mode="fast",
    )

    assert result["method"] == "kendall_tau_fast"
    assert result["score"] == 1.0
    assert len(result["criteria_met"]) == 3


def test_fast_event_ordering_judge_penalizes_wrong_order(monkeypatch) -> None:
    def _should_not_call(*args, **kwargs):
        raise AssertionError("fast event ordering judge should not call the judge LLM")

    monkeypatch.setattr("benchmarks.beam.judge._call_judge_llm", _should_not_call)

    result = judge_answer(
        _event_ordering_question(),
        "\n".join([
            "vector alignment with language detection",
            "feedback parsing plus reranking problems",
            "segmentation and token limit handling",
        ]),
        event_ordering_mode="fast",
    )

    assert result["method"] == "kendall_tau_fast"
    assert result["score"] < 1.0
    assert result["tau_norm"] < 1.0


def test_run_beam_parse_args_accepts_fast_event_ordering_mode() -> None:
    args = _parse_args([
        "--chat-size", "10M",
        "--tagger-model", "claude-haiku-4-5-20251001",
        "--event-ordering-judge-mode", "fast",
    ])
    assert args.event_ordering_judge_mode == "fast"


def test_fast_event_ordering_judge_returns_zero_not_nan_for_no_matches(monkeypatch) -> None:
    def _should_not_call(*args, **kwargs):
        raise AssertionError("fast event ordering judge should not call the judge LLM")

    monkeypatch.setattr("benchmarks.beam.judge._call_judge_llm", _should_not_call)

    result = judge_answer(
        _event_ordering_question(),
        "\n".join([
            "completely unrelated infrastructure planning",
            "team retro and onboarding notes",
        ]),
        event_ordering_mode="fast",
    )

    assert result["method"] == "kendall_tau_fast"
    assert isinstance(float(result["score"]), float)
    assert result["score"] == result["score"]
    assert result["alignments"] == []
