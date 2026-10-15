"""DeepSeek-style token logprob scoring (no network)."""
from __future__ import annotations

import asyncio

import pytest

from datamind.core.errors import ExternalServiceError
from datamind.core.model_clients import OpenAIChatCompletionsModelClient
from datamind.agent.evidence_gated import ScoringUnavailable, TokenScoringAdapter


def _body(tokens: list[tuple[str, float]]) -> dict:
    """Build a DeepSeek-shaped logprobs body with one content position."""
    return {
        "model":"fixture",
        "choices": [{
            "logprobs": {
                "content": [{
                    "token": tokens[0][0],
                    "logprob": tokens[0][1],
                    "top_logprobs": [{"token": tok, "logprob": lp} for tok, lp in tokens],
                }]
            }
        }]
    }


def test_parse_score_labels_rejects_leading_space():
    body = _body([("A", -0.3), (" B", -0.7), ("C", -3.0)])
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(body, ["A", "B", "C"])


def test_parse_score_labels_sampled_token_position():
    body = _body([("B", -0.4), ("A", -0.9), ("C",-5.0)])
    scores = OpenAIChatCompletionsModelClient._parse_score_labels(body, ["A", "B", "C"])
    assert scores.logprobs["A"] == -0.9 and scores.logprobs["B"] == -0.4
    assert scores.position == 0 and scores.tokens == {"A":"A", "B":"B", "C":"C"}


def test_parse_score_labels_missing_label_raises():
    body = _body([("A", -0.3), ("C", -3.0)])
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(body, ["A", "B", "C"])


def test_parse_score_labels_sentinel_raises():
    body = _body([("A", -0.3), ("B", -9999.0)])
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(body, ["A", "B", "C"])


def test_parse_score_labels_no_logprobs_raises():
    body = {"choices": [{"message": {"content": "A"}}]}
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(body, ["A", "B", "C"])


class _FakeScoringClient:
    def __init__(self, scores: dict[str, float] | None = None, exc: Exception | None = None):
        self._scores = scores
        self._exc = exc

    async def score_labels(self, prompt, *, labels, model=None, disable_thinking=False):
        if self._exc is not None:
            raise self._exc
        from datamind.core.protocols import LabelScores
        return LabelScores(logprobs=self._scores,
            model=model, call_id="fixture", tokens={"A":"A", "B":"B", "C":"C"})


def test_adapter_preserves_independent_label_scores():
    adapter = TokenScoringAdapter(_FakeScoringClient({"A": -0.3, "B": -0.7, "C":-5.0}), model="m")
    scores = asyncio.run(adapter.score_labels("prompt"))
    assert scores.logprobs["A"] == -0.3 and scores.logprobs["B"] == -0.7


def test_adapter_wraps_client_error():
    adapter = TokenScoringAdapter(
        _FakeScoringClient(exc=ExternalServiceError("llm", "no logprobs")),
        model="m",
    )
    with pytest.raises(ScoringUnavailable):
        asyncio.run(adapter.score_labels("prompt"))


def test_adapter_requires_score_labels():
    adapter = TokenScoringAdapter(object(), model="m")
    with pytest.raises(ScoringUnavailable):
        asyncio.run(adapter.score_labels("prompt"))
