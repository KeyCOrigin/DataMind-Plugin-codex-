"""Scoring-gate compatibility: thinking disabled, build-time probe, decoupled model."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from datamind.agent.evidence_gated.scoring import ScoringUnavailable, TokenScoringAdapter
from datamind.agent.options import _probe_scoring
from datamind.config import EvidenceGateConfig
from datamind.core.errors import ConfigError
from datamind.core.model_clients import OpenAIChatCompletionsModelClient


def _ab_body() -> dict:
    return {
        "model":"fixture",
        "choices": [{
            "logprobs": {
                "content": [{
                    "token": "A",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "A", "logprob": -0.1},
                        {"token": "B", "logprob": -2.3},
                        {"token": "C", "logprob": -5.0},
                    ],
                }]
            }
        }]
    }


def test_score_labels_disables_thinking_in_payload():
    async def run():
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=_ab_body())

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wire = OpenAIChatCompletionsModelClient(
            api_key="k", api_base="https://x.test", default_model="m", client=http,
        )
        await wire.score_labels("p", labels=["A", "B", "C"], disable_thinking=True)
        await wire.aclose()
        assert requests[0]["thinking"] == {"type": "disabled"}
        assert requests[0]["max_tokens"] == 1 and requests[0]["logprobs"] is True

    asyncio.run(run())


def test_score_labels_leaves_thinking_absent_by_default():
    async def run():
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=_ab_body())

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        wire = OpenAIChatCompletionsModelClient(
            api_key="k", api_base="https://x.test", default_model="m", client=http,
        )
        await wire.score_labels("p", labels=["A", "B", "C"])
        await wire.aclose()
        assert "thinking" not in requests[0]

    asyncio.run(run())


def test_probe_fails_closed_when_scoring_unavailable():
    class BadAdapter:
        async def score_labels(self, prompt):
            raise ScoringUnavailable("SCORING_UNAVAILABLE")

    with pytest.raises(ConfigError):
        asyncio.run(_probe_scoring(BadAdapter()))


def test_probe_passes_when_abc_returned():
    from datamind.core.protocols import LabelScores

    class GoodAdapter:
        async def score_labels(self, prompt):
            return LabelScores(
                logprobs={"A":-0.1, "B":-2.3, "C":-5.0}, model="m", call_id="c",
                tokens={"A": "A", "B": "B", "C": "C"},
            )

    asyncio.run(_probe_scoring(GoodAdapter()))


def test_scoring_config_defaults_decouple_model():
    cfg = EvidenceGateConfig()
    assert cfg.scoring_model is None
    assert cfg.scoring_protocol == "openai_chat_completions"
    assert cfg.scoring_disable_thinking is True
    assert EvidenceGateConfig(scoring_model="deepseek-v4-pro").scoring_model == "deepseek-v4-pro"


def test_adapter_forwards_disable_thinking():
    seen = {}

    class FakeClient:
        async def score_labels(self, prompt, *, labels, model=None, disable_thinking=False):
            seen["disable_thinking"] = disable_thinking
            from datamind.core.protocols import LabelScores
            return LabelScores(
                logprobs={"A":-0.1, "B":-2.3, "C":-5.0}, model=model, call_id="c",
                tokens={"A": "A", "B": "B", "C": "C"},
            )

    async def run():
        await TokenScoringAdapter(FakeClient(), model="m", disable_thinking=True).score_labels("p")

    asyncio.run(run())
    assert seen["disable_thinking"] is True
