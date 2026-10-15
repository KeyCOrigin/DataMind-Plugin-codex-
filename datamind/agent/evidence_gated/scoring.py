"""True three-label scores, stable normalization and strict threshold boundaries."""
import math
from datamind.core.protocols import LabelScores

LABELS = ("A", "B", "C")
SCORE_TYPE = "raw_abc_expected_value"

class ScoringUnavailable(RuntimeError):
    pass

def validate_thresholds(support, refute):
    if not all(math.isfinite(x) for x in (support, refute)) or not -1 <= refute < support <= 1:
        raise ValueError("Require -1 <= refute_threshold < support_threshold <= 1")

def normalize_abc(logprobs):
    if set(logprobs) != set(LABELS) or any(
            not math.isfinite(v) or not -9900 < v <= 0 for v in logprobs.values()):
        raise ValueError("Require finite actual ABC logprobs")
    m = max(logprobs.values())
    weights = {k: math.exp(logprobs[k] - m) for k in LABELS}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def classify(score, support_threshold, refute_threshold):
    validate_thresholds(support_threshold, refute_threshold)
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("Invalid expected score")
    return "support" if score > support_threshold else "refute" if score < refute_threshold else "uncertain"

class TokenScoringAdapter:
    def __init__(self, client, *, model, disable_thinking=False):
        self.client, self.model = client, model
        self.disable_thinking = disable_thinking

    async def score_labels(self, prompt, labels=LABELS):
        method = getattr(self.client, "score_labels", None)
        if not callable(method):
            raise ScoringUnavailable("SCORING_UNAVAILABLE")
        try:
            if list(labels) != list(LABELS):
                raise ValueError("ABC protocol required")
            result = await method(prompt, labels=list(LABELS), model=self.model,
                                  disable_thinking=self.disable_thinking)
            if not isinstance(result, LabelScores):
                raise ValueError("Client must return LabelScores with token metadata")
            # Revalidate even model_construct / in-place mutation from an external client.
            return LabelScores.model_validate(result.model_dump())
        except Exception as exc:
            raise ScoringUnavailable("SCORING_UNAVAILABLE") from exc
