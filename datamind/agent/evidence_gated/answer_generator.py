"""Generate a candidate before any atoms exist; identities belong to the server."""
import json
from uuid import uuid4
from .evidence import canonical, candidate_digest
from .generation_common import check_budget, evidence_input, GenerationError
from .prompts import ANSWER_GENERATOR, GENERATION_TEMPLATE_VERSION
from .schemas import AnswerGenerationOutput, AnswerCandidate

class AnswerGenerator:
    def __init__(self, client, *, model, max_output_tokens=4096, max_input_bytes=200000):
        self.client, self.model = client, model
        self.max_output_tokens, self.max_input_bytes = max_output_tokens, max_input_bytes

    async def generate(self, group, state=None):
        prompt = ANSWER_GENERATOR + "\n" + canonical(evidence_input(group))
        check_budget(prompt, self.max_input_bytes)
        if state is not None:
            state["answer_generation_call"] = {"model": self.model,
                "template_version": GENERATION_TEMPLATE_VERSION, "prompt": prompt}
            state["generation_calls"] = state.get("generation_calls", 0) + 1
        text = await self.client.generate_text(prompt, model=self.model,
            max_tokens=self.max_output_tokens, temperature=0.0)
        if state is not None:
            state["answer_generator_output"] = text
        try:
            output = AnswerGenerationOutput.model_validate(json.loads(text))
            if output.status == "insufficient" and output.answer_value is not None:
                raise ValueError("An insufficient candidate cannot carry an answer")
            if output.status == "candidate" and output.gaps:
                raise ValueError("Candidate has unresolved gaps")
        except (ValueError, TypeError) as exc:
            raise GenerationError("INVALID_CANDIDATE", str(exc)) from exc
        candidate = AnswerCandidate(**output.model_dump(mode="json"),
            candidate_id=uuid4().hex, request_id=group.request_id, group_id=group.group_id)
        candidate.candidate_sha256 = candidate_digest(group, candidate)
        return candidate
