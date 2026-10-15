"""Build evidence-backed support for a frozen candidate, without an answer field."""
from copy import deepcopy
import json
from uuid import uuid4
from .evidence import canonical
from .generation_common import check_budget, evidence_input, GenerationError
from .prompts import ATOM_BUILDER, GENERATION_TEMPLATE_VERSION
from .schemas import AtomBuildOutput, Atom, ReasoningDraft

FINAL_ATOM_ID = "a_final"

class AtomBuilder:
    def __init__(self, client, *, model, max_atoms=24, max_output_tokens=8192,
                 max_input_bytes=200000, max_attempts=2):
        self.client, self.model = client, model
        self.max_atoms, self.max_output_tokens = max_atoms, max_output_tokens
        self.max_input_bytes = max_input_bytes
        self.max_attempts = max(1, max_attempts)

    def _parse_output(self, text):
        return json.loads(text)

    def _validate(self, output):
        if output.status == "candidate" and (output.final_support is None or output.gaps):
            raise ValueError("A complete build requires final_support and no gaps")
        if len(output.atoms) + (output.status == "candidate") > self.max_atoms:
            raise ValueError("ATOM_BUDGET_EXCEEDED")
        if any(atom.atom_id == FINAL_ATOM_ID for atom in output.atoms):
            raise ValueError("RESERVED_FINAL_ATOM_ID")

    async def build(self, group, candidate, state=None):
        data = evidence_input(group)
        data["candidate"] = candidate.model_dump(mode="json")
        base_prompt = (ATOM_BUILDER
            + f"\natom 预算（含服务端最终 atom）：{self.max_atoms}\n"
            + canonical(data))
        check_budget(base_prompt, self.max_input_bytes)
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            feedback = ""
            if last_error is not None:
                feedback = ("\n\n上一次输出未通过校验："
                    + str(last_error)
                    + "。候选答案保持不变；只修正 atom 结构、引用与 final_support，"
                    + "不得修改或重新生成候选答案。\n")
            prompt = base_prompt + feedback
            check_budget(prompt, self.max_input_bytes)
            if state is not None:
                state["atom_building_call"] = {"model": self.model,
                    "template_version": GENERATION_TEMPLATE_VERSION,
                    "attempt": attempt, "prompt": prompt}
                state["generation_calls"] = state.get("generation_calls", 0) + 1
            text = await self.client.generate_text(prompt, model=self.model,
                max_tokens=self.max_output_tokens, temperature=0.0)
            if state is not None:
                state["atom_builder_output"] = text
            try:
                output = AtomBuildOutput.model_validate(self._parse_output(text))
                self._validate(output)
                return output
            except (ValueError, TypeError) as exc:
                last_error = exc
        raise GenerationError("INVALID_ATOM_BUILD", str(last_error)) from last_error

def assemble_draft(group, candidate, output):
    """All candidate identity and final answer content is copied by the server."""
    fields = dict(draft_id=uuid4().hex, group_id=group.group_id,
        candidate_id=candidate.candidate_id, candidate_sha256=candidate.candidate_sha256,
        status=output.status, gaps=list(output.gaps))
    if output.status == "insufficient":
        return ReasoningDraft(**fields, atoms=[])
    if candidate.status != "candidate" or output.final_support is None or output.gaps:
        raise GenerationError("INVALID_ATOM_BUILD")
    final = Atom(atom_id=FINAL_ATOM_ID, kind="answer",
        statement="对原问题 " + canonical(group.question) + " 的答案为：" + canonical(candidate.answer_value),
        value=deepcopy(candidate.answer_value),
        citations=deepcopy(output.final_support.citations),
        depends_on=list(output.final_support.depends_on))
    atoms = [Atom(**atom.model_dump(mode="json")) for atom in output.atoms]
    return ReasoningDraft(**fields, atoms=atoms + [final], final_atom_id=FINAL_ATOM_ID,
        answer_value=deepcopy(candidate.answer_value))
