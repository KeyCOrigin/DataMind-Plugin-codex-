"""One original-question retrieval; only observed tool payloads are evidence."""
import asyncio
from uuid import uuid4
from .evidence import EvidenceRegistry, canonical, digest
from .recorder import captured, start_recording, stop_recording
from .schemas import EvidenceGroup

class RetrievalBridge:
    def __init__(self, retrieve_agent, *, registry=None, timeout_s=180.0):
        self.retrieve_agent = retrieve_agent
        self.registry = registry or EvidenceRegistry()
        self.timeout_s = timeout_s

    async def collect(self, *, question, context, answer_contract, history=None, request_id=None, state=None):
        tokens = start_recording()
        cancelled = False
        try:
            message = question
            if context:
                message += "\nExplicit runtime request context (do not infer additional identifiers):\n" + canonical(context)
            try:
                # Retrieval's answer and model-written normalized evidence are ignored.
                raw = await asyncio.wait_for(self.retrieve_agent.query(message, history=history, final_contract=None), self.timeout_s)
                if state is not None and isinstance(raw, dict):
                    state["retrieval_usage"] = raw.get("usage", {})
                    state["retrieval_iterations"] = raw.get("iterations", 0)
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:
                context = {**context, "retrieval_error": type(exc).__name__}
            trace = captured()
        finally:
            stop_recording(tokens)
        evidence = self.registry.register(trace, context)
        identity = digest({"question": question, "context": context, "contract": answer_contract,
                           "evidence": [e.model_dump(mode="json") for e in evidence]})
        group = EvidenceGroup(request_id=request_id or uuid4().hex, question=question, context=context,
            answer_contract=answer_contract, group_id=identity, evidence=evidence, tool_trace=trace)

        if state is not None:
            state["group"] = group
        if cancelled:
            raise asyncio.CancelledError
        return group
