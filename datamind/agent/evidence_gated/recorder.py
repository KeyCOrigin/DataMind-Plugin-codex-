"""Observe-only hook; ContextVar isolates concurrent requests and nested calls."""
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
from pydantic import TypeAdapter
from typing import Any
from datamind.core.hooks import Allow
from .schemas import ToolRecord

_records: ContextVar[list | None] = ContextVar("evidence_tool_records", default=None)
_pending: ContextVar[dict | None] = ContextVar("evidence_pending_calls", default=None)

def start_recording():
    return _records.set([]), _pending.set({})

def captured():
    return deepcopy(_records.get() or [])

def stop_recording(tokens):
    _records.reset(tokens[0])
    _pending.reset(tokens[1])

class EvidenceRecorderHook:
    name = "evidence_recorder"
    async def pre_tool_use(self, ctx, tool_name, args):
        pending = _pending.get()
        if pending is not None:
            from .evidence import canonical
            key = (tool_name, canonical(args))
            pending.setdefault(key, []).append((datetime.now(timezone.utc).isoformat(), perf_counter()))
        return Allow()

    async def post_tool_use(self, ctx, tool_name, args, result, error):
        records = _records.get()
        if records is None:
            return
        from .evidence import canonical
        times = (_pending.get() or {}).get((tool_name, canonical(args)), [])
        started = times.pop(0) if times else None
        # Copy now; later mutations of a tool result must never alter evidence.
        try:
            # KB and Graph return nested Pydantic DTOs, not just plain dicts.
            payload = TypeAdapter(Any).dump_python(deepcopy(result), mode="json")
            canonical(payload)  # Reject non-finite/unsupported serialized values.
        except Exception as exc:
            payload = {"serialization_error": type(exc).__name__}
            error = error or RuntimeError("UNSERIALIZABLE_TOOL_RESULT")
        records.append(ToolRecord(call_id=uuid4().hex, tool_name=tool_name,
            args=deepcopy(args), result=payload, error=str(error) if error else None,
            started_at=started[0] if started else None,
            read_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=(perf_counter() - started[1]) * 1000 if started else None))
