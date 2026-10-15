"""Shared generation limits. Never silently truncate registered evidence."""

class ContextBudgetExceeded(ValueError):
    pass

class GenerationError(ValueError):
    def __init__(self, code, detail=None):
        super().__init__(detail or code)
        self.code = code

def check_budget(prompt, max_input_bytes):
    if len(prompt.encode("utf-8")) > max_input_bytes:
        raise ContextBudgetExceeded("CONTEXT_BUDGET_EXCEEDED")

def evidence_input(group):
    data = group.model_dump(mode="json")
    data.pop("tool_trace")
    return data
