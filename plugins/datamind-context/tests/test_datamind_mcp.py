from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "datamind_mcp.py"
SPEC = importlib.util.spec_from_file_location("datamind_mcp", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
datamind_mcp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(datamind_mcp)


def test_read_only_file_tools_skip_embedding_surfaces():
    assert datamind_mcp.enabled_surfaces("datamind_raw_file_read") == {"graph"}
    assert datamind_mcp.enabled_surfaces("datamind_workspace_inspect") == {"graph"}


def test_direct_surface_tools_build_only_required_services():
    assert datamind_mcp.enabled_surfaces("datamind_rag_query") == {"kb"}
    assert datamind_mcp.enabled_surfaces("datamind_table_ingest") == {"db"}
    assert datamind_mcp.enabled_surfaces("datamind_graph_query") == {"graph"}
    assert datamind_mcp.enabled_surfaces("datamind_remember") == {"memory"}


def test_agent_and_router_tools_keep_all_surfaces():
    assert datamind_mcp.enabled_surfaces("datamind_ask") is None
    assert datamind_mcp.enabled_surfaces("datamind_store") is None
    assert datamind_mcp.enabled_surfaces("datamind_surface_ingest") is None


@pytest.mark.asyncio
async def test_execute_reuses_runtime_until_factory_shutdown(monkeypatch):
    """Skill manifests warm once and shared MCP resources close on shutdown."""

    class FakeSpec:
        async def handler(self, **_kwargs):
            return {"results": []}

    class FakeTools:
        def get(self, _name):
            return FakeSpec()

    class FakeSystem:
        retrieve = types.SimpleNamespace(tools=FakeTools())
        store = types.SimpleNamespace(tools=FakeTools())

        def __init__(self):
            self.warmup_calls = 0
            self.closed = False

        async def warmup(self):
            self.warmup_calls += 1
            return {"skills": {"manifests": 1}}

        async def aclose(self):
            self.closed = True

    built: list[FakeSystem] = []

    async def build(_settings, *, enable):
        assert enable == {"kb"}
        fake = FakeSystem()
        built.append(fake)
        return fake

    class FakeSettings:
        def __init__(self):
            self.data = types.SimpleNamespace(profile="default")

    monkeypatch.setattr("datamind.config.Settings", FakeSettings)
    monkeypatch.setattr("datamind.agent.build_datamind", build)
    factory = datamind_mcp.RuntimeFactory()
    result = await datamind_mcp.execute("datamind_rag_query", {"query": "x"}, runtime_factory=factory)
    again = await datamind_mcp.execute("datamind_rag_query", {"query": "y"}, runtime_factory=factory)

    assert result == {"results": []}
    assert again == {"results": []}
    assert len(built) == 1
    assert built[0].warmup_calls == 1
    assert built[0].closed is False
    await factory.aclose()
    assert built[0].closed is True
