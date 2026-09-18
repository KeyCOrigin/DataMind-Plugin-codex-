"""Deterministic RAG contract tests.

These tests deliberately avoid real LLMs, Embedding APIs, and Chroma.  They
prove data flow and failure semantics, not answer quality or recall quality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from datamind.agent.base import AgentLoopConfig
from datamind.agent.loop_native import NativeAgentLoop
from datamind.capabilities.ingest.formats import extract_document
from datamind.capabilities.kb.indexer import _split_text, build_index
from datamind.capabilities.kb.providers.simple_retriever import SimpleRetriever
from datamind.capabilities.kb.service import KBService
from datamind.capabilities.kb.filters import matches_metadata
from datamind.core.protocols import ModelResponse, ModelUsage, RetrievedChunk
from datamind.core.tools import ToolRegistry, ToolSpec


class _FakeEmbedding:
    name = "fake"
    dimension = 2

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "苹果" in text or "apple" in lowered:
            return [1.0, 0.0]
        if "香蕉" in text or "banana" in lowered:
            return [0.0, 1.0]
        return [0.7, 0.7]


class _VectorStore:
    dimension = 2

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, list[float], dict[str, Any]]] = {}

    async def add(self, ids, texts, embeddings, metadatas=None):
        metas = list(metadatas or [{} for _ in ids])
        for cid, text, vector, metadata in zip(ids, texts, embeddings, metas):
            self.rows[str(cid)] = (str(text), list(vector), dict(metadata))

    async def query(self, embedding, *, top_k=5, where=None):
        query = list(embedding)
        query_norm = math.sqrt(sum(value * value for value in query))
        ranked: list[tuple[float, int, RetrievedChunk]] = []
        for position, (cid, (text, vector, metadata)) in enumerate(self.rows.items()):
            if not matches_metadata(metadata, where):
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = sum(a * b for a, b in zip(query, vector)) / (query_norm * vector_norm)
            ranked.append((score, position, RetrievedChunk(
                id=cid, text=text, score=score,
                source=metadata.get("source"), metadata=dict(metadata),
            )))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:top_k]]

    async def count(self):
        return len(self.rows)

    async def delete(self, ids):
        for cid in ids:
            self.rows.pop(str(cid), None)

    async def reset(self):
        self.rows.clear()

    async def get_all_texts(self):
        return [(cid, text, metadata) for cid, (text, _, metadata) in self.rows.items()]


def test_chunk_boundaries_are_deterministic_and_loss_bounded():
    cases = [
        ("short", 20, 0),
        ("x" * 20, 20, 0),
        ("苹果 香蕉 " * 20, 17, 0),
        ("无标点" * 30, 13, 4),
    ]
    for text, size, overlap in cases:
        chunks = _split_text(text, chunk_size=size, chunk_overlap=overlap)
        assert all(chunk for chunk in chunks)
        assert all(len(chunk) <= size for chunk in chunks)
        reconstructed = "".join(chunks).replace(" ", "")
        assert all(char in reconstructed for char in text if not char.isspace())


@pytest.mark.parametrize(
    ("size", "overlap", "message"),
    [(0, 0, "chunk_size"), (-1, 0, "chunk_size"), (10, -1, "chunk_overlap"), (10, 10, "chunk_overlap")],
)
def test_chunk_parameters_fail_before_processing(size, overlap, message):
    with pytest.raises(ValueError, match=message):
        _split_text("data", chunk_size=size, chunk_overlap=overlap)


def test_document_parser_preserves_source_text_and_metadata(tmp_path):
    source = tmp_path / "中文 文件 😀.md"
    source.write_text("标题\n\n关键测试字符串：BlueBird", encoding="utf-8")

    document = extract_document(source)

    assert document.source == str(source.resolve())
    assert document.format == "md"
    assert document.text
    assert "关键测试字符串" in document.text
    assert document.blocks[0]["type"] == "text"


@pytest.mark.asyncio
async def test_index_pipeline_preserves_source_and_chunk_metadata(tmp_path):
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    (data_dir / "facts.md").write_text("苹果负责人是张三。\n预算为120万元。", encoding="utf-8")
    store = _VectorStore()

    stats = await build_index(
        data_dir=data_dir, vector_store=store, embedding=_FakeEmbedding(),
        chunk_size=512, chunk_overlap=0,
    )

    assert stats["total_embedded"] == 1
    assert len(store.rows) == 1
    text, vector, metadata = next(iter(store.rows.values()))
    assert "苹果负责人" in text
    assert vector == [1.0, 0.0]
    assert metadata["source"] == "facts.md"
    assert metadata["_origin"] == "raw"
    assert metadata["_chunk_ordinal"] == 0


@pytest.mark.asyncio
async def test_vector_store_crud_and_metadata_scope_are_deterministic():
    store = _VectorStore()
    await store.add(["c1"], ["苹果"], [[1.0, 0.0]], [{"workspace": "a", "source": "doc.md"}])
    await store.add(["c1"], ["苹果更新"], [[1.0, 0.0]], [{"workspace": "a", "source": "doc-v2.md"}])

    assert await store.count() == 1
    assert (await store.query([1.0, 0.0], top_k=5))[0].text == "苹果更新"
    assert await store.query([1.0, 0.0], where={"workspace": "b"}) == []

    await store.delete(["c1"])
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_retriever_has_stable_rank_filter_and_invalid_input_contracts():
    store = _VectorStore()
    await store.add(
        ["c1", "c2", "c3"],
        ["苹果", "香蕉", "水果"],
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        [{"workspace": "a"}, {"workspace": "a"}, {"workspace": "b"}],
    )
    retriever = SimpleRetriever(vector_store=store, embedding=_FakeEmbedding())

    hits = await retriever.aretrieve("苹果", top_k=2, filters={"workspace": "a"})
    assert [hit.id for hit in hits] == ["c1", "c2"]
    assert hits[0].score >= hits[1].score
    assert await retriever.aretrieve("苹果", top_k=5, filters={"workspace": "missing"}) == []
    with pytest.raises(ValueError, match="top_k"):
        await retriever.aretrieve("苹果", top_k=0)
    with pytest.raises(ValueError, match="query"):
        await retriever.aretrieve("  ")


@pytest.mark.asyncio
async def test_kb_service_wires_index_retrieve_and_empty_failure_contract(tmp_path):
    store = _VectorStore()
    embedding = _FakeEmbedding()
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    (data_dir / "doc.md").write_text("苹果项目预算", encoding="utf-8")
    await build_index(
        data_dir=data_dir, vector_store=store, embedding=embedding,
        chunk_size=100, chunk_overlap=0,
    )
    service = KBService(
        embedding=embedding,
        vector_store=store,
        retriever=SimpleRetriever(vector_store=store, embedding=embedding),
        data_dir=data_dir,
        retrieval_cfg=type("Retrieval", (), {"top_k": 5})(),
    )

    result = await service.search("苹果", top_k=1)
    assert result[0]["text"] == "苹果项目预算"
    assert result[0]["metadata"]["source"] == "doc.md"
    with pytest.raises(ValueError, match="top_k"):
        await service.search("苹果", top_k=0)
    with pytest.raises(ValueError, match="query"):
        await service.search(" ")


@dataclass
class _ScriptClient:
    responses: list[ModelResponse]
    calls: list[dict[str, Any]]

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_retrieval_context_and_evidence_mapping_use_same_chunk_ids():
    async def search(query: str, top_k: int = 5) -> dict[str, Any]:
        return {"query": query, "results": [
            {"id": "c1", "source": "facts.md", "text": "项目代号 BlueBird", "score": 1.0},
            {"id": "c2", "source": "facts.md", "text": "负责人 张三", "score": 0.9},
        ]}

    registry = ToolRegistry()
    registry.add(ToolSpec(
        name="kb_search", description="search", input_schema={"type": "object"},
        handler=search, metadata={"surface": "kb", "access": "read"},
    ))
    client = _ScriptClient(
        responses=[
            ModelResponse(
                content=[{"type": "tool_use", "id": "call-1", "name": "kb_search", "input": {"query": "项目代号"}}],
                stop_reason="tool_use", usage=ModelUsage(input_tokens=1, output_tokens=1),
            ),
            ModelResponse(
                content=[{"type": "text", "text": "已找到证据。"}],
                stop_reason="end_turn", usage=ModelUsage(input_tokens=1, output_tokens=1),
            ),
        ],
        calls=[],
    )
    loop = NativeAgentLoop(
        client=client, tools=registry,
        config=AgentLoopConfig(model="fake", max_tool_turns=2, system_prompt="system"),
    )

    result = await loop.run_turn(user_message="项目代号是什么？")

    assert result["answer"] == "已找到证据。"
    assert [item["locator"]["chunk_id"] for item in result["evidence"]] == ["c1", "c2"]
    tool_result_messages = [
        block for message in client.calls[1]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_result_messages
    assert "项目代号 BlueBird" in tool_result_messages[0]["content"]
