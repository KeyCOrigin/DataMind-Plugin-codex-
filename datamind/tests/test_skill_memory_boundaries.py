"""Boundary contracts for Skill and Memory capabilities.

These tests avoid model quality judgments.  They exercise validation,
isolation, lifecycle and failure behavior with deterministic fakes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from datamind.capabilities.memory import ShortTermMemory
from datamind.capabilities.memory.providers.sqlite_store import SQLiteMemoryStore
from datamind.capabilities.skills.service import SkillsService
from datamind.core.errors import CapabilityError


class _Embed:
    name = "fake"
    dimension = 2

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, query):
        return [1.0, 0.0]


class _BadCountEmbed(_Embed):
    async def embed_texts(self, texts):
        return []


class _BadDimensionEmbed(_Embed):
    async def embed_texts(self, texts):
        return [[1.0] for _ in texts]


class _BadValueEmbed(_Embed):
    async def embed_texts(self, texts):
        return [[True, 0.0] for _ in texts]


class _SkillStore:
    def __init__(self):
        self.reset_calls = 0
        self.added = []

    async def reset(self):
        self.reset_calls += 1
        self.added.clear()

    async def add(self, ids, texts, embeddings, metadatas=None):
        self.added.extend(zip(ids, texts, embeddings, metadatas or []))

    async def query(self, embedding, *, top_k=3, where=None):
        return []


def _write_skill(root: Path, name: str = "demo") -> None:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill\nkeywords: [demo]\n---\n\n# Body\n\nUse it.\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_skill_search_rejects_empty_and_non_positive_queries(tmp_path: Path):
    service = SkillsService(
        skills_dir=tmp_path / "skills", embedding=None, vector_store=None,
    )
    for query in ("", "   "):
        with pytest.raises(CapabilityError, match="non-empty"):
            await service.search(query)
    for top_k in (0, -1, True):
        with pytest.raises(CapabilityError, match="top_k"):
            await service.search("query", top_k=top_k)


@pytest.mark.asyncio
async def test_skill_load_rejects_malformed_embedding_without_resetting_store(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills)
    for embedding, message in (
        (_BadCountEmbed(), "count mismatch"),
        (_BadDimensionEmbed(), "dimension mismatch"),
        (_BadValueEmbed(), "non-finite or non-numeric"),
    ):
        store = _SkillStore()
        service = SkillsService(skills_dir=skills, embedding=embedding, vector_store=store)
        with pytest.raises(CapabilityError, match=message):
            await service.load()
        assert store.reset_calls == 0


@pytest.mark.asyncio
async def test_skill_upsert_rejects_types_and_preserves_overwrite_boundary(tmp_path: Path):
    service = SkillsService(
        skills_dir=tmp_path / "base", profile_skills_dir=tmp_path / "profile",
        embedding=None, vector_store=None,
    )
    invalid = [
        {"name": 1, "description": "d", "body": "b"},
        {"name": "demo", "description": 1, "body": "b"},
        {"name": "demo", "description": "d", "body": 1},
        {"name": "demo", "description": "d", "body": "b", "keywords": "demo"},
        {"name": "demo", "description": "d", "body": "b", "keywords": [1]},
    ]
    for arguments in invalid:
        with pytest.raises(CapabilityError):
            await service.upsert(**arguments)

    await service.upsert(name="demo", description="d", body="v1")
    with pytest.raises(CapabilityError, match="already exists"):
        await service.upsert(name="demo", description="d", body="v2", overwrite=False)
    assert service.get("demo")["body"] == "v1"


@pytest.mark.asyncio
async def test_memory_top_k_scope_budget_and_kind_boundaries(tmp_path: Path):
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.db", embedding=None)
    await store.save("profile fact", scope="profile", profile="A")

    for top_k in (0, -1, True):
        with pytest.raises(CapabilityError, match="top_k"):
            await store.recall("fact", profile="A", top_k=top_k)
    with pytest.raises(CapabilityError, match="per_scope"):
        await store.recall("fact", profile="A", per_scope={"profile": -1})
    with pytest.raises(CapabilityError, match="unsupported memory scope"):
        await store.recall("fact", profile="A", per_scope={"tenant": 1})
    with pytest.raises(CapabilityError, match="supported memory kinds"):
        await store.recall("fact", profile="A", kinds=["unknown"])


@pytest.mark.asyncio
async def test_memory_scope_fields_and_metadata_are_not_silently_discarded(tmp_path: Path):
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.db", embedding=None)
    invalid = [
        {"scope": "global", "profile": "A"},
        {"scope": "global", "session_id": "s"},
        {"scope": "profile", "profile": "A", "session_id": "s"},
        {"scope": "session", "profile": "A", "session_id": "s"},
        {"scope": "global", "metadata": []},
        {"scope": "global", "metadata": {"bad": object()}},
    ]
    for arguments in invalid:
        with pytest.raises(CapabilityError):
            await store.save("fact", **arguments)
    assert await store.count(include_archived=True) == 0


@pytest.mark.asyncio
async def test_memory_embedding_boolean_coordinate_is_rejected(tmp_path: Path):
    class _BoolEmbed(_Embed):
        dimension = 2

        async def embed_query(self, query):
            return [True, 0.0]

    store = SQLiteMemoryStore(db_path=tmp_path / "memory.db", embedding=_BoolEmbed())
    with pytest.raises(CapabilityError, match="non-finite or non-numeric"):
        await store.save("fact", scope="global")
    assert await store.count(include_archived=True) == 0


@pytest.mark.asyncio
async def test_short_term_memory_rejects_invalid_turns_and_zero_limit():
    with pytest.raises(ValueError, match="max_turns"):
        ShortTermMemory(max_turns=0)
    memory = ShortTermMemory(max_turns=2)
    for args in (("", "user", "x"), ("s", "other", "x"), ("s", "user", " ")):
        with pytest.raises(ValueError):
            await memory.append(*args)
    await memory.append("s", "user", "one")
    await memory.append("s", "assistant", "two")
    assert await memory.recent("s", limit=0) == []
    with pytest.raises(ValueError, match="limit"):
        await memory.recent("s", limit=-1)

