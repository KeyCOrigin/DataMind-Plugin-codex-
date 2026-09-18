"""Skills tests — loader + code skills (no network required)."""
from __future__ import annotations

from pathlib import Path

import pytest

from datamind.capabilities.skills import (
    SkillManifest,
    discover_skills,
    load_skill,
)
from datamind.capabilities.skills.code_skills import build_code_skills
from datamind.capabilities.skills.service import SkillsService
from datamind.capabilities.skills.tools import build_skills_tools


# ------------------------------------------------------------- loader ---


def test_load_skill_with_frontmatter(tmp_path: Path):
    sk = tmp_path / "my-skill"
    sk.mkdir()
    (sk / "SKILL.md").write_text(
        "---\n"
        "name: my-skill\n"
        'description: "A test skill"\n'
        "keywords: [alpha, beta]\n"
        "---\n\n"
        "# Body\n\nSome content.\n",
        encoding="utf-8",
    )
    m = load_skill(sk / "SKILL.md")
    assert isinstance(m, SkillManifest)
    assert m.name == "my-skill"
    assert m.description == "A test skill"
    assert m.keywords == ("alpha", "beta")
    assert "Body" in m.body


def test_load_skill_without_frontmatter_falls_back(tmp_path: Path):
    sk = tmp_path / "no-fm"
    sk.mkdir()
    (sk / "SKILL.md").write_text("# First heading\n\nBody.\n", encoding="utf-8")
    m = load_skill(sk / "SKILL.md")
    assert m is not None
    assert m.name == "no-fm"
    # Description falls back to the first non-empty markdown line.
    assert "First heading" in m.description


def test_discover_skills_scans_subdirs(tmp_path: Path):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Skill {name}\n---\n\nBody {name}\n",
            encoding="utf-8",
        )
    # Also a directory without SKILL.md — must be skipped silently.
    (tmp_path / "c").mkdir()

    ms = discover_skills(tmp_path)
    assert [m.name for m in ms] == ["a", "b"]


def test_discover_skills_rejects_incomplete_manifests(tmp_path: Path):
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "SKILL.md").write_text(
        "---\nname: invalid\n---\n\nBody without description\n", encoding="utf-8",
    )
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "SKILL.md").write_text("---\nname: malformed\n", encoding="utf-8")

    assert discover_skills(tmp_path) == []


@pytest.mark.asyncio
async def test_profile_skill_override_and_tool_lookup_are_isolated(tmp_path: Path):
    base = tmp_path / "base"
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for root in (base, profile_a, profile_b):
        root.mkdir()
    (base / "shared").mkdir()
    (base / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: base\n---\n\nbase body\n", encoding="utf-8",
    )
    (profile_a / "shared").mkdir()
    (profile_a / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: override\n---\n\nprofile A body\n", encoding="utf-8",
    )
    (profile_a / "private").mkdir()
    (profile_a / "private" / "SKILL.md").write_text(
        "---\nname: private\ndescription: private\n---\n\nprivate body\n", encoding="utf-8",
    )

    service_a = SkillsService(
        skills_dir=base, profile_skills_dir=profile_a, embedding=None, vector_store=None,
    )
    service_b = SkillsService(
        skills_dir=base, profile_skills_dir=profile_b, embedding=None, vector_store=None,
    )
    await service_a.load()
    await service_b.load()

    assert service_a.get("shared")["body"] == "profile A body"
    assert service_a.get("private")["found"] is True
    assert service_b.get("private")["found"] is False

    get_tool = next(t for t in build_skills_tools(service_a) if t.name == "skill_get")
    assert (await get_tool.handler(name="shared"))["found"] is True


@pytest.mark.asyncio
async def test_skill_upsert_validates_and_reloads_manifest(tmp_path: Path):
    service = SkillsService(
        skills_dir=tmp_path / "base",
        profile_skills_dir=tmp_path / "profile",
        embedding=None,
        vector_store=None,
    )
    with pytest.raises(Exception, match="description"):
        await service.upsert(name="demo", description="", body="body")

    result = await service.upsert(
        name="Demo", description="A demo skill", body="Run the demo.", keywords=["demo"],
    )
    assert result["created"] is True
    assert service.get("demo")["found"] is True
    assert service.get("demo")["body"] == "Run the demo."


# ---------------------------------------------------------- code skills ---


@pytest.mark.asyncio
async def test_calculator_evaluates():
    spec = next(s for s in build_code_skills() if s.name == "calculator")
    out = await spec.handler(expression="2 * (3 + sqrt(16))")
    assert out["result"] == 14


@pytest.mark.asyncio
async def test_calculator_rejects_escape():
    spec = next(s for s in build_code_skills() if s.name == "calculator")
    with pytest.raises(ValueError):
        await spec.handler(expression="__import__('os').system('echo boom')")


@pytest.mark.asyncio
async def test_unit_convert_round_trip():
    spec = next(s for s in build_code_skills() if s.name == "unit_convert")
    a = await spec.handler(value=1.0, from_unit="km", to_unit="m")
    assert a["result"] == 1000
    b = await spec.handler(value=1000.0, from_unit="m", to_unit="km")
    assert b["result"] == 1.0


@pytest.mark.asyncio
async def test_unit_convert_unknown_pair_raises():
    spec = next(s for s in build_code_skills() if s.name == "unit_convert")
    with pytest.raises(ValueError):
        await spec.handler(value=1.0, from_unit="parsec", to_unit="m")


@pytest.mark.asyncio
async def test_analyze_text():
    spec = next(s for s in build_code_skills() if s.name == "analyze_text")
    out = await spec.handler(text="Hello world\n\nSecond paragraph.")
    assert out["paragraphs"] == 2
    assert out["words"] == 4


@pytest.mark.asyncio
async def test_get_current_time_fields():
    spec = next(s for s in build_code_skills() if s.name == "get_current_time")
    out = await spec.handler()
    assert "iso" in out and "weekday" in out
