"""Exercise the public experiment CLI, including a real child process exit."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_cli(*args):
    result = subprocess.run([sys.executable, "-m", "benchmark.robustness", *args],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.asyncio
async def test_embedding_interruption_counts_batches_within_one_request(tmp_path, monkeypatch):
    from benchmark.robustness import EMBEDDING_CRASH_STAGE, observe_build_embedding
    from datamind.capabilities.embedding.providers.huggingface import HuggingFaceEmbedding

    forwarded = []

    class Batch(list):
        def cpu(self):
            return self

    def forward(texts):
        forwarded.append(list(texts))
        return {"sentence_embedding": Batch(texts)}

    async def embed(provider, texts):
        for start in range(0, len(texts), provider._batch_size):
            provider._model.forward(texts[start:start + provider._batch_size])
        return texts

    class Interrupted(Exception):
        pass

    def crash(stage, **state):
        assert stage == EMBEDDING_CRASH_STAGE
        (tmp_path / "component-state.json").write_text(json.dumps(state))
        raise Interrupted

    monkeypatch.setattr(HuggingFaceEmbedding, "embed_texts", embed)
    provider = object.__new__(HuggingFaceEmbedding)
    provider._batch_size = 2
    provider._model = SimpleNamespace(forward=forward)
    args = SimpleNamespace(crash_stage=EMBEDDING_CRASH_STAGE, embedding_after_batches=2,
                           runtime=tmp_path)
    texts = list("abcde")
    with observe_build_embedding(args, crash):
        # A shorter file finishes without contributing to the next file's count.
        assert await provider.embed_texts(list("wxyz")) == list("wxyz")
        with pytest.raises(Interrupted):
            await provider.embed_texts(texts)
    assert forwarded == [list("wx"), list("yz"), list("ab"), list("cd")]
    state = json.loads((tmp_path / "component-state.json").read_text())["embedding_interruption"]
    assert (state["batches_completed"], state["chunks_completed"], state["chunks_pending"]) == (2, 4, 1)

    args.crash_stage = "none"
    with observe_build_embedding(args, crash) as metrics:
        assert await provider.embed_texts(texts) == texts
    assert metrics["interrupted_embedding_request_replays"] == 1
    assert metrics["interrupted_embedding_chunks_recomputed"] == 4


def test_query_experiment_pairs_policies_and_labels_fixture_results(tmp_path):
    output = tmp_path / "query"
    run_cli("query", "--fixture", "--output", str(output),
            "--failure-probabilities", "0", "1", "--repeats", "1")
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["result_kind"] == "fixture_smoke" and manifest["code_hash"]
    groups = {(g["policy"], g["failure_probability"]): g for g in summary["groups"]}
    assert groups["none", 1]["completion_rate"] == 0
    assert groups["retry", 1]["completion_rate"] == 0
    assert groups["fallback", 1]["source_hit_rate"] == 1
    assert groups["fallback", 1]["exact_answer_match_rate"] is None
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert len({r["workspace_revision"] for r in rows}) == 1
    degraded = [r for r in rows if r["policy"] == "fallback" and r["failure_probability"] == 1]
    assert all(r["result"]["retrieval"]["degraded"] for r in degraded)


def test_process_exit_recovery_experiment_matches_cold_content(tmp_path):
    output = tmp_path / "build"
    run_cli("build", "--fixture", "--after", "2", "--output", str(output))
    result = json.loads((output / "summary.json").read_text())
    assert result["fixture"] and result["logical_content_equal"]
    assert result["resume_metrics"]["sources_reused"] == 2
    assert result["cold_metrics"]["sources_reused"] == 0
    assert result["resume_metrics"]["graph_extraction_calls"] < result["cold_metrics"]["graph_extraction_calls"]
    assert result["resume_metrics"]["materialization_ms"] > 0


@pytest.mark.parametrize(("stage", "flags", "reused", "rebuilt"), [
    ("partial", (), ["kb"], ["db", "graph"]),
    ("partial", (), ["db"], ["graph", "kb"]),
    ("write-unconfirmed", ("--crash-component", "graph"), ["db", "kb"], ["graph"]),
    ("publication", ("--damage-component", "db", "--damage-checkpoint"), ["graph", "kb"], ["db"]),
    ("publication", ("--damage-component", "kb"), ["db", "graph"], ["kb"]),
    ("publication", ("--damage-component", "graph"), ["db", "kb"], ["graph"]),
    ("publication", (), ["db", "graph", "kb"], []),
])
def test_component_recovery_reuses_complete_artifacts(tmp_path, stage, flags, reused, rebuilt):
    source = tmp_path / "source"
    source.mkdir()
    (source / "article.md").write_text("nebula project budget is 12 units")
    table_file = "aa-orders.csv" if stage == "partial" and reused == ["db"] else "orders.csv"
    (source / table_file).write_text("item,amount\nA,12\n")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"task_id":"rag","question":"nebula","expected_sources":["article.md"]}\n'
        '{"task_id":"rows","sql":"SELECT COUNT(*) FROM \\"{table}\\"","expected_rows":[[1]]}\n'
        '{"task_id":"edge","entity":"Fixture"}\n'
    )
    output = tmp_path / "build"
    run_cli("build", "--fixture", "--offline", "--source", str(source), "--questions", str(questions),
            "--crash-stage", stage, "--output", str(output), *flags)
    result = json.loads((output / "summary.json").read_text())
    assert result["logical_content_equal"]
    assert result["external_api_calls"] == 0 and result["partial_revision_visible"] is False
    metrics = result["component_metrics"]
    assert metrics["components_reused"] == reused
    assert metrics["components_rebuilt"] == rebuilt
    assert set(metrics["component_writes"]) == set(rebuilt)
    assert metrics["embedding_calls"] == ("kb" in rebuilt)
    assert metrics["parsed_document_files"] == bool({"kb", "graph"}.intersection(rebuilt))
    assert metrics["parsed_table_files"] == ("db" in rebuilt)
    assert result["component_probes"] == result["cold_probes"] == result["resume_probes"]
    if stage == "partial":
        resumed = result["resume_metrics"]
        assert resumed["resumed_snapshot_candidate"]
        assert resumed["workspace_revision"] == result["interrupted"]["revision"]
        assert resumed["resumed_surfaces"] == reused
        assert not set(reused) & resumed["component_writes"].keys()
    if stage == "publication":
        resumed = result["resume_metrics"]
        assert bool(resumed.get("resumed_snapshot_candidate")) == (not flags)
        assert (resumed["workspace_revision"] == result["interrupted"]["revision"]) == (not flags)
        if not flags:
            assert resumed["embedding_calls"] == 0
            assert resumed["materialization_ms"] == 0
    rag, sql, graph = result["component_probes"]
    assert rag["source_hit"] is True and sql["result_match"] is True
    assert graph["outcome"] == "ok" and graph["result"]
    if "--damage-checkpoint" in flags:
        assert result["resume_metrics"]["sources_reused"] == result["resume_metrics"]["sources_computed"] == 1


def test_build_batch_records_unavailable_interruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "orders.csv").write_text("item,amount\nA,12\n")
    units = tmp_path / "units.json"
    units.write_text(json.dumps([{"unit": "orders", "source": "source"}]))
    output = tmp_path / "batch"
    run_cli("build", "--fixture", "--offline", "--build-units", str(units),
            "--crash-stage", "partial", "--output", str(output))
    rows = [json.loads(line) for line in (output / "units.jsonl").read_text().splitlines()]
    assert rows == [{"unit": "orders", "crash_stage": "partial", "surfaces": ["db"],
                     "outcome": "interruption_not_applicable", "logical_content_equal": None}]


@pytest.mark.parametrize("error,retry_attempts", [
    ("sqlite_busy", 2), ("sqlite_locked", 1), ("sqlite_error", 1),
])
def test_db_experiment_executes_sql_and_distinguishes_attempts_from_execution(tmp_path, error, retry_attempts):
    output = tmp_path / "db"
    run_cli("query", "--fixture", "--surfaces", "db", "--execution", "sql",
            "--component", "db", "--db-error", error, "--output", str(output),
            "--failure-probabilities", "0", "1", "--repeats", "1")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["result_kind"] == "fixture_smoke"
    assert manifest["harness"]["db_error"] == error
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert {row["policy"] for row in rows} == {"none", "retry"}
    assert len({row["workspace_revision"] for row in rows}) == 1
    for row in rows:
        assert row["fault_origin"] == "simulated_exception"
        assert row["source_hit"] is None and row["exact_answer_match"] is None
        if row["failure_probability"] == 0:
            assert row["outcome"] == "ok" and row["result_match"] is True
            assert row["component_attempts"] == row["sql_executions"] == 1
            assert row["injected_faults"] == row["recovery"]["retries"] == 0
        else:
            attempts = retry_attempts if row["policy"] == "retry" else 1
            assert row["outcome"] == "error" and row["result_match"] is False
            assert row["component_attempts"] == row["injected_faults"] == attempts
            assert row["sql_executions"] == 0
            assert row["recovery"]["retries"] == attempts - 1
    summary = json.loads((output / "summary.json").read_text())
    for group in summary["groups"]:
        expected = 1 if group["failure_probability"] == 0 else 0
        assert group["completion_rate"] == group["result_match_rate"] == expected


def test_db_experiment_scores_explicit_sql_tasks_without_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id":"q","sql":"SELECT 42"}\n')
    output = tmp_path / "db"
    run_cli("query", "--fixture", "--surfaces", "db", "--execution", "sql",
            "--component", "db", "--questions", str(tasks), "--output", str(output),
            "--failure-probabilities", "0", "--repeats", "1")
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["result"]["rows"] == [[42]] and row["result_match"] is None for row in rows)


@pytest.mark.parametrize("extra", [
    ("--component", "db"),
    ("--execution", "sql"),
    ("--component", "db", "--execution", "sql", "--surfaces", "kb"),
    ("--outage-ms", "nan"),
])
def test_invalid_db_experiment_options_fail_before_build(tmp_path, extra):
    output = tmp_path / "db"
    result = subprocess.run([sys.executable, "-m", "benchmark.robustness", "query", "--fixture",
                             "--output", str(output), *extra], cwd=REPO, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert not output.exists()
