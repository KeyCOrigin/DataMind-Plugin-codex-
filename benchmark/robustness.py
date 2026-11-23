"""Reproducible local fault experiments (fixtures are never quality results).

    python -m benchmark.robustness query --fixture --output /tmp/query-faults
    python -m benchmark.robustness build --fixture --after 2 --output /tmp/build-faults

Live mode uses Settings/environment and requires --source and --questions for
queries. JSONL tasks may contain question, task_id, filters, expected_sources,
and reference_answer. Labels are used only by the external scoring code below.
"""
from __future__ import annotations

import argparse
import copy
import ipaddress
import ctypes
import errno
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

from datamind.capabilities.ingest.formats import DOCUMENT_EXTS, TABLE_EXTS
from datamind.capabilities.kb.indexer import build_index_manifest, write_manifest_atomic
from datamind.config import EmbeddingConfig, Settings
from datamind.core.errors import CapabilityError
from datamind.core.context import RequestContext
from datamind.core.logging import bind_context
from datamind.core.recovery import recovery_summary
from datamind.experiment_faults import ExperimentFaultError, db_counts, db_faults
from datamind.experiment_db_errors import DB_FAULTS, build_db_fault
from datamind.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]
BUILD_POLICIES = ("cold", "component", "resume")
EMBEDDING_CRASH_STAGE = "embedding-mid-file"
EMBEDDING_PERSISTED_STAGE = "embedding-batch-persisted"
EMBEDDING_CRASH_STAGES = {EMBEDDING_CRASH_STAGE, EMBEDDING_PERSISTED_STAGE}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def append_row(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def provenance(args, source):
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True).stdout
    return {
        "schema_version": 1, "command": args.command, "fixture": args.fixture,
        "result_kind": "fixture_smoke" if args.fixture else "live_experiment",
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_hash": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        "code_hash": digest([(str(p.relative_to(REPO)), hashlib.sha256(p.read_bytes()).hexdigest())
                             for package in ("datamind", "benchmark")
                             for p in sorted((REPO / package).rglob("*.py"))]),
        "python": platform.python_version(), "seed": getattr(args, "seed", None),
        "source_hash": digest([(str(p.relative_to(source)), hashlib.sha256(p.read_bytes()).hexdigest())
                               for p in sorted(source.rglob("*")) if p.is_file()]),
        "questions_hash": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest() if args.questions else None,
        # Only harness arguments, never Settings secrets/endpoints.
        "harness": vars(args),
    }


def setup_fixture():
    """Deterministic CPU-only providers with real local snapshot storage."""
    from datamind import runtime as options

    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    class Embed:
        name = "robustness-fixture"
        dimension = 8

        async def embed_texts(self, texts):
            return [[float(i == int(hashlib.sha256(t.encode()).hexdigest()[:4], 16) % 8)
                     for i in range(8)] for t in texts]

        async def embed_query(self, text):
            return (await self.embed_texts([text]))[0]

    class Model:
        async def generate_text(self, prompt, **kwargs):
            return '[{"subject":"Fixture","relation":"owns","object":"Demo"}]'

        async def complete(self, **kwargs):
            raise RuntimeError("Fixture mode evaluates retrieval, not LLM answer quality")

        async def aclose(self):
            pass

    options.build_embedding = lambda *a, **k: Embed()
    global _fixture_model
    _fixture_model = Model()



_fixture_model = None

def workspace_for(settings):
    if _fixture_model is None:
        return Workspace(settings)
    async def builder(cfg, enable):
        from datamind.data_plane import build_data_plane
        return await build_data_plane(cfg,enable=enable,text_client=_fixture_model)
    return Workspace(settings,builder=builder)


def prepared_graph(args):
    if args.fixture or 'graph' not in args.surfaces.split(','):
        return None
    if not args.graph_sources:
        raise ValueError('Live graph builds require --graph-sources prepared by the main Codex agent')
    value=json.loads(Path(args.graph_sources).read_text())
    if not isinstance(value,list):
        raise ValueError('--graph-sources must contain a JSON array of versioned source assertions')
    return value


def settings_for(runtime, fixture, args=None):
    offline = getattr(args, "offline", False)
    settings = (Settings(_env_file=None, llm={"api_key": "offline", "model": "offline"})
                if fixture or offline else Settings())
    if offline and not fixture:
        settings.kb.vector_store = "lancedb"
        settings.db.dialect = "dolt" if getattr(args, "native_incremental", False) else "sqlite"
        settings.db.dsn = None
        settings.graph.backend = "networkx"
        if args.embedding_model_dir:
            embedding = EmbeddingConfig(provider="huggingface", model=args.embedding_model_dir)
            if getattr(args, "embedding_batch_size", None) is not None:
                embedding.batch_size = args.embedding_batch_size
            settings.embedding = embedding

    settings.data.base_dir = Path(runtime).resolve()
    settings.workspace.retention = 'all'  # Experiments compare explicit historical revisions.
    settings.retrieval.strategy = "hybrid"
    settings.retrieval.vector_retries = 0  # Reader owns the one shared retry budget.
    settings.retrieval.vector_failure_fallback = "error"
    if offline:
        settings.workspace.auto_gc = False
        settings.retrieval.rerank = False
    if fixture:
        settings.llm.api_key = "fixture"
        settings.llm.model = "fixture"
        settings.llm.fallback_model = None
        settings.embedding.dimension = 8
        settings.embedding.model = "fixture"
        settings.kb.vector_store = "chroma"
        settings.kb.ingest_mode = "direct"
        settings.graph.backend = "networkx"
        settings.db.dialect = "sqlite"
        settings.db.dsn = None
        setup_fixture()
    if args is not None and getattr(args, "deadline_s", None):
        settings.agent.wall_clock_timeout_s = args.deadline_s
    return settings


def source_and_tasks(args, output):
    if args.source:
        source = Path(args.source).resolve()
        if not source.is_dir():
            raise ValueError("--source must be a directory")
        if output == source or output.is_relative_to(source):
            raise ValueError("--output must be outside the source workspace")
    elif args.fixture:
        source = output / "inputs"
        source.mkdir()
        for i, term in enumerate(("nebula", "orchid", "cobalt", "quartz", "maple", "harbor")):
            (source / f"doc{i}.md").write_text(f"{term} project budget is {i + 10} units.\n")
        (source / "orders.csv").write_text("item,amount\nA,12\nB,24\n")
    else:
        raise ValueError("Live experiments require --source")
    if args.questions:
        tasks = load_build_tasks(args.questions) if args.command == "build" else [
            json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
        for i, task in enumerate(tasks):
            field = "sql" if getattr(args, "execution", None) == "sql" else "question"
            if not isinstance(task.get(field), str) and not (
                args.command == "build" and args.offline
                and (isinstance(task.get("sql"), str) or isinstance(task.get("entity"), str))
            ):
                raise ValueError(f"task {i} needs {field}")
            task.setdefault("task_id", str(i))
    elif args.fixture and getattr(args, "execution", None) == "sql":
        tasks = [
            {"task_id": "0", "sql": 'SELECT SUM(CAST(amount AS INTEGER)) FROM "{table}"',
             "expected_rows": [[36]]},
            {"task_id": "1", "sql": 'SELECT COUNT(*) FROM "{table}"', "expected_rows": [[2]]},
        ]
    elif args.fixture:
        tasks = [{"task_id": "0", "question": "nebula", "expected_sources": ["doc0.md"]},
                 {"task_id": "1", "question": "orchid", "expected_sources": ["doc1.md"]}]
    else:
        tasks = []
    return source, tasks


def score(task, result, execution):
    """No LLM judge or heuristic pretending to be official benchmark accuracy."""
    if execution == "sql":
        return {"source_hit": None, "exact_answer_match": None,
                "result_match": result.get("rows") == task["expected_rows"] if "expected_rows" in task else None}
    expected = task.get("expected_sources")
    observed = [r.get("source") for r in result.get("results", [])] if execution == "retrieval" else [
        e.get("locator", {}).get("source") or e.get("source_id") for e in result.get("evidence", [])]
    source_hit = None if not expected else any(
        s and (s in expected or Path(s).name in expected) for s in observed)
    reference = task.get("reference_answer")
    exact = None if execution != "agent" or reference is None else str(result.get("answer", "")).strip() == str(reference).strip()
    return {"source_hit": source_hit, "exact_answer_match": exact}


class Faults:
    """Per-task schedules are paired across policies; retries consume later events."""
    def __init__(self, seed, task, probability, outage_ms, delay_ms):
        self.seed, self.task, self.p = seed, task, probability
        self.started = time.monotonic()
        self.outage = outage_ms / 1000
        self.delay = delay_ms / 1000
        self.attempts, self.injected = 0, 0

    async def before(self):
        attempt = self.attempts
        self.attempts += 1
        rng = random.Random(f"{self.seed}:{self.task}:{attempt}")
        if time.monotonic() - self.started < self.outage or rng.random() < self.p:
            self.injected += 1
            await asyncio.sleep(self.delay)
            raise TimeoutError("injected transient fault")


def install_fault(reader, faults, component):
    retriever = reader._system.services.kb.retriever
    if component == "embedding":
        original = retriever._embed.embed_query
        async def failing(query):
            await faults.before()
            return await original(query)
        retriever._embed.embed_query = failing
    else:
        original = retriever._store.query
        async def failing(*a, **k):
            await faults.before()
            return await original(*a, **k)
        retriever._store.query = failing


async def query_experiment(args, output, source, tasks):
    if not tasks:
        raise ValueError("Query experiments require --questions (or --fixture)")
    if args.fixture and args.execution == "agent":
        raise ValueError("Fixture mode cannot evaluate LLM answers; use retrieval or sql")
    settings = settings_for(output / "runtime", args.fixture, args)
    if args.component == "db":
        backend = {"dolt": "mysql", "postgres": "postgresql"}.get(settings.db.dialect, settings.db.dialect)
        if backend != DB_FAULTS[args.db_error].backend:
            raise ValueError("--db-error must match the configured database backend")
        build_db_fault(args.db_error)  # fail before building if the optional driver is absent
    workspace = workspace_for(settings)
    built = await workspace.update(str(source), surfaces=args.surfaces.split(","), graph_sources=prepared_graph(args))
    manifest = provenance(args, source)
    manifest.update(workspace_revision=built["workspace_revision"], pipeline_hash=built["pipeline_hash"],
                    retrieval=settings.retrieval.model_dump(),
                    model=settings.codex.model, embedding_model=settings.embedding.model,
                    scoring="source-hit / exact string match only; use official scorers for paper results")
    write_json(output / "manifest.json", manifest)
    write_json(output / "build.json", built)
    rows = []
    # Seeded task/policy ordering limits systematic cold/warm and order effects.
    policies = ("none", "retry") if args.component == "db" else ("none", "retry", "fallback")
    jobs = [(repeat, task, policy, p) for repeat in range(args.repeats) for task in tasks
            for policy in policies for p in args.failure_probabilities]
    random.Random(args.seed).shuffle(jobs)
    for repeat, task, policy, probability in jobs:
        workspace.settings.retrieval.vector_failure_fallback = "bm25" if policy == "fallback" else "error"
        reader_started = time.monotonic()
        row = {"task_id": task["task_id"], "repeat": repeat, "policy": policy,
               "failure_probability": probability, "component": args.component,
               "workspace_revision": built["workspace_revision"], "fixture": args.fixture,
               "source_hit": False if task.get("expected_sources") else None,
               "exact_answer_match": False if task.get("reference_answer") is not None and args.execution == "agent" else None}
        if args.component == "db":
            row.update(fault_origin="simulated_exception", db_error=args.db_error,
                       result_match=False if "expected_rows" in task else None)
        try:
            async with workspace.reader(built["workspace_revision"], read_retries=0 if policy == "none" else args.retries) as reader:
                row["reader_setup_ms"] = (time.monotonic() - reader_started) * 1000
                faults = Faults(args.seed + repeat, str(task["task_id"]), probability,
                                args.outage_ms, args.fault_delay_ms)
                runtime_env=None
                state=None
                if args.execution == 'retrieval':
                    install_fault(reader, faults, args.component)
                else:
                    state=output/'fault-states'/(digest([repeat,task['task_id'],policy,probability])+'.sqlite3')
                    plan={'state':str(state),'seed':args.seed+repeat,'task':str(task['task_id']),
                          'component':args.component,'probability':probability,
                          'outage_ms':args.outage_ms,'delay_ms':args.fault_delay_ms}
                    if args.component == 'db':
                        plan['error'] = args.db_error
                    runtime_env={'DATAMIND_TEST_READ_FAULT':json.dumps(plan),
                                 'DATAMIND_WORKSPACE_READ_RETRIES':str(0 if policy=='none' else args.retries)}
                started = time.monotonic()
                context = RequestContext.new()
                context.extra['recovery_events'] = []
                try:
                    with bind_context(context), db_faults(plan) if args.execution == 'sql' else nullcontext():
                        async with asyncio.timeout(args.deadline_s):
                            if args.execution == "sql":
                                sql = task['sql']
                                if '{table}' in sql:
                                    tables = await reader._system.services.db.list_tables()
                                    if len(tables) != 1:
                                        raise ValueError("{table} requires exactly one published table")
                                    sql = sql.replace('{table}', tables[0].replace('"', '""'))
                                result = await reader.call("db_query_sql", sql=sql)
                            elif args.execution == "retrieval":
                                result = await reader.call("kb_search", query=task["question"], filters=task.get("filters"))
                            else:
                                result = await reader.query(task["question"], runtime_env=runtime_env)
                    row.update(outcome="ok", result=result, **score(task, result, args.execution))
                except Exception as exc:
                    row.update(outcome="error", error_type=type(exc).__name__, recovery=getattr(exc, "datamind_recovery", {}), source_hit=False if task.get("expected_sources") else None,
                               exact_answer_match=False if task.get("reference_answer") is not None and args.execution == "agent" else None)
                    if isinstance(exc, ExperimentFaultError):
                        row['outcome'] = 'experiment_error'
                if args.execution == 'sql':
                    row['recovery'] = recovery_summary(context.extra['recovery_events'])
                if state is not None:
                    from datamind.experiment_faults import counts
                    actual = db_counts(state) if args.component == 'db' else counts(state)
                    faults.attempts,faults.injected=actual['attempts'],actual['injected']
                    if args.component == 'db':
                        row['sql_executions'] = actual['executed']
                row.update(latency_ms=(time.monotonic() - started) * 1000,
                           injected_faults=faults.injected, component_attempts=faults.attempts)
        except Exception as exc:
            row.update(outcome="experiment_error" if isinstance(exc, ExperimentFaultError) else "reader_error",
                       error_type=type(exc).__name__)
        rows.append(row)
        append_row(output / "rows.jsonl", row)
    groups = []
    for probability in args.failure_probabilities:
        for policy in policies:
            batch = [r for r in rows if r["policy"] == policy and r["failure_probability"] == probability]
            latencies = sorted(r["latency_ms"] for r in batch if "latency_ms" in r)
            groups.append({"policy": policy, "failure_probability": probability, "tasks": len(batch),
                           "completion_rate": sum(r["outcome"] == "ok" for r in batch) / len(batch),
                           "p95_latency_ms": latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
                           **{key + "_rate": (sum(r.get(key) is True for r in batch) / sum(r.get(key) is not None for r in batch))
                              if any(r.get(key) is not None for r in batch) else None for key in ("source_hit", "exact_answer_match", "result_match")}})
    write_json(output / "summary.json", {"fixture": args.fixture, "groups": groups})


async def snapshot_content(workspace, revision):
    """Logical output fingerprint, excluding physical index IDs and timings."""
    manifest = workspace.manifest(revision)
    contents = {"sources": [(s["uri"], s["version"]) for s in manifest["sources"]]}
    async with workspace.reader(revision) as reader:
        services = reader._system.services
        if services.kb:
            contents["kb"] = sorted(await services.kb.vector_store.get_all_texts(), key=lambda x: x[0])
        if services.graph:
            graph = json.loads(services.graph.store._path.read_text())
            contents["graph"] = {k: sorted(v, key=lambda x: json.dumps(x, sort_keys=True)) for k, v in graph.items()}
        if services.db:
            # Local SQLite workspace API: read all rows, not a row-limit sample.
            import sqlite3
            with sqlite3.connect(services.db.engine.url.database) as conn:
                contents["db"] = {}
                for table, in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                    quoted = '"' + table.replace('"', '""') + '"'
                    contents["db"][table] = sorted(conn.execute(f"SELECT * FROM {quoted}").fetchall(), key=str)
    return digest(contents)


def deny_network():
    """Fail closed on Linux and inherit the socket restriction in child workers."""
    class Compare(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]

    lib = ctypes.CDLL("libseccomp.so.2")
    signature = {
        "seccomp_init": ([ctypes.c_uint32], ctypes.c_void_p),
        "seccomp_syscall_resolve_name": ([ctypes.c_char_p], ctypes.c_int),
        "seccomp_rule_add_array": (
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Compare)], ctypes.c_int),
        "seccomp_load": ([ctypes.c_void_p], ctypes.c_int),
        "seccomp_release": ([ctypes.c_void_p], None),
    }
    for name, (argtypes, restype) in signature.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = argtypes, restype
    context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("could not initialize the offline socket filter")
    try:
        for family in (socket.AF_INET, socket.AF_INET6):
            rule = Compare(0, 4, family, 0)  # SCMP_CMP_EQ
            if lib.seccomp_rule_add_array(
                context, 0x50000 | errno.EPERM, lib.seccomp_syscall_resolve_name(b"socket"),
                1, ctypes.byref(rule),
            ):
                raise RuntimeError("could not configure the offline socket filter")
        if lib.seccomp_load(context):
            raise RuntimeError("could not load the offline socket filter")
    finally:
        lib.seccomp_release(context)

    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family):
                raise RuntimeError("offline socket filter did not block network access")
        except PermissionError as exc:
            if exc.errno != errno.EPERM:
                raise
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DATAMIND_MINERU="off",
                      ANONYMIZED_TELEMETRY="False")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def load_build_tasks(path):
    raw = Path(path).read_text()
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


async def build_probes(workspace, revision, args):
    if not args.questions:
        return []
    probes = []
    async with workspace.reader(revision, read_retries=0) as reader:
        for index, task in enumerate(load_build_tasks(args.questions)):
            row = {"task_id": task.get("task_id", task.get("id", str(index)))}
            if args.offline:
                surface = "db" if "sql" in task else "graph" if "entity" in task else "kb"
                if surface not in reader.manifest["surfaces"]:
                    probes.append({**row, "outcome": "not_applicable", "missing_surface": surface})
                    continue
            try:
                if args.offline and "sql" in task:
                    sql = task["sql"]
                    if "{table}" in sql:
                        tables = await reader._system.services.db.list_tables()
                        table, = tables
                        sql = sql.replace("{table}", table.replace('"', '""'))
                    response = await reader.call("db_query_sql", sql=sql)
                    row.update(result=response.get("rows"), **score(task, response, "sql"))
                elif args.offline and "entity" in task:
                    response = await reader.call("graph_neighbors", entity=task["entity"])
                    row["result"] = response.get("edges", [])
                elif args.offline or args.fixture:
                    response = await reader.call("kb_search", query=task["question"], filters=task.get("filters"))
                    row.update(result=[hit.get("source") for hit in response.get("results", [])],
                               **score(task, response, "retrieval"))
                else:
                    response = await reader.query(task["question"])
                    row.update(result=response, **score(task, response, "agent"))
                row["outcome"] = "ok"
            except Exception as exc:
                row.update(outcome="error", error_type=type(exc).__name__)
            probes.append(row)
    return probes


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def component_paths(settings, component):
    storage = settings.data.storage_dir
    if component == "kb":
        return [storage / settings.kb.vector_store, storage / "kb_index_manifest.json"]
    if component == "db":
        return [storage / name for name in ("demo.db", "demo.db-wal", "demo.db-shm", "demo.db-journal")]
    return [storage / "graph.json"]


def component_files(settings, component):
    root = settings.data.base_dir
    files = {}
    for path in component_paths(settings, component):
        for entry in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if entry.is_file():
                files[entry.relative_to(root).as_posix()] = file_hash(entry)
    return files


class BuildEmbeddingTokens:
    def __init__(self, result):
        self.path = Path(result).with_suffix(".embedding-tokens.json")
        self.input_tokens = 0
        self.active = True

    def save(self):
        write_json(self.path, {"input_tokens": self.input_tokens})

    @contextmanager
    def observe(self):
        from unittest.mock import patch
        from datamind.capabilities.embedding.providers.huggingface import HuggingFaceEmbedding

        original = HuggingFaceEmbedding.embed_texts

        async def embed(provider, texts):
            if not self.active:
                return await original(provider, texts)
            forward = provider._model.forward

            def counted(features, *args, **kwargs):
                self.input_tokens += int(features["attention_mask"].sum().item())
                return forward(features, *args, **kwargs)

            with patch.object(provider._model, "forward", counted):
                return await original(provider, texts)

        with patch.object(HuggingFaceEmbedding, "embed_texts", embed):
            yield


def save_build_embedding_tokens(args, *, stop=False):
    counter = getattr(args, "_embedding_token_counter", None)
    if counter is not None:
        if stop:
            counter.active = False
        counter.save()


class RandomEmbeddingCut:
    def __init__(self, args):
        self.path = Path(args.worker_result).with_suffix(".embedding-draws.json")
        self.seed = args.embedding_fault_seed
        self.probability = args.embedding_fault_probability
        self.rng = random.Random(f"{self.seed}:{args.source}")
        self.eligible_batches_examined = 0
        self.trigger_draw = None

    def choose(self):
        self.eligible_batches_examined += 1
        draw = self.rng.random()
        if draw < self.probability:
            self.trigger_draw = draw
            return True
        return False

    def save(self):
        write_json(self.path, {
            "probability_per_eligible_batch": self.probability,
            "seed": self.seed,
            "eligible_batches_examined": self.eligible_batches_examined,
            "trigger_draw": self.trigger_draw,
        })


def choose_embedding_cut(args, batches):
    cut = getattr(args, "_random_embedding_cut", None)
    return cut.choose() if cut is not None else batches == args.embedding_after_batches


def midpoint_batch(total, batch_size):
    """Nearest full batch to half the requested chunks, with work left."""
    return min((total - 1) // batch_size, max(1, (total + batch_size) // (2 * batch_size)))


@contextmanager
def observe_build_embedding(args, crash, source_uri=None):
    """Observe local model batches inside a single workspace embedding call."""
    interrupt = args.crash_stage == EMBEDDING_CRASH_STAGE
    interrupted = None
    if args.crash_stage == "none":
        state_path = Path(args.runtime) / "component-state.json"
        if state_path.exists():
            interrupted = json.loads(state_path.read_text()).get("embedding_interruption")
    metrics = {}
    if not interrupt and interrupted is None:
        yield metrics
        return

    from unittest.mock import patch
    from datamind.capabilities.embedding.providers.huggingface import HuggingFaceEmbedding

    original = HuggingFaceEmbedding.embed_texts
    target = getattr(args, "embedding_midpoint_file", None)
    counter = getattr(args, "_embedding_token_counter", None)
    if interrupted is not None:
        metrics["interrupted_embedding_request_replays"] = 0
        metrics["interrupted_embedding_chunks_recomputed"] = None
        if target:
            metrics["interrupted_embedding_tokens_recomputed"] = None

    async def embed(provider, texts):
        if target and source_uri() != target:
            return await original(provider, texts)
        if interrupted is not None:
            vectors = await original(provider, texts)
            if digest(list(texts)) == interrupted["request_hash"]:
                metrics["interrupted_embedding_request_replays"] += 1
                metrics["interrupted_embedding_chunks_recomputed"] = interrupted["chunks_completed"]
                if target:
                    metrics["interrupted_embedding_tokens_recomputed"] = interrupted["input_tokens_completed"]
            return vectors

        total = len(texts)
        minimum_batches = 1 if target or getattr(args, "_random_embedding_cut", None) else args.embedding_after_batches
        if total <= minimum_batches * provider._batch_size:
            return await original(provider, texts)

        request_hash = digest(list(texts))
        forward = provider._model.forward
        batches = completed = 0
        start_tokens = counter.input_tokens if target else None
        cut_batch = midpoint_batch(total, provider._batch_size) if target else None

        def observe_forward(*a, **kw):
            nonlocal batches, completed
            output = forward(*a, **kw)
            embeddings = output["sentence_embedding"]
            batches += 1
            completed += len(embeddings)
            if completed < total and (batches == cut_batch if target else choose_embedding_cut(args, batches)):
                # Ensure accelerator work has completed before terminating the process.
                embeddings.cpu()
                crash(EMBEDDING_CRASH_STAGE, embedding_interruption={
                    "request_hash": request_hash,
                    "batch_size": provider._batch_size,
                    "batches_completed": batches,
                    "chunks_requested": total,
                    "chunks_completed": completed,
                    "chunks_pending": total - completed,
                    **({"target_file": target, "chunks_persisted": 0,
                        "input_tokens_completed": counter.input_tokens - start_tokens,
                        "input_tokens_persisted": 0} if target else {}),
                })
            return output

        with patch.object(provider._model, "forward", observe_forward):
            return await original(provider, texts)

    with patch.object(HuggingFaceEmbedding, "embed_texts", embed):
        yield metrics


@contextmanager
def observe_incremental_embedding(args, crash):
    """Track one file's batches and compare exact vector keys on restart."""
    stage = args.crash_stage
    state_path = Path(args.runtime) / "component-state.json"
    interrupted = (
        json.loads(state_path.read_text()).get("embedding_interruption")
        if stage == "none" and state_path.exists() else None
    )
    metrics = {}
    if stage not in EMBEDDING_CRASH_STAGES and interrupted is None:
        yield metrics
        return
    if interrupted and "computed_vector_refs" not in interrupted:
        # Older interrupted workers recorded only a request hash.
        with observe_build_embedding(args, crash) as legacy_metrics:
            yield legacy_metrics
        return

    from unittest.mock import patch
    from datamind import workspace_incremental as incremental

    original_embed = incremental.embed_missing_vectors
    original_write = incremental.persist_vector_batch
    target = getattr(args, "embedding_midpoint_file", None)
    counter = getattr(args, "_embedding_token_counter", None)
    target_blob = file_hash(Path(target)) + Path(target).suffix.lower() if target and not interrupted else None
    current_is_target = False
    current_cut = None
    start_tokens = persisted_tokens = 0
    completed_refs = []
    batch_count = total = batch_size = 0
    previous_refs = set(interrupted["computed_vector_refs"]) if interrupted else set()
    recomputed = set()
    if interrupted:
        metrics["interrupted_embedding_chunks_recomputed"] = 0
        if target:
            metrics["interrupted_embedding_tokens_recomputed"] = None

    extract = incremental.extract_document

    def observe_extract(path):
        nonlocal current_is_target
        current_is_target = Path(path).name == target_blob
        return extract(path)

    async def embed(root, missing, embedding, size, build_metrics):
        nonlocal total, batch_size, batch_count, current_cut, start_tokens, persisted_tokens
        total, batch_size, batch_count = len(missing), size, 0
        completed_refs.clear()
        if target and not interrupted:
            current_cut = midpoint_batch(total, size) if current_is_target else None
            start_tokens, persisted_tokens = counter.input_tokens, 0
        await original_embed(root, missing, embedding, size, build_metrics)

    def write(root, batch, vectors, dimension):
        nonlocal batch_count, persisted_tokens
        if interrupted:
            original_write(root, batch, vectors, dimension)
            recomputed.update(previous_refs.intersection(ref for _, ref in batch))
            metrics["interrupted_embedding_chunks_recomputed"] = len(recomputed)
            if target:
                metrics["interrupted_embedding_tokens_recomputed"] = (
                    interrupted["input_tokens_completed"] - interrupted["input_tokens_persisted"]
                    if recomputed == set(interrupted["unpersisted_vector_refs"]) else None
                )
            return

        if target and current_cut is None:
            original_write(root, batch, vectors, dimension)
            return
        prior_count = len(completed_refs)
        completed_refs.extend(ref for _, ref in batch)
        batch_count += 1
        persisted = stage == EMBEDDING_PERSISTED_STAGE
        if persisted:
            original_write(root, batch, vectors, dimension)
        if len(completed_refs) < total and (batch_count == current_cut if target else choose_embedding_cut(args, batch_count)):
            crash(stage, embedding_interruption={
                "batch_size": batch_size,
                "batches_completed": batch_count,
                "chunks_requested": total,
                "chunks_completed": len(completed_refs),
                "chunks_persisted": len(completed_refs) if persisted else prior_count,
                "chunks_pending": total - len(completed_refs),
                "computed_vector_refs": completed_refs,
                **({"target_file": target, "input_tokens_completed": counter.input_tokens - start_tokens,
                    "input_tokens_persisted": persisted_tokens,
                    "unpersisted_vector_refs": completed_refs[prior_count:]} if target else {}),
            })
        if not persisted:
            original_write(root, batch, vectors, dimension)
        if target:
            persisted_tokens = counter.input_tokens - start_tokens

    with ExitStack() as stack:
        stack.enter_context(patch.object(incremental, "embed_missing_vectors", embed))
        stack.enter_context(patch.object(incremental, "persist_vector_batch", write))
        if target_blob:
            stack.enter_context(patch.object(incremental, "extract_document", observe_extract))
        yield metrics


class BuildCapture:
    """Observe the native snapshot build without changing its write order."""

    def __init__(self, args, workspace, atomic):
        self.args = args
        self.workspace = workspace
        self.atomic = atomic
        self.surfaces = set(args.surfaces.split(","))
        self.inventory = workspace._inventory(Path(args.source))
        self.last = {
            surface: max((i for i, item in enumerate(self.inventory)
                          if Path(item["uri"]).suffix.lower() in extensions), default=-1)
            for surface, extensions in (("kb", DOCUMENT_EXTS), ("db", TABLE_EXTS))
            if surface in self.surfaces
        }
        self.source_index = 0
        self.source_uri = None
        self.checkpoints = 0
        self.completed = set()
        self.truncated_sources = []
        self.writes = {}

    def finish_sources(self):
        self.completed.update(surface for surface, last in self.last.items() if last < self.source_index)
        if self.args.crash_stage == "partial" and self.completed and self.completed != self.surfaces:
            self.crash("partial")

    def crash(self, stage, **details):
        receipts = {}
        if "kb" in self.completed:
            self.bindings["kb"]["version"] = getattr(self.kb_store, "latest_version", None)
        for surface in sorted(self.completed):
            files = component_files(self.settings, surface)
            if files:
                receipts[surface] = {
                    "files": files,
                    "binding": self.bindings[surface],
                }
        self.atomic(Path(self.args.runtime) / "component-state.json", {
            "revision": self.settings.data.base_dir.name,
            "stage": stage,
            "receipts": receipts,
            "sources": self.inventory,
            "complete": sorted(self.completed),
            "truncated_sources": self.truncated_sources,
            "writes": self.writes,
            **details,
        })
        save_build_embedding_tokens(self.args)
        if cut := getattr(self.args, "_random_embedding_cut", None):
            cut.save()
        os._exit(23)

    def observe(self):
        original_load = self.workspace._load_checkpoint
        original_system = self.workspace._system

        def load(path, *, source_uri, **kwargs):
            self.finish_sources()
            self.source_index += 1
            self.source_uri = source_uri
            value = original_load(path, source_uri=source_uri, **kwargs)
            if value is not None and value.truncated:
                self.truncated_sources.append(source_uri)
            return value

        def observe_write(owner, attribute, surface, size):
            original = getattr(owner, attribute)

            async def call(*args, **kwargs):
                result = await original(*args, **kwargs)
                entry = self.writes.setdefault(surface, {"calls": 0, "items": 0})
                entry["calls"] += 1
                entry["items"] += size(args, kwargs)
                if self.args.crash_stage == "write-unconfirmed" and self.args.crash_component == surface:
                    self.crash("write-unconfirmed:" + surface)
                return result

            setattr(owner, attribute, call)

        async def system(settings, surfaces, **kwargs):
            plane = await original_system(settings, surfaces, **kwargs)
            self.settings = settings
            self.bindings = {}
            if plane.services.kb:
                self.bindings["kb"] = {
                    "provider": settings.kb.vector_store,
                    "path": settings.data.storage_dir.relative_to(settings.data.base_dir).as_posix(),
                    "dimension": plane.services.kb.embedding.dimension,
                }
                self.kb_store = plane.services.kb.vector_store
                observe_write(self.kb_store, "add", "kb", lambda a, k: len(k["ids"]))
            if plane.services.db:
                path = Path(plane.services.db.engine.url.database)
                self.bindings["db"] = {"provider": "sqlite", "path": path.relative_to(settings.data.base_dir).as_posix()}
                observe_write(plane.services.ingest, "_ingest_rows", "db", lambda a, k: len(k["rows"]))
            if plane.services.graph:
                store = plane.services.graph.store
                upsert = store.upsert_triples

                async def graph_write(triples):
                    self.finish_sources()
                    result = await upsert(triples)
                    self.graph_items = len(triples)
                    return result

                store.upsert_triples = graph_write
                observe_write(store, "persist", "graph", lambda a, k: self.graph_items)
            return plane

        self.workspace._load_checkpoint = load
        self.workspace._system = system

    def write_json(self, path, value, **kwargs):
        self.atomic(path, value, **kwargs)
        if path.parent.name == "checkpoints":
            self.checkpoints += 1
            if value["truncated"]:
                self.truncated_sources.append(self.inventory[self.source_index - 1]["uri"])
            if self.args.crash_stage == "checkpoint" and self.checkpoints == self.args.after:
                self.crash("checkpoint")
        if path.name == "manifest.json" and path.parent == self.settings.data.base_dir:
            self.finish_sources()
            self.completed = set(self.surfaces)
            if self.args.crash_stage == "publication":
                for surface, binding in value["snapshots"].items():
                    self.bindings.setdefault(surface, {}).update(binding)
                self.crash("publication")


def copy_component(workspace, previous, target, surface, receipt):
    old = workspace._settings_for(previous)
    cfg = workspace._settings_for(target)
    copied_bytes = 0
    for src in component_paths(old, surface):
        if not src.exists():
            continue
        dst = cfg.data.base_dir / src.relative_to(old.data.base_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
            copied_bytes += sum(path.stat().st_size for path in dst.rglob("*") if path.is_file())
        else:
            shutil.copy2(src, dst)
            copied_bytes += dst.stat().st_size
    if surface == "kb" and not (cfg.data.storage_dir / "kb_index_manifest.json").exists():
        # The vector store is complete even if the interrupted build had not yet
        # generated its compatibility metadata over the final raw snapshot.
        manifest = build_index_manifest(
            data_dir=cfg.data.data_dir, embedding_provider=cfg.embedding.provider,
            embedding_model=cfg.embedding.model, dimension=receipt["binding"]["dimension"],
            chunk_size=cfg.retrieval.chunk_size, chunk_overlap=cfg.retrieval.chunk_overlap,
        )
        write_manifest_atomic(cfg.data.storage_dir / "kb_index_manifest.json", manifest)
    return copied_bytes


def publish_completed_candidate(workspace, state, source):
    """Publish a validated pre-publication candidate in its private experiment clone."""
    revision = state["revision"]
    path = workspace._settings_for(revision).data.base_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    current_inputs = [(item["uri"], item["version"]) for item in workspace._inventory(source)]
    if current_inputs != [(item["uri"], item["version"]) for item in manifest["sources"]]:
        raise RuntimeError("workspace inputs changed after the interruption")
    from datamind.workspace import _atomic_json

    # The component baseline discards the source-level cache before recovery.
    for item in manifest["sources"]:
        item.pop("checkpoint", None)
    _atomic_json(path, manifest)
    with workspace._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        head = db.execute("SELECT revision, epoch FROM head WHERE singleton=1").fetchone()
        if head != (manifest["parent_revision"], manifest["parent_epoch"]):
            raise RuntimeError("workspace head changed after the interruption")
        changed = db.execute("UPDATE revisions SET state='PUBLISHED' WHERE id=? AND state='BUILDING'", (revision,))
        if changed.rowcount != 1:
            raise RuntimeError("interrupted revision is not eligible for publication")
        db.execute("UPDATE head SET revision=?, epoch=epoch+1 WHERE singleton=1", (revision,))
    return manifest


def local_dolt_only():
    """Keep local Dolt TCP usable while rejecting non-loopback Python sockets."""
    original_socket = socket.socket

    class LoopbackSocket(original_socket):
        def _check(self, address):
            if self.family not in (socket.AF_INET, socket.AF_INET6):
                return
            try:
                local = isinstance(address, tuple) and ipaddress.ip_address(address[0]).is_loopback
            except ValueError:
                local = False
            if not local:
                raise PermissionError(errno.EPERM, "only numeric loopback addresses are allowed")

        def connect(self, address):
            self._check(address)
            return super().connect(address)

        def connect_ex(self, address):
            self._check(address)
            return super().connect_ex(address)

        def bind(self, address):
            self._check(address)
            return super().bind(address)

        def sendto(self, data, *args):
            self._check(args[-1])
            return super().sendto(data, *args)

    socket.socket = LoopbackSocket
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DATAMIND_MINERU="off",
                      ANONYMIZED_TELEMETRY="False", DOLT_DISABLE_EVENT_FLUSH="true")
    if not os.environ.get("DOLT_ROOT_PATH"):
        raise ValueError("use a local DOLT_ROOT_PATH with version checks disabled")
    config_only = subprocess.run(
        ["dolt", "config", "--global", "--get", "versioncheck.disabled"],
        capture_output=True, text=True, check=True,
    )
    if config_only.stdout.strip().lower() != "true":
        raise ValueError("incremental offline runs require Dolt versioncheck.disabled=true")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


async def native_readback(workspace, manifest):
    from datamind.workspace_graph import graph_artifacts_available
    from datamind.workspace_native import DoltRepository

    snapshots = manifest["snapshots"]
    checked = {}
    if "kb" in snapshots:
        from datamind.workspace_incremental import kb_version_available

        if not await asyncio.to_thread(kb_version_available, workspace.root, snapshots["kb"]):
            raise RuntimeError("published LanceDB binding is unavailable")
        checked["kb_chunks"] = len(snapshots["kb"]["chunks"])
    if "db" in snapshots:
        binding = snapshots["db"]
        async with DoltRepository(workspace).session(binding["version"]) as (dialect, engine):
            counts = {}
            for table in sorted(binding["tables"]):
                quoted = "`" + table.replace("`", "``") + "`"
                rows = await asyncio.to_thread(dialect._run_sql, engine, f"SELECT COUNT(*) FROM {quoted}")
                counts[table] = rows[0][0]
            checked["db_table_counts"] = counts
    if "graph" in snapshots:
        if not await asyncio.to_thread(graph_artifacts_available, workspace.root, manifest["sources"]):
            raise RuntimeError("published KG artifacts are unavailable")
        checked["graph_artifacts"] = len({
            item["graph_artifact"] for item in manifest["sources"] if item.get("graph_artifact")
        })
    return checked


async def native_incremental_worker(args, workspace, graph, surfaces, started):
    from unittest.mock import patch
    from datamind import runtime, workspace as module, workspace_graph
    from datamind.workspace_native import DoltRepository
    from datamind.capabilities.kb.providers.lancedb_store import LanceDBVectorStore

    atomic = module._atomic_json
    original_graph = workspace_graph.put_graph
    original_db = DoltRepository.build
    original_kb = LanceDBVectorStore.add
    original_embedding = runtime.build_embedding
    embedding_initializations = 0

    def build_embedding(*a, **kw):
        nonlocal embedding_initializations
        embedding_initializations += 1
        return original_embedding(*a, **kw)

    def candidate():
        with workspace._connect() as conn:
            revision, receipt = conn.execute("SELECT revision, receipt FROM incremental_progress").fetchone()
        return revision, json.loads(receipt)

    def crash(stage, **details):
        revision, receipt = candidate()
        atomic(Path(args.runtime) / "component-state.json", {
            "revision": revision, "stage": stage, "bindings": receipt["snapshots"],
            **details,
        })
        save_build_embedding_tokens(args)
        if cut := getattr(args, "_random_embedding_cut", None):
            cut.save()
        os._exit(23)

    def graph_write(*a, **kw):
        if args.crash_stage == "partial":
            _, receipt = candidate()
            if receipt["snapshots"] and set(receipt["snapshots"]) < surfaces:
                crash("partial")
        result = original_graph(*a, **kw)
        if args.crash_stage == "write-unconfirmed" and args.crash_component == "graph":
            crash("write-unconfirmed:graph")
        return result

    async def db_write(*a, **kw):
        if args.crash_stage == "partial" and "graph" not in surfaces:
            _, receipt = candidate()
            if "kb" in receipt["snapshots"]:
                crash("partial")
        result = await original_db(*a, **kw)
        if args.crash_stage == "write-unconfirmed" and args.crash_component == "db":
            crash("write-unconfirmed:db")
        return result

    async def kb_write(*a, **kw):
        result = await original_kb(*a, **kw)
        if args.crash_stage == "write-unconfirmed" and args.crash_component == "kb":
            crash("write-unconfirmed:kb")
        return result

    def manifest_write(path, value, **kw):
        atomic(path, value, **kw)
        if args.crash_stage == "publication" and path.name == "manifest.json":
            crash("publication")

    with ExitStack() as stack:
        embedding_metrics = stack.enter_context(observe_incremental_embedding(args, crash))
        stack.enter_context(patch.object(runtime, "build_embedding", build_embedding))
        if args.crash_stage != "none":
            stack.enter_context(patch.object(module, "_atomic_json", manifest_write))
            stack.enter_context(patch.object(workspace_graph, "put_graph", graph_write))
            stack.enter_context(patch.object(DoltRepository, "build", db_write))
            stack.enter_context(patch.object(LanceDBVectorStore, "add", kb_write))
        result = await workspace.update(args.source, surfaces=sorted(surfaces), graph_sources=graph)
    elapsed = (time.monotonic() - started) * 1000
    if args.crash_stage != "none":
        return  # This source has no materialization event at the requested cut.
    metrics = result["metrics"]
    metrics.update(embedding_metrics)
    reused = metrics["resumed_surfaces"]
    metrics.update(recovery_wall_ms=elapsed, components_reused=reused,
                   components_rebuilt=sorted(surfaces - set(reused)), component_copy_bytes=0,
                   embedding_initializations=embedding_initializations,
                   component_writes={
                       "kb_rows_upserted": metrics["kb_rows_upserted"],
                       "kb_rows_deleted": metrics["kb_rows_deleted"],
                       "db_tables_changed": metrics["db_tables_changed"],
                   })
    state = json.loads((Path(args.runtime) / "component-state.json").read_text())
    if result["workspace_revision"] != state["revision"] or set(reused) != set(state["bindings"]):
        raise RuntimeError("incremental recovery did not reuse the expected candidate and bindings")
    if any(result["snapshots"][surface] != binding for surface, binding in state["bindings"].items()):
        raise RuntimeError("a confirmed storage version changed during recovery")
    save_build_embedding_tokens(args, stop=True)
    write_json(Path(args.worker_result), {
        "build": result, "readback": await native_readback(workspace, result),
        "probes": await build_probes(workspace, result["workspace_revision"], args),
    })


async def build_worker(args):
    from unittest.mock import patch
    import datamind.workspace as module

    recovery_started = time.monotonic()
    settings = settings_for(args.runtime, args.fixture, args)
    settings.workspace.storage_mode = "incremental" if args.native_incremental else "snapshot"
    workspace = workspace_for(settings)
    surfaces = set(args.surfaces.split(","))
    graph = prepared_graph(args)
    if args.native_incremental:
        await native_incremental_worker(args, workspace, graph, surfaces, recovery_started)
        return
    capture = BuildCapture(args, workspace, module._atomic_json)
    capture.observe()
    reused = {}
    state = None
    if args.component_resume:
        state = json.loads((Path(args.runtime) / "component-state.json").read_text())
        if state["sources"] != capture.inventory:
            raise RuntimeError("workspace inputs changed after the interruption")
        previous = workspace._settings_for(state["revision"])
        reused = {surface: receipt for surface, receipt in state["receipts"].items()
                  if surface in surfaces and component_files(previous, surface) == receipt["files"]}
    rebuild = surfaces - reused.keys()
    component_copy_bytes = 0

    def commit_manifest(path, value, **kwargs):
        nonlocal component_copy_bytes
        if path.name == "manifest.json" and path.parent == capture.settings.data.base_dir and reused:
            for surface, receipt in reused.items():
                component_copy_bytes += copy_component(
                    workspace, state["revision"], value["workspace_revision"], surface, receipt)
                value["snapshots"][surface] = receipt["binding"]
                if surface == "graph":
                    value["truncated_sources"] = sorted(set(value["truncated_sources"]) | set(state["truncated_sources"]))
            value["surfaces"] = sorted(surfaces)
        capture.write_json(path, value, **kwargs)

    document_parse_calls = 0
    embedding_metrics = {}
    original_parse = module.extract_document

    def parse(path):
        nonlocal document_parse_calls
        if args.component_resume and "kb" not in rebuild and ("graph" not in rebuild or graph is not None):
            return SimpleNamespace(text="")
        document_parse_calls += 1
        return original_parse(path)

    if rebuild:
        with ExitStack() as scope:
            embedding_metrics = scope.enter_context(observe_build_embedding(args, capture.crash, lambda: capture.source_uri))
            scope.enter_context(patch.object(module, "_atomic_json", commit_manifest))
            scope.enter_context(patch.object(module, "extract_document", parse))
            result = await workspace.update(args.source, surfaces=sorted(rebuild),
                                            graph_sources=graph if "graph" in rebuild else None)
    else:
        result = publish_completed_candidate(workspace, state, Path(args.source))
        result["metrics"] = {key: 0 for key in ("sources_reused", "sources_computed", "embedding_calls",
                                               "embedding_texts", "graph_extraction_calls", "parsed_table_files")}
    result["metrics"].update(
        recovery_wall_ms=(time.monotonic() - recovery_started) * 1000,
        components_reused=sorted(set(reused) | set(result["metrics"].get("resumed_surfaces", ()))),
        components_rebuilt=sorted(rebuild - set(result["metrics"].get("resumed_surfaces", ()))),
        component_writes=capture.writes, parsed_document_files=document_parse_calls,
        component_copy_bytes=component_copy_bytes,
        **embedding_metrics,
    )
    save_build_embedding_tokens(args, stop=True)
    fingerprint = await snapshot_content(workspace, result["workspace_revision"])
    probes = await build_probes(workspace, result["workspace_revision"], args)
    write_json(Path(args.worker_result), {"build": result, "content_fingerprint": fingerprint,
                                         "probes": probes})


def run_worker(args, runtime, result, crash_stage="none", *, component_resume=False):
    command = [sys.executable, "-m", "benchmark.robustness", "_build_worker",
               "--source", args.source, "--runtime", str(runtime), "--worker-result", str(result),
               "--surfaces", args.surfaces, "--crash-stage", crash_stage, "--after", str(args.after)]
    if args.fixture:
        command.append("--fixture")
    if args.offline:
        command.append("--offline")
    if args.count_embedding_tokens:
        command.append("--count-embedding-tokens")
    if args.native_incremental:
        command.append("--native-incremental")
    if getattr(args, "embedding_midpoint_file", None):
        command.extend(["--embedding-midpoint-file", args.embedding_midpoint_file])
    if crash_stage in EMBEDDING_CRASH_STAGES:
        if args.embedding_fault_probability is not None:
            command.extend(["--embedding-fault-probability", str(args.embedding_fault_probability),
                            "--embedding-fault-seed", str(args.embedding_fault_seed)])
        elif not args.embedding_midpoint_file:
            command.extend(["--embedding-after-batches", str(args.embedding_after_batches)])
    if args.embedding_model_dir:
        command.extend(["--embedding-model-dir", args.embedding_model_dir])
    if getattr(args, "embedding_batch_size", None) is not None:
        command.extend(["--embedding-batch-size", str(args.embedding_batch_size)])
    if args.crash_component:
        command.extend(["--crash-component", args.crash_component])
    if component_resume:
        command.append("--component-resume")
    if args.questions:
        command.extend(["--questions", args.questions])
    if args.graph_sources:
        command.extend(["--graph-sources",args.graph_sources])
    with result.with_suffix(".log").open("w") as log:
        pythonpath = os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH"))))
        return subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": pythonpath},
                              stdout=log, stderr=log, timeout=args.worker_timeout_s).returncode


def build_experiment(args, output, source):
    args.source = str(source)
    crashed = output / "crashed"
    settings = settings_for(crashed, args.fixture, args)
    workspace = workspace_for(settings)
    inventory = workspace._inventory(source)
    count = len(inventory)
    if args.embedding_midpoint_file:
        target = (source / args.embedding_midpoint_file).resolve()
        matching = [item for item in inventory if Path(item["uri"]) == target]
        if len(matching) != 1 or target.suffix.lower() not in DOCUMENT_EXTS:
            raise ValueError("embedding midpoint target must be a document in the workspace inventory")
        version = matching[0]["version"]
        if args.native_incremental and sum(
            item["version"] == version and Path(item["uri"]).suffix.lower() == target.suffix.lower()
            for item in inventory
        ) != 1:
            raise ValueError("native midpoint target must have an unambiguous source blob")
        args.embedding_midpoint_file = str(target)
    if args.crash_stage == "checkpoint" and not 1 <= args.after <= count:
        raise ValueError(f"--after must be between 1 and {count} supported sources")
    if args.offline and not args.fixture and "kb" in args.surfaces.split(",") and not args.embedding_model_dir:
        raise ValueError("offline KB builds require --embedding-model-dir")
    write_json(output / "manifest.json", provenance(args, source))
    code = run_worker(args, crashed, output / "interrupted.json", args.crash_stage)
    draw_path = output / "interrupted.embedding-draws.json"
    draws = json.loads(draw_path.read_text()) if draw_path.exists() else None
    if code != 23:
        if code == 0 and args.crash_stage in {"partial", "write-unconfirmed"} | EMBEDDING_CRASH_STAGES:
            summary = {"outcome": "interruption_not_applicable", "crash_stage": args.crash_stage}
            if args.crash_stage in EMBEDDING_CRASH_STAGES:
                summary["reason"] = "no embedding request with a remaining batch at the selected cut"
            if draws is not None:
                summary["embedding_fault"] = draws
                if draws["eligible_batches_examined"]:
                    summary.update(outcome="interruption_not_triggered",
                                   reason="no eligible batch was selected by the random draws")
            write_json(output / "summary.json", summary)
            return summary
        raise RuntimeError(f"expected injected process exit 23, got {code}; inspect interrupted.log")
    if workspace.current_revision is not None:
        raise RuntimeError("partial build was published")
    state = json.loads((crashed / "component-state.json").read_text())
    if workspace.status(state["revision"])["state"] != "BUILDING":
        raise RuntimeError("interrupted revision is not private")
    try:
        workspace.manifest(state["revision"])
    except CapabilityError:
        pass
    else:
        raise RuntimeError("interrupted manifest is readable")
    checkpoint_dir = workspace.root / "checkpoints"
    damage = {}
    if args.damage_component:
        if args.damage_component not in state["receipts"]:
            summary = {"outcome": "damage_not_applicable", "crash_stage": args.crash_stage}
            write_json(output / "summary.json", summary)
            return summary
        previous = workspace._settings_for(state["revision"])
        target = component_paths(previous, args.damage_component)[0]
        if target.is_dir():
            target = next(path for path in sorted(target.rglob("*")) if path.is_file())
        target.write_bytes(b"damaged experiment component")
        damage["component"] = args.damage_component
    if args.damage_checkpoint:
        checkpoints = sorted(checkpoint_dir.glob("*.json"))
        if not checkpoints:
            summary = {"outcome": "damage_not_applicable", "crash_stage": args.crash_stage}
            write_json(output / "summary.json", summary)
            return summary
        checkpoints[0].write_bytes(b"{")
        damage["source_checkpoint"] = checkpoints[0].name
    results, errors = {}, {}
    for policy in (("resume",) if args.native_incremental else BUILD_POLICIES):
        runtime = crashed if args.native_incremental else output / policy
        started = time.monotonic()
        if not args.native_incremental:
            shutil.copytree(crashed, runtime)
        clone_ms = (time.monotonic() - started) * 1000 if not args.native_incremental else 0
        if policy != "resume":
            shutil.rmtree(runtime / checkpoint_dir.relative_to(crashed), ignore_errors=True)
            candidate = checkpoint_dir.parent / "revisions" / state["revision"]
            for name in ("snapshot-ready.json", "snapshot-progress.json"):
                (runtime / (candidate / name).relative_to(crashed)).unlink(missing_ok=True)
        result_path = output / f"{policy}.json"
        try:
            code = run_worker(args, runtime, result_path, component_resume=policy == "component")
        except subprocess.TimeoutExpired:
            errors[policy] = {"outcome": "timeout", "log": result_path.with_suffix(".log").name}
            continue
        if code:
            errors[policy] = {"outcome": "error", "returncode": code, "log": result_path.with_suffix(".log").name}
            continue
        results[policy] = json.loads(result_path.read_text())
        results[policy]["build"]["metrics"]["experiment_clone_ms"] = clone_ms
    summary = {
        "outcome": "recovery_error" if errors else "ok", "policy_errors": errors,
        "fixture": args.fixture, "offline": args.offline,
        "external_api_calls": 0 if args.offline else None,
        "crash_stage": args.crash_stage, "after_sources": args.after, "damage": damage,
        "interrupted": state, "partial_revision_visible": False,
        **{p + "_metrics": r["build"]["metrics"] for p, r in results.items()},
        "logical_content_equal": (
            None if errors or args.native_incremental
            else len({r["content_fingerprint"] for r in results.values()}) == 1
        ),
        "resumed_bindings_verified": True if args.native_incremental and not errors else None,
        "downstream_probes_equal": (
            None if errors or args.native_incremental or not args.questions
            else all(r["probes"] == results["cold"]["probes"] for r in results.values())
        ),
        **{p + "_probes": r["probes"] for p, r in results.items()},
        "note": "Recovery costs exclude validation and experiment state cloning; offline probes do not generate answers.",
    }
    if draws is not None:
        summary["embedding_fault"] = draws
    write_json(output / "summary.json", summary)
    return summary



def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("query", "build", "_build_worker"):
        p = sub.add_parser(command)
        p.add_argument("--fixture", action="store_true", help="CPU-only smoke fixtures, never paper quality results")
        p.add_argument("--source")
        p.add_argument("--questions")
        p.add_argument("--graph-sources",help="Codex-prepared JSON assertions outside the raw workspace")
        p.add_argument("--surfaces", default="kb,graph,db" if command != "query" else "kb")
        if command == "_build_worker":
            p.add_argument("--runtime", required=True)
            p.add_argument("--worker-result", required=True)
        else:
            p.add_argument("--output", required=True, help="new directory; never overwrite an experiment")
        if command in {"build", "_build_worker"}:
            p.add_argument("--after", type=int, default=2)
            stages = ["none", "checkpoint", "partial", "write-unconfirmed", "publication", *sorted(EMBEDDING_CRASH_STAGES)]
            p.add_argument("--crash-stage", choices=stages + (["all"] if command == "build" else []),
                           default="checkpoint" if command == "build" else "none")
            p.add_argument("--embedding-after-batches", type=int,
                           help="completed embedding batches within one file before interrupting (default: 1)")
            p.add_argument("--embedding-fault-probability", type=float,
                           help="sample each eligible embedding batch instead of using a fixed batch number")
            p.add_argument("--embedding-fault-seed", type=int, default=0)
            p.add_argument("--embedding-midpoint-file", help="cut at the midpoint of this workspace document")
            p.add_argument("--crash-component", choices=["kb", "db", "graph"],
                           help="target of write-unconfirmed; defaults to the last available surface")
            p.add_argument("--offline", action="store_true", help="use local providers and probes")
            p.add_argument("--native-incremental", action="store_true",
                           help="only run native incremental recovery using local Dolt")
            p.add_argument("--embedding-model-dir", help="local HuggingFace model for offline KB builds")
            p.add_argument("--embedding-batch-size", type=int,
                           help="number of text chunks per local model batch (default: model configuration)")
            p.add_argument("--count-embedding-tokens", action="store_true",
                           help="record non-padding local embedding input tokens during build and recovery")
        if command == "_build_worker":
            p.add_argument("--component-resume", action="store_true")
        if command == "build":
            p.add_argument("--embedding-midpoints", action="store_true",
                           help="use each build unit's embedding_midpoint_file as its target")
            p.add_argument("--worker-timeout-s", type=float, default=300.)
            p.add_argument("--build-units", help="JSON array of {unit, source, ...} workspace descriptors")
            p.add_argument("--graph-sources-dir", help="root containing <unit>/graph_sources.json")
            p.add_argument("--evidence-source-map", help="benchmark evidence names mapped to original source names")
            p.add_argument("--damage-component", choices=["kb", "db", "graph"])
            p.add_argument("--damage-checkpoint", action="store_true")
        if command == "query":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--execution", choices=["retrieval", "sql", "agent"], default="retrieval")
            p.add_argument("--component", choices=["embedding", "vector", "db"], default="embedding")
            p.add_argument("--db-error", choices=sorted(DB_FAULTS), default="sqlite_busy")
            p.add_argument("--failure-probabilities", type=float, nargs="+", default=[0., .1, .5, 1.])
            p.add_argument("--outage-ms", type=float, default=0.)
            p.add_argument("--fault-delay-ms", type=float, default=0.)
            p.add_argument("--retries", type=int, default=1)
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--deadline-s", type=float, default=30.)
    return root


def resolve_crash_component(args):
    surfaces = args.surfaces.split(",")
    if args.crash_component is None:
        args.crash_component = next((surface for surface in ("graph", "db", "kb") if surface in surfaces), None)
    if args.crash_stage == "write-unconfirmed" and args.crash_component not in surfaces:
        raise ValueError("--crash-component must be one of the enabled surfaces")


def build_batch(args, output):
    manifest = Path(args.build_units)
    units = json.loads(manifest.read_text())
    if not isinstance(units, list) or not units:
        raise ValueError("--build-units requires a nonempty array")
    tasks = {}
    if args.questions:
        for entry in load_build_tasks(args.questions):
            tasks.setdefault(entry["unit"], []).append(entry.get("task", entry))
    registry = {}
    if args.evidence_source_map:
        registry = {name.lower(): value for name, value in
                    json.loads(Path(args.evidence_source_map).read_text()).items()}
    seen = set()
    for unit in units:
        name = unit["unit"]
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} or name in seen:
            raise ValueError(f"invalid or duplicate build unit id: {name!r}")
        seen.add(name)
        source = (manifest.parent / unit["source"]).resolve()
        if output.is_relative_to(source):
            raise ValueError(f"experiment output must be outside the source for {name}")
    stages = ("partial", "write-unconfirmed", "publication") if args.crash_stage == "all" else (args.crash_stage,)
    output.mkdir(parents=True, exist_ok=False)
    batch = {"units": len(units), "stages": stages, "native_incremental": args.native_incremental,
             "offline": args.offline, "units_hash": file_hash(manifest)}
    if args.embedding_fault_probability is not None:
        batch["embedding_fault"] = {"probability_per_eligible_batch": args.embedding_fault_probability,
                                    "seed": args.embedding_fault_seed}
    write_json(output / "batch.json", batch)
    outcomes = {}
    token_totals = {}
    paired_token_totals = {}
    for unit in units:
        name = unit["unit"]
        local = copy.copy(args)
        local.source = str((manifest.parent / unit["source"]).resolve())
        source = Path(local.source)
        if args.embedding_midpoints:
            local.embedding_midpoint_file = unit["embedding_midpoint_file"]
        applicable = set()
        if unit.get("surfaces"):
            local.surfaces = ",".join(unit["surfaces"]) if isinstance(unit["surfaces"], list) else unit["surfaces"]
        else:
            suffixes = {path.suffix.lower() for path in source.rglob("*") if path.is_file()
                        and not any(part.startswith(".") for part in path.relative_to(source).parts)}
            if suffixes & DOCUMENT_EXTS:
                applicable.update({"kb", "graph"})
            if suffixes & TABLE_EXTS:
                applicable.add("db")
            local.surfaces = ",".join(surface for surface in args.surfaces.split(",") if surface in applicable)
        local.questions = str((manifest.parent / unit["questions"]).resolve()) if unit.get("questions") else None
        unit_output = output / name
        unit_output.mkdir()
        if name in tasks and local.questions is None:
            local.questions = str(unit_output / "questions.json")
            translated = []
            for task in tasks[name]:
                if not task.get("expected_sources") and task.get("gold_evidence"):
                    gold = {Path(registry.get(e["file"].lower(), e["file"])).name.lower()
                            for e in task["gold_evidence"] if e.get("surface") == "rag" and e.get("file")}
                    mapped = [uri for uri, original in unit.get("mapping", {}).items()
                              if Path(original).name.lower() in gold]
                    task = {**task, "expected_sources": mapped or sorted(gold)}
                translated.append(task)
            write_json(Path(local.questions), translated)
        if unit.get("graph_sources"):
            local.graph_sources = str((manifest.parent / unit["graph_sources"]).resolve())
        elif args.graph_sources_dir:
            local.graph_sources = str(Path(args.graph_sources_dir) / name / "graph_sources.json")
        for stage in stages:
            dest = unit_output / stage
            local.crash_stage, local.crash_component = stage, args.crash_component
            local.damage_component = args.damage_component if stage == "publication" else None
            local.output = str(dest)
            row = {"unit": name, "crash_stage": stage, "surfaces": local.surfaces.split(",") if local.surfaces else []}
            if not local.surfaces:
                row["outcome"] = "no_applicable_surfaces"
            elif stage in EMBEDDING_CRASH_STAGES and "kb" not in row["surfaces"]:
                row.update(outcome="interruption_not_applicable", reason="KB is not enabled")
            elif stage == "write-unconfirmed" and local.crash_component and local.crash_component not in row["surfaces"]:
                row["outcome"] = "interruption_not_applicable"
            else:
                try:
                    resolve_crash_component(local)
                    dest.mkdir()
                    result = build_experiment(local, dest, source)
                    row.update(outcome=result["outcome"], logical_content_equal=result.get("logical_content_equal"))
                    if "reason" in result:
                        row["reason"] = result["reason"]
                    if "embedding_fault" in result:
                        row["embedding_fault"] = result["embedding_fault"]
                    if stage in EMBEDDING_CRASH_STAGES and "interrupted" in result:
                        row["embedding_interruption"] = result["interrupted"]["embedding_interruption"]
                    if result["outcome"] in {"ok", "recovery_error"}:
                        row["downstream_probes_equal"] = result["downstream_probes_equal"]
                        row["policy_errors"] = result["policy_errors"]
                        if local.native_incremental:
                            row["resumed_bindings_verified"] = result["resumed_bindings_verified"]
                        row["policies"] = {
                            policy: {key: metrics[key] for key in (
                                "recovery_wall_ms", "components_reused", "components_rebuilt",
                                "embedding_calls", "component_writes", "component_copy_bytes",
                            )} for policy in BUILD_POLICIES if (metrics := result.get(policy + "_metrics"))
                        }
                        if stage in EMBEDDING_CRASH_STAGES:
                            for policy, metrics in row["policies"].items():
                                recorded = result[policy + "_metrics"]
                                metrics.update({key: recorded[key] for key in (
                                    "embedding_texts", "interrupted_embedding_request_replays",
                                    "interrupted_embedding_chunks_recomputed",
                                    "interrupted_embedding_tokens_recomputed",
                                ) if key in recorded})
                except (OSError, RuntimeError, ValueError, CapabilityError, subprocess.TimeoutExpired) as exc:
                    row.update(outcome="error", error_type=type(exc).__name__,
                               detail=str(exc).replace(str(output), "."))
            if args.count_embedding_tokens:
                phase_tokens = {}
                for phase in ("interrupted", *BUILD_POLICIES):
                    path = dest / f"{phase}.embedding-tokens.json"
                    if path.is_file():
                        phase_tokens[phase] = json.loads(path.read_text())["input_tokens"]
                row["embedding_input_tokens"] = phase_tokens
                for phase, value in phase_tokens.items():
                    totals = token_totals.setdefault(phase, {"recorded_tasks": 0, "input_tokens": 0})
                    totals["recorded_tasks"] += 1
                    totals["input_tokens"] += value
                if row["outcome"] == "ok" and "interrupted" in phase_tokens:
                    row["embedding_input_tokens_init_plus_recovery"] = {}
                    for policy in row["policies"]:
                        if policy not in phase_tokens:
                            continue
                        initial = phase_tokens["interrupted"]
                        recovery = phase_tokens[policy]
                        row["embedding_input_tokens_init_plus_recovery"][policy] = initial + recovery
                        sums = paired_token_totals.setdefault(
                            policy, {"recorded_tasks": 0, "initial": 0, "recovery": 0, "total": 0}
                        )
                        sums["recorded_tasks"] += 1
                        sums["initial"] += initial
                        sums["recovery"] += recovery
                        sums["total"] += initial + recovery
            append_row(output / "units.jsonl", row)
            status = row["outcome"]
            outcomes[status] = outcomes.get(status, 0) + 1
    summary = {"units": len(units), "stages": stages, "outcomes": outcomes}
    if args.count_embedding_tokens:
        summary["embedding_input_tokens"] = token_totals
        summary["successful_recoveries_embedding_input_tokens"] = paired_token_totals
    write_json(output / "summary.json", summary)
    print(output / "summary.json")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.questions:
        args.questions = str(Path(args.questions).resolve())
    if args.graph_sources:
        args.graph_sources = str(Path(args.graph_sources).resolve())
    if getattr(args, "embedding_model_dir", None):
        model_dir = Path(args.embedding_model_dir).resolve()
        if not model_dir.is_dir():
            raise ValueError("--embedding-model-dir must be an existing local directory")
        args.embedding_model_dir = str(model_dir)
    if getattr(args, "embedding_batch_size", None) is not None:
        if args.embedding_batch_size < 1:
            raise ValueError("--embedding-batch-size must be positive")
        if not args.offline or args.fixture or not args.embedding_model_dir:
            raise ValueError("--embedding-batch-size requires --offline and a real --embedding-model-dir")
    probability = getattr(args, "embedding_fault_probability", None)
    midpoint = getattr(args, "embedding_midpoint_file", None) or getattr(args, "embedding_midpoints", False)
    if midpoint:
        if (args.crash_stage not in {EMBEDDING_CRASH_STAGE, "none"} or probability is not None
                or args.embedding_after_batches is not None or not args.count_embedding_tokens):
            raise ValueError("embedding midpoints require token counting and the embedding-mid-file stage")
        if args.command == "build" and args.embedding_midpoints and not args.build_units:
            raise ValueError("--embedding-midpoints requires --build-units")
    if probability is not None:
        if args.crash_stage != EMBEDDING_CRASH_STAGE or args.embedding_after_batches is not None:
            raise ValueError("random embedding interruption requires embedding-mid-file without a fixed batch number")
        if not 0 <= probability <= 1:
            raise ValueError("--embedding-fault-probability must be in [0,1]")
    if getattr(args, "crash_stage", None) in EMBEDDING_CRASH_STAGES:
        if probability is None and not midpoint:
            if args.embedding_after_batches is None:
                args.embedding_after_batches = 1
            if args.embedding_after_batches < 1:
                raise ValueError("--embedding-after-batches must be positive")
        if not args.offline or args.fixture or not args.embedding_model_dir:
            raise ValueError("embedding interruptions require --offline and a real --embedding-model-dir")
        if args.crash_stage == EMBEDDING_PERSISTED_STAGE and not args.native_incremental:
            raise ValueError("embedding-batch-persisted requires --native-incremental")
        if (not getattr(args, "build_units", None) and "kb" not in args.surfaces.split(",")):
            raise ValueError("embedding interruptions require the KB surface")
        if getattr(args, "damage_checkpoint", False):
            raise ValueError("embedding interruptions cannot be combined with --damage-checkpoint")
    elif getattr(args, "embedding_after_batches", None) is not None:
        raise ValueError("--embedding-after-batches requires an embedding crash stage")
    if getattr(args, "native_incremental", False):
        if not args.offline or args.fixture:
            raise ValueError("--native-incremental requires --offline and real local inputs")
        local_dolt_only()
    elif getattr(args, "offline", False):
        deny_network()
    if getattr(args, "count_embedding_tokens", False) and (args.fixture or not args.offline):
        raise ValueError("--count-embedding-tokens requires a real offline build")
    if args.command == "_build_worker":
        if probability is not None:
            args._random_embedding_cut = RandomEmbeddingCut(args)
        try:
            with ExitStack() as stack:
                if args.count_embedding_tokens:
                    counter = args._embedding_token_counter = BuildEmbeddingTokens(args.worker_result)
                    stack.callback(counter.save)
                    stack.enter_context(counter.observe())
                asyncio.run(build_worker(args))
        finally:
            if cut := getattr(args, "_random_embedding_cut", None):
                cut.save()
        return
    if args.command == "query":
        if any(not 0 <= p <= 1 for p in args.failure_probabilities) or not 0 <= args.retries <= 5:
            raise ValueError("probabilities must be in [0,1]; retries must be in [0,5]")
        if (args.repeats < 1 or args.deadline_s <= 0 or min(args.outage_ms, args.fault_delay_ms) < 0
                or not all(math.isfinite(v) for v in (args.deadline_s, args.outage_ms, args.fault_delay_ms))):
            raise ValueError("invalid repeats/deadline/fault duration")
        required_surface = "db" if args.component == "db" else "kb"
        if required_surface not in args.surfaces.split(","):
            raise ValueError(f"query fault experiments require the {required_surface} surface")
        if args.execution == "sql" and args.component != "db":
            raise ValueError("--execution sql requires --component db")
        if args.component == "db" and args.execution == "retrieval":
            raise ValueError("--component db requires --execution sql or agent")
    else:
        if args.crash_stage == "none":
            raise ValueError("build experiment requires an interruption stage")
        if args.crash_stage == "all" and not args.build_units:
            raise ValueError("--crash-stage all requires --build-units")
        if args.build_units and args.source:
            raise ValueError("--build-units and --source are mutually exclusive")
        if args.damage_component and args.crash_stage not in {"publication", "all"}:
            raise ValueError("--damage-component requires a pre-publication snapshot")
        if args.native_incremental and (args.damage_checkpoint or args.damage_component or args.crash_stage == "checkpoint"):
            raise ValueError("native incremental runs do not support checkpoint interruption or artifact damage")
        if args.build_units:
            if args.graph_sources_dir:
                args.graph_sources_dir = str(Path(args.graph_sources_dir).resolve())
            build_batch(args, Path(args.output).resolve())
            return
        resolve_crash_component(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, tasks = source_and_tasks(args, output)
    if args.command == "query":
        asyncio.run(query_experiment(args, output, source, tasks))
    else:
        experiment = build_experiment(args, output, source)
        if experiment["outcome"] == "recovery_error":
            print(output / "summary.json")
            raise SystemExit(1)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
