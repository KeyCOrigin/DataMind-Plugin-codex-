"""Shared immutable source objects plus native KB/DB snapshot bindings."""

from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import stat
import struct
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from filelock import AsyncFileLock
from datamind.core.errors import CapabilityError, ConfigError
from datamind.core.protocols import GraphTriple, GraphEvidence
from datamind.core.logging import current_context
from datamind.capabilities.ingest.formats import (
    DOCUMENT_EXTS,
    TABLE_EXTS,
    extract_document,
    xlsx_rows_without_empty_tail,
)
from datamind.capabilities.kb.indexer import _hash, _split_text
from datamind.workspace_native import blocking, completed


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()


def incremental_enabled(workspace, surfaces):
    mode = workspace.settings.workspace.storage_mode
    if mode == "snapshot":
        return False
    if mode == "incremental":
        return True
    head = workspace.current_revision
    if head:
        return workspace.manifest(head).get("format_version") == 2
    cfg = workspace.settings
    return cfg.kb.vector_store == "lancedb" and cfg.db.dialect == "dolt"


def store_bytes(root, kind, data, suffix):
    key = digest(data)
    path = root / kind / key[:2] / (key + suffix)
    if path.exists():
        return path.relative_to(root).as_posix(), False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return path.relative_to(root).as_posix(), True


def save_artifact(root, key, value):
    from datamind.workspace import _atomic_json

    path = root / "artifacts" / key[:2] / (key + ".json")
    _atomic_json(path, value, overwrite=False)
    return path.relative_to(root).as_posix()


def vector_cached(path, dimension):
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        return False
    if dimension is None:
        return info.st_size % 4 == 0
    return info.st_size == 4 * dimension


def persist_vector_batch(root, batch, vectors, dimension):
    if len(vectors) != len(batch):
        raise CapabilityError("kb", "embedding count mismatch")
    if any(len(vector) != dimension for vector in vectors):
        raise CapabilityError("kb", "embedding dimension mismatch")
    for (_, ref), vector in zip(batch, vectors):
        dest = root / ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = struct.pack("<" + "f" * dimension, *vector)
        fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix=".vector-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, dest)
        finally:
            Path(temporary).unlink(missing_ok=True)


async def embed_missing_vectors(root, missing, embedding, batch_size, metrics):
    if batch_size < 1:
        raise ConfigError("Embedding batch size must be positive")
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        vectors = await embedding.embed_texts([text for text, _ in batch])
        await blocking(persist_vector_batch, root, batch, vectors, embedding.dimension)
        metrics["embedding_calls"] += 1
        metrics["embedding_texts"] += len(batch)


def kb_version_available(root, binding):
    import lancedb

    path = root / binding["path"]
    if not path.is_dir():
        return False
    try:
        table = lancedb.connect(path).open_table("workspace")
        table.checkout(binding["version"])
        ids = table.search(None).select(["id"]).to_list()
        return (
            table.schema.field("embedding").type.list_size == binding["dimension"]
            and len(ids) == len(binding["chunks"])
            and {row["id"] for row in ids} == set(binding["chunks"])
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError):
        return False


def canonical_tables(workspace, item, snapshot):
    """Stream CSV/TSV and XLSX rows into immutable typed-cell table objects."""
    source = Path(item["uri"])
    streams = []
    closers = []
    if source.suffix.lower() in {".csv", ".tsv"}:
        handle = snapshot.open(newline="", encoding="utf-8", errors="replace")
        closers.append(handle)
        rows = iter(
            csv.reader(
                handle, delimiter="\t" if source.suffix.lower() == ".tsv" else ","
            )
        )
        header = next(rows, None)
        if header is not None:
            streams.append((source.stem, header, rows))
    elif source.suffix.lower() == ".xlsx":
        import openpyxl

        book = openpyxl.load_workbook(snapshot, read_only=True, data_only=True)
        closers.append(book)
        for sheet in book.worksheets:
            values, width = xlsx_rows_without_empty_tail(sheet)
            if width:
                rows = iter(row + (None,) * (width - len(row)) for row in values)
                streams.append((sheet.title, next(rows), rows))
    else:
        raise ConfigError(
            "Incremental DB supports CSV, TSV and XLSX; convert legacy XLS first"
        )
    output = []
    try:
        for sheet, header, rows in streams:
            used = set()
            cols = []
            for i, cell in enumerate(header, 1):
                value = str(cell).strip() if cell is not None else ""
                if (
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value)
                    or value.casefold() in used
                    or value.casefold() == "_source_file"
                ):
                    value = "col_" + str(i)
                base = value
                n = 2
                while value.casefold() in used:
                    value = base[:58] + "_" + str(n)
                    n += 1
                cols.append(value)
                used.add(value.casefold())
            cols.append("_source_file")
            fd, temp = tempfile.mkstemp(dir=workspace.root, prefix=".table-")
            count = 0
            sha = hashlib.sha256()
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(cols)
                    for row in rows:
                        values = list(row[: len(header)])
                        values += [""] * (len(header) - len(values))
                        # Preserve existing TEXT import semantics, including NULL.
                        values = [None if x is None else str(x) for x in values] + [
                            item["uri"]
                        ]
                        writer.writerow(
                            [json.dumps(x, ensure_ascii=False) for x in values]
                        )
                        count += 1
                    handle.flush()
                    os.fsync(handle.fileno())
                with open(temp, "rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        sha.update(block)
                key = sha.hexdigest()
                target = workspace.root / "tables" / key[:2] / (key + ".csv")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.replace(temp, target)
                source_id = "file:" + digest(item["uri"].encode())
                output.append(
                    {
                        "name": "ws_" + digest((source_id + sheet).encode())[:24],
                        "sheet": sheet,
                        "columns": cols,
                        "rows": count,
                        "hash": key,
                        "path": target.relative_to(workspace.root).as_posix(),
                    }
                )
            finally:
                Path(temp).unlink(missing_ok=True)
    finally:
        for closer in closers:
            closer.close()
    return output


@asynccontextmanager
async def _writer(workspace):
    if getattr(workspace, "_writer_held", False):
        yield
    else:
        async with AsyncFileLock(
            str(workspace.root / "writer.lock"),
            timeout=workspace.settings.workspace.lock_timeout_s,
        ):
            yield


async def update_incremental(
    workspace,
    path,
    *,
    surfaces,
    relation_types,
    max_triples_per_file,
    metrics,
    graph_sources,
):
    from datamind.workspace import _atomic_json
    from datamind.workspace_graph import graph_artifacts_available, put_graph
    from datamind.workspace_native import DoltRepository

    selected = set(surfaces)
    cfg = workspace.settings
    if not selected or not selected <= {"kb", "graph", "db"}:
        raise ConfigError("Invalid workspace surfaces")
    if not 1 <= max_triples_per_file <= 10000:
        raise ConfigError("Invalid graph extraction limit")
    if "kb" in selected and cfg.kb.vector_store != "lancedb":
        raise ConfigError(
            "Incremental Workspace KB requires lancedb; configure kb.vector_store or use snapshot mode"
        )
    if "db" in selected and (cfg.db.dialect != "dolt" or cfg.db.dsn):
        raise ConfigError(
            "Incremental Workspace DB requires local Dolt without an external DSN"
        )
    if "graph" in selected and graph_sources is None:
        raise ConfigError("Incremental Workspace requires Codex-prepared graph_sources")
    source = Path(path).expanduser().resolve()
    started = time.monotonic()
    inventory = await blocking(workspace._inventory, source)
    metrics.update(
        inventory_ms=(time.monotonic() - started) * 1000,
        sources_total=len(inventory),
        object_bytes_written=0,
        objects_reused=0,
        kb_rows_upserted=0,
        kb_rows_deleted=0,
        db_tables_changed=0,
        db_tables_reused=0,
    )
    # Capture the expected parent BEFORE acquiring the writer lock. A queued
    # stale build must conflict, not silently rebase to another caller's update.
    with workspace._connect() as db:
        parent, epoch = db.execute(
            "SELECT revision,epoch FROM head WHERE singleton=1"
        ).fetchone()
    if (
        hasattr(workspace, "_expected_parent")
        and (parent, epoch) != workspace._expected_parent
    ):
        raise CapabilityError("workspace", "publication conflict during migration")
    if (
        parent
        and workspace.manifest(parent).get("format_version") != 2
        and not hasattr(workspace, "_migration_prepared")
    ):
        raise ConfigError(
            "Run explicit workspace migration before changing an existing snapshot Workspace to incremental storage"
        )
    old = workspace.manifest(parent) if parent else None
    prior = (
        {s["uri"]: s for s in old["sources"]}
        if old and old.get("format_version") == 2
        else {}
    )
    assertions = {}
    docs = {
        s["uri"]: s
        for s in inventory
        if Path(s["uri"]).suffix.lower() in DOCUMENT_EXTS - TABLE_EXTS
    }
    for entry in graph_sources or []:
        uri = entry["uri"]
        if (
            uri not in docs
            or uri in assertions
            or entry["version"] != docs[uri]["version"]
        ):
            raise ConfigError("Graph source must match inventory and content hash")
        triples = [GraphTriple(**v) for v in entry["triples"]]
        if len(triples) > max_triples_per_file:
            raise ConfigError("Too many graph triples for source")
        for triple in triples:
            if relation_types is not None and triple.relation not in relation_types:
                raise ConfigError("Graph relation is outside vocabulary")
            triple.source = uri
            triple.evidence = [
                GraphEvidence(
                    source_id="file:" + digest(uri.encode()),
                    source_version=entry["version"],
                    locator={"uri": uri},
                    method="extracted",
                    pipeline_version="codex-explicit-v1",
                )
            ]
        assertions[uri] = {
            "triples": [t.model_dump(mode="json") for t in triples],
            "truncated": bool(entry.get("truncated")),
        }
    if "graph" in selected:
        for uri, item in docs.items():
            if uri not in assertions:
                before = prior.get(uri)
                if (
                    before is None
                    or before["version"] != item["version"]
                    or not before.get("graph_artifact")
                ):
                    raise ConfigError(
                        "Supply graph assertions for every new or changed document, including empty lists"
                    )
                # Unchanged assertions are immutable and reused, not re-extracted.
    context = current_context()
    skills = list(context.extra.get("applied_skills", {}).values()) if context else []
    recipe = {
        "version": "workspace-v2",
        "surfaces": sorted(selected),
        "embedding": cfg.embedding.model_dump(mode="json", exclude={"api_key"}),
        "chunk_size": cfg.retrieval.chunk_size,
        "chunk_overlap": cfg.retrieval.chunk_overlap,
        "relation_types": relation_types,
        "max_triples_per_file": max_triples_per_file,
        "applied_skills": skills,
    }
    if "db" in selected and any(Path(item["uri"]).suffix.lower() == ".xlsx" for item in inventory):
        recipe["xlsx_schema"] = "trim-empty-tail-v1"
    pipeline = digest(encoded(recipe))
    # The input plan intentionally excludes graph assertions and graph policy.
    # Correcting the graph must not invalidate completed KB and DB bindings.
    data_pipeline = digest(encoded({
        k: v for k, v in recipe.items() if k not in {"relation_types", "max_triples_per_file"}
    }))
    plan = digest(encoded(["incremental-resume-v1", parent, epoch, str(source), inventory, sorted(selected)]))
    graph_key = digest(encoded([relation_types, max_triples_per_file, assertions]))
    async with _writer(workspace):
        with workspace._lock(), workspace._connect() as db:
            if db.execute(
                "SELECT revision,epoch FROM head WHERE singleton=1"
            ).fetchone() != (parent, epoch):
                raise CapabilityError(
                    "workspace", "publication conflict; current revision changed"
                )
            saved = db.execute(
                """SELECT p.revision,p.receipt FROM incremental_progress p
                   JOIN revisions r ON r.id=p.revision
                   WHERE p.plan=? AND r.state IN ('BUILDING','FAILED')""",
                (plan,),
            ).fetchone()
            if saved:
                revision, payload = saved
                progress = json.loads(payload)
                db.execute("UPDATE revisions SET state='BUILDING',error=NULL WHERE id=?", (revision,))
            else:
                revision = "revision-" + uuid.uuid4().hex
                progress = {"format_version": 2, "sources": [], "snapshots": {}}
                db.execute(
                    "INSERT INTO revisions VALUES (?,?,?,NULL)", (revision, parent, "BUILDING")
                )
                db.execute("INSERT INTO incremental_builds VALUES (?)", (revision,))
                db.execute(
                    "INSERT INTO incremental_progress VALUES (?,?,?,?)",
                    (revision, epoch, plan, json.dumps(progress)),
                )
            # A new plan supersedes the last interrupted build in this profile.
            db.execute("DELETE FROM incremental_progress WHERE revision<>?", (revision,))
        metrics["workspace_revision"] = revision
        metrics["resumed_surfaces"] = []
        root = workspace.root / "revisions" / revision
        _atomic_json(root / "build.json", {"format_version": 2, "parent": parent})
        embedding = None
        sources = []
        bindings = {}
        all_chunks = {}
        tables = {}
        pinned_vectors = set(progress.get("vector_refs", ()))

        def write_progress():
            with workspace._connect() as db:
                db.execute("UPDATE incremental_progress SET receipt=? WHERE revision=?",
                           (json.dumps(progress), revision))

        def save_progress(surface):
            if surface == "graph":
                progress["graph_sources"] = {
                    s["uri"]: {"graph_artifact": s["graph_artifact"], "truncated": s["truncated"]}
                    for s in sources
                }
            graph_refs = progress.get("graph_sources", {})
            progress["sources"] = [{**s, **graph_refs.get(s["uri"], {})} for s in sources]
            progress["snapshots"][surface] = bindings[surface]
            write_progress()

        embedding_recipe = cfg.embedding.model_dump(mode="json", exclude={"api_key"})
        embedding_fingerprint = digest(encoded(embedding_recipe))
        old_kb = old.get("snapshots", {}).get("kb") if old and old.get("format_version") == 2 else None
        dimension = cfg.embedding.dimension
        if dimension is None and old_kb and old_kb["embedding_fingerprint"] == embedding_fingerprint:
            dimension = old_kb["dimension"]
        if dimension is None and progress.get("vector_embedding_fingerprint") == embedding_fingerprint:
            dimension = progress.get("vector_dimension")

        def load_embedding():
            nonlocal embedding, dimension
            if embedding is None:
                from datamind import runtime

                embedding = runtime.build_embedding(cfg.embedding, fallback_llm=cfg.llm)
                if dimension is not None and embedding.dimension != dimension:
                    raise CapabilityError("kb", "embedding dimension mismatch")
                dimension = embedding.dimension
            return embedding

        try:
            for item in inventory:
                start = time.monotonic()
                original = Path(
                    getattr(workspace, "_migration_sources", {}).get(
                        item["uri"], item["uri"]
                    )
                )
                suffix = Path(item["uri"]).suffix.lower()
                blob = (
                    workspace.root
                    / "objects"
                    / item["version"][7:9]
                    / (item["version"][7:] + suffix)
                )
                if blob.exists():
                    blob_ref = blob.relative_to(workspace.root).as_posix()
                    metrics["objects_reused"] += 1
                else:
                    # Stream the original into immutable object storage.
                    fd, temp = tempfile.mkstemp(dir=workspace.root, prefix=".source-")
                    sha = hashlib.sha256()
                    try:
                        with os.fdopen(fd, "wb") as out, original.open("rb") as inp:
                            for block in iter(lambda: inp.read(1024 * 1024), b""):
                                out.write(block)
                                sha.update(block)
                            out.flush()
                            os.fsync(out.fileno())
                        if "sha256:" + sha.hexdigest() != item["version"]:
                            raise CapabilityError(
                                "workspace", "source changed during build; retry"
                            )
                        blob.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(temp, blob)
                        blob_ref = blob.relative_to(workspace.root).as_posix()
                        metrics["object_bytes_written"] += blob.stat().st_size
                    finally:
                        Path(temp).unlink(missing_ok=True)
                metrics["source_copy_ms"] += (time.monotonic() - start) * 1000
                key = digest(encoded([item["uri"], item["version"], data_pipeline, "prepared-data-v1"]))
                artifact = workspace.root / "artifacts" / key[:2] / (key + ".json")
                start = time.monotonic()
                if artifact.exists():
                    prepared = json.loads(artifact.read_text())
                    metrics["sources_reused"] += 1
                    metrics["checkpoint_read_ms"] += (time.monotonic() - start) * 1000
                else:
                    # Read existing v2 manifests as well as independently prepared data.
                    previous = prior.get(item["uri"], {})
                    if (
                        old
                        and (old.get("data_pipeline_hash") == data_pipeline or old["pipeline_hash"] == pipeline)
                        and previous.get("version") == item["version"]
                    ):
                        artifact = workspace.root / previous["artifact"]
                        prepared = json.loads(artifact.read_text())
                        metrics["sources_reused"] += 1
                    elif item["uri"] in getattr(workspace, "_migration_prepared", {}):
                        prepared = dict(workspace._migration_prepared[item["uri"]])
                        artifact = workspace.root / save_artifact(
                            workspace.root, key, prepared
                        )
                        metrics["sources_reused"] += 1
                    else:
                        prepared = {
                            "chunks": [],
                            "tables": [],
                            "graph_artifact": None,
                            "truncated": False,
                        }
                        if "kb" in selected and suffix in DOCUMENT_EXTS:
                            text_value = (await blocking(extract_document, blob)).text
                            parts = _split_text(
                                text_value,
                                chunk_size=cfg.retrieval.chunk_size,
                                chunk_overlap=cfg.retrieval.chunk_overlap,
                            )
                            missing = []
                            missing_refs = set()
                            refs = []
                            for part in parts:
                                vector_key = digest(encoded([embedding_recipe, part]))
                                ref = (
                                    "vectors/"
                                    + vector_key[:2]
                                    + "/"
                                    + vector_key
                                    + ".f32"
                                )
                                refs.append(ref)
                                if ref not in missing_refs and not vector_cached(workspace.root / ref, dimension):
                                    missing.append((part, ref))
                                    missing_refs.add(ref)
                            dirty = False
                            if missing:
                                load_embedding()
                                if progress.get("vector_embedding_fingerprint") != embedding_fingerprint:
                                    progress["vector_embedding_fingerprint"] = embedding_fingerprint
                                    progress["vector_dimension"] = dimension
                                    dirty = True
                            if new_refs := set(refs) - pinned_vectors:
                                # Register planned objects before the first batch can reach disk.
                                pinned_vectors.update(new_refs)
                                progress["vector_refs"] = sorted(pinned_vectors)
                                dirty = True
                            if dirty:
                                write_progress()
                            if missing:
                                await embed_missing_vectors(
                                    workspace.root, missing, embedding, cfg.embedding.batch_size, metrics
                                )
                            prepared["chunks"] = [
                                {
                                    "id": _hash(part, item["uri"], ordinal=i),
                                    "text": part,
                                    "vector_ref": refs[i],
                                    "metadata": {
                                        "source": item["uri"],
                                        "source_id": "file:"
                                        + digest(item["uri"].encode()),
                                        "source_version": item["version"],
                                        "_chunk_ordinal": i,
                                    },
                                }
                                for i, part in enumerate(parts)
                            ]
                        if "db" in selected and suffix in TABLE_EXTS:
                            prepared["tables"] = await blocking(
                                canonical_tables, workspace, item, blob
                            )
                            metrics["parsed_table_files"] += 1
                        artifact = workspace.root / save_artifact(
                            workspace.root, key, prepared
                        )
                        metrics["sources_computed"] += 1
                        metrics["source_compute_ms"] += (
                            time.monotonic() - start
                        ) * 1000
                source_entry = {
                    **item,
                    "source_id": "file:" + digest(item["uri"].encode()),
                    "snapshot": blob_ref,
                    "artifact": artifact.relative_to(workspace.root).as_posix(),
                    "graph_artifact": None,
                    "truncated": False,
                }
                sources.append(source_entry)
                all_chunks.update((c["id"], c) for c in prepared["chunks"])
                tables.update((t["name"], t) for t in prepared["tables"])
                metrics["sources_completed"] += 1
            start = time.monotonic()
            if "kb" in selected:
                from datamind.capabilities.kb.providers.lancedb_store import (
                    LanceDBVectorStore,
                )

                new_fingerprints = {key: digest(encoded(c)) for key, c in all_chunks.items()}
                saved_kb = progress["snapshots"].get("kb")
                if (
                    saved_kb
                    and saved_kb["embedding_fingerprint"] == embedding_fingerprint
                    and saved_kb["chunks"] == new_fingerprints
                    and await blocking(kb_version_available, workspace.root, saved_kb)
                ):
                    bindings["kb"] = saved_kb
                    metrics["resumed_surfaces"].append("kb")
                else:
                    if dimension is None:
                        load_embedding()
                    kb_key = digest(encoded([embedding_recipe, dimension]))
                    location = "shared/kb/" + kb_key
                    store = LanceDBVectorStore(
                        persist_dir=str(workspace.root / location),
                        collection_name="workspace",
                        dimension=dimension,
                    )
                    base = old_kb
                    previous_chunks = (
                        base.get("chunks", {}) if base and base["path"] == location else {}
                    )
                    if (
                        base
                        and base["path"] == location
                        and store.latest_version != base["version"]
                    ):
                        await blocking(store._collection.restore, base["version"])
                    elif not previous_chunks and store.existing_count:
                        await blocking(store._collection.delete, "true")
                    removed = set(previous_chunks) - set(all_chunks)
                    if removed:
                        await completed(store.delete(sorted(removed)))
                    changed = [
                        c
                        for key, c in all_chunks.items()
                        if previous_chunks.get(key) != new_fingerprints[key]
                    ]
                    for offset in range(0, len(changed), 256):
                        batch = changed[offset : offset + 256]
                        vectors = []
                        for chunk in batch:
                            data = (workspace.root / chunk["vector_ref"]).read_bytes()
                            vectors.append(
                                list(struct.unpack("<" + "f" * (len(data) // 4), data))
                            )
                        await completed(
                            store.add(
                                ids=[c["id"] for c in batch],
                                texts=[c["text"] for c in batch],
                                embeddings=vectors,
                                metadatas=[c["metadata"] for c in batch],
                            )
                        )
                    version = store.latest_version
                    tag = "dm_workspace_" + str(version)
                    if tag not in store._collection.tags.list():
                        await blocking(store._collection.tags.create, tag, version)
                    bindings["kb"] = {
                        "provider": "lancedb",
                        "path": location,
                        "version": version,
                        "dimension": dimension,
                        "embedding_fingerprint": embedding_fingerprint,
                        "chunks": new_fingerprints,
                    }
                    metrics["kb_rows_upserted"] = len(changed)
                    metrics["kb_rows_deleted"] = len(removed)
                    save_progress("kb")
            if "db" in selected:
                base = (
                    old.get("snapshots", {}).get("db")
                    if old and old.get("format_version") == 2
                    else None
                )
                hashes = {n: t["hash"] for n, t in tables.items()}
                repository = DoltRepository(workspace)
                saved_db = progress["snapshots"].get("db")
                if (
                    saved_db
                    and saved_db["tables"] == hashes
                    and await repository.has_commit(saved_db["version"])
                ):
                    bindings["db"] = saved_db
                    metrics["resumed_surfaces"].append("db")
                elif base and base["tables"] == hashes:
                    bindings["db"] = base
                    save_progress("db")
                else:
                    bindings["db"] = await repository.build(base, tables, revision)
                    save_progress("db")
                before = base.get("tables", {}) if base else {}
                if "db" in metrics["resumed_surfaces"]:
                    before = hashes
                metrics["db_tables_reused"] = sum(
                    before.get(n) == v for n, v in hashes.items()
                )
                metrics["db_tables_changed"] = (
                    len(hashes)
                    - metrics["db_tables_reused"]
                    + len(before.keys() - hashes.keys())
                )
            if "graph" in selected:
                saved_graph = progress["snapshots"].get("graph")
                graph_entries = progress.get("graph_sources", {})
                if (
                    saved_graph
                    and progress.get("graph_key") == graph_key
                    and set(graph_entries) == {item["uri"] for item in sources}
                    and await blocking(graph_artifacts_available, workspace.root, graph_entries.values())
                ):
                    for item in sources:
                        entry = graph_entries[item["uri"]]
                        item.update(graph_artifact=entry["graph_artifact"], truncated=entry["truncated"])
                    bindings["graph"] = saved_graph
                    metrics["resumed_surfaces"].append("graph")
                else:
                    for item in sources:
                        uri = item["uri"]
                        assertion = assertions.get(uri)
                        if assertion is not None:
                            item["graph_artifact"] = put_graph(workspace.root, assertion["triples"])
                            item["truncated"] = assertion["truncated"]
                        else:
                            previous = prior.get(uri, {})
                            if previous.get("version") == item["version"] and previous.get("graph_artifact"):
                                item["graph_artifact"] = previous["graph_artifact"]
                                item["truncated"] = previous["truncated"]
                            else:
                                item["graph_artifact"] = put_graph(workspace.root, [])
                    bindings["graph"] = {"provider": "sqlite-source-graph", "path": "graph.sqlite3"}
                    progress["graph_key"] = graph_key
                    save_progress("graph")
            metrics["materialization_ms"] += (time.monotonic() - start) * 1000
            if await blocking(workspace._inventory, source) != inventory:
                raise CapabilityError(
                    "workspace", "workspace changed during build; retry"
                )
            manifest = {
                "format_version": 2,
                "workspace_revision": revision,
                "parent_revision": parent,
                "parent_epoch": epoch,
                "source_root": str(source),
                "surfaces": sorted(selected),
                "pipeline_hash": pipeline,
                "data_pipeline_hash": data_pipeline,
                "sources": sources,
                "snapshots": bindings,
                "reused_sources": metrics["sources_reused"],
                "applied_skills": skills,
                "truncated_sources": [s["uri"] for s in sources if s["truncated"]],
            }
            _atomic_json(root / "manifest.json", manifest)
            start = time.monotonic()
            with workspace._lock(), workspace._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute(
                    "SELECT revision,epoch FROM head WHERE singleton=1"
                ).fetchone() != (parent, epoch):
                    raise CapabilityError(
                        "workspace",
                        "publication conflict; another build published first",
                    )
                db.execute(
                    "UPDATE revisions SET state='PUBLISHED' WHERE id=?", (revision,)
                )
                db.execute(
                    "UPDATE head SET revision=?,epoch=epoch+1 WHERE singleton=1",
                    (revision,),
                )
                db.execute("DELETE FROM incremental_progress WHERE revision=?", (revision,))
            metrics["publication_ms"] = (time.monotonic() - start) * 1000
            return manifest
        except BaseException as exc:
            with workspace._connect() as db:
                db.execute(
                    "UPDATE revisions SET state='FAILED',error=? WHERE id=?",
                    (type(exc).__name__, revision),
                )
            raise
        finally:
            if embedding and callable(getattr(embedding, "aclose", None)):
                await embedding.aclose()


async def reader_system(workspace, manifest, selected):
    from datamind.capabilities.graph import GraphService, build_graph_tools
    from datamind.capabilities.db import build_db_tools
    from datamind.capabilities.kb import build_kb_tools
    from datamind.capabilities.kb.service import KBService
    from datamind.capabilities.kb.providers.lancedb_store import LanceDBVectorStore
    from datamind.core.registry import retriever_registry
    from datamind.core.contracts import ToolAccess
    from datamind.workspace_native import PinnedVectors, PinnedDB
    from datamind.workspace_graph import GraphView
    from datamind import runtime

    settings = workspace.settings.model_copy(deep=True)
    settings.hooks.path_allowlist_extra = [
        *settings.hooks.path_allowlist_extra,
        str(workspace.root),
        manifest["source_root"],
    ]
    system = await workspace._system(settings, set())
    try:
        if "kb" in selected:
            binding = manifest["snapshots"]["kb"]
            # A historical index must use the model that built its embeddings.
            current = settings.embedding.model_dump(mode="json", exclude={"api_key"})
            if digest(encoded(current)) != binding["embedding_fingerprint"]:
                raise ConfigError(
                    "Pinned KB embedding configuration changed; use the original configuration"
                )
            embedding = runtime.build_embedding(
                settings.embedding, fallback_llm=settings.llm
            )
            raw = LanceDBVectorStore(
                persist_dir=str(workspace.root / binding["path"]),
                collection_name="workspace",
                dimension=binding["dimension"],
            )
            store = PinnedVectors(raw, binding["version"])
            strategy = settings.retrieval.strategy
            if strategy not in {"simple", "hybrid"}:
                raise ConfigError(
                    "Incremental reads support simple and hybrid retrieval"
                )
            options = {"vector_store": store, "embedding": embedding}
            if strategy == "hybrid":
                options.update(
                    vector_failure_fallback=settings.retrieval.vector_failure_fallback,
                    vector_retries=settings.retrieval.vector_retries,
                    vector_timeout_s=settings.retrieval.vector_timeout_s,
                )
            retriever = retriever_registry.create(strategy, **options)
            kb = KBService(
                embedding=embedding,
                vector_store=store,
                retriever=retriever,
                data_dir=settings.data.data_dir,
                retrieval_cfg=settings.retrieval,
            )

            async def documents():
                return [
                    {
                        "path": s["uri"],
                        "source_version": s["version"],
                        "size": (workspace.root / s["snapshot"]).stat().st_size,
                    }
                    for s in manifest["sources"]
                ]

            kb.list_documents = documents
            system.services.kb = kb
            system.services.embedding = embedding
            system.tools.extend(
                [s for s in build_kb_tools(kb) if s.access != ToolAccess.WRITE]
            )
        if "graph" in selected:
            graph = GraphService(
                store=GraphView(
                    workspace.root,
                    {
                        s["graph_artifact"]
                        for s in manifest["sources"]
                        if s.get("graph_artifact")
                    },
                ),
                data_dir=settings.data.data_dir,
                storage_dir=workspace.root,
            )
            system.services.graph = graph
            system.tools.extend(
                [s for s in build_graph_tools(graph) if s.access != ToolAccess.WRITE]
            )
        if "db" in selected:
            db = PinnedDB(workspace, manifest["snapshots"]["db"])
            system.services.db = db
            system.tools.extend(
                [
                    s
                    for s in build_db_tools(db)
                    if s.access != ToolAccess.WRITE
                    and s.name not in {"db_query_nl", "db_query_nl_at_version"}
                ]
            )
        return system, settings
    except BaseException:
        await system.aclose()
        raise
