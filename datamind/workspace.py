"""Immutable workspace revisions over the existing DataMind capability services.

The catalog is safe across processes on one host. Builds prepare private indexes;
publication uses compare-and-swap against their starting revision. Readers retain
one immutable revision across all tool calls. Incremental storage shares source
objects, source-scoped graph assertions, LanceDB versions and Dolt table trees.
Legacy local snapshots remain readable and can be migrated explicitly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import replace
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator
from filelock import FileLock, Timeout as LockTimeout

from datamind.capabilities.ingest.formats import DOCUMENT_EXTS, TABLE_EXTS, extract_document, extract_tabular
from datamind.capabilities.kb.indexer import _hash, _split_text, build_index_manifest, write_manifest_atomic
from datamind.core.context import RequestContext
from datamind.core.errors import CapabilityError, ConfigError, ExternalServiceError
from datamind.core.logging import bind_context, current_context, get_logger
from datamind.core.protocols import GraphTriple, GraphEvidence
from datamind.core.recovery import recover_read, recovery_summary
from datamind.workspace_catalog import WorkspaceCatalog


logger = get_logger('workspace')


class _CheckpointChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]


class _SourceCheckpoint(BaseModel):
    chunks: list[_CheckpointChunk]
    vectors: list[list[float]]
    triples: list[GraphTriple]
    tables: list[tuple[str, list[str], list[dict[str, Any]]]]
    truncated: bool

    @model_validator(mode='after')
    def validate_vectors(self, info: ValidationInfo):
        if len(self.chunks) != len(self.vectors):
            raise ValueError('checkpoint chunk and vector counts differ')
        if self.vectors:
            dimension = (info.context or {}).get('dimension')
            if dimension is None:
                dimension = len(self.vectors[0])
            if not dimension or any(len(vector) != dimension for vector in self.vectors):
                raise ValueError('checkpoint vector dimension mismatch')
        return self

    @classmethod
    def read(cls, path: Path, *, dimension=None) -> _SourceCheckpoint | None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        return cls.model_validate_json(raw, strict=True, context={'dimension': dimension})


class _SnapshotEmbedding:
    """Supply the index dimension without loading the local model for cached vectors."""

    def __init__(self, settings, dimension):
        self.name = settings.embedding.provider
        self.dimension = dimension
        self._settings = settings
        self._provider = None

    def _load(self):
        if self._provider is None:
            from datamind.runtime import build_embedding

            self._provider = build_embedding(self._settings.embedding, fallback_llm=self._settings.llm)
        if self._provider.dimension != self.dimension:
            raise CapabilityError('kb', 'embedding dimension differs from the snapshot checkpoint')
        return self._provider

    async def embed_texts(self, texts):
        return await self._load().embed_texts(texts)

    async def embed_query(self, query):
        return await self._load().embed_query(query)

    async def aclose(self):
        close = getattr(self._provider, 'aclose', None)
        if callable(close):
            await close()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
    finally:
        temporary.unlink(missing_ok=True)


class Workspace(WorkspaceCatalog):
    """A profile-scoped snapshot catalog. Pass separate settings per tenant."""

    def __init__(self, settings, *, builder=None):
        self.settings = settings.model_copy(deep=True)
        self.root = (settings.data.storage_dir / 'workspace').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._catalog = self.root / 'catalog.sqlite3'
        self._builder = builder
        with self._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY, parent TEXT, state TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS head (singleton INTEGER PRIMARY KEY, revision TEXT, epoch INTEGER NOT NULL DEFAULT 0);
                INSERT OR IGNORE INTO head (singleton, revision) VALUES (1, NULL);
            ''')
            if 'epoch' not in {row[1] for row in conn.execute('PRAGMA table_info(head)')}:
                conn.execute('ALTER TABLE head ADD COLUMN epoch INTEGER NOT NULL DEFAULT 0')
        self._init_catalog()

    def _connect(self):
        return sqlite3.connect(self._catalog, timeout=30)

    @property
    def current_revision(self) -> str | None:
        with self._connect() as conn:
            return conn.execute('SELECT revision FROM head WHERE singleton=1').fetchone()[0]

    def status(self, revision: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute('SELECT id, parent, state, error FROM revisions WHERE id=?', (revision,)).fetchone()
        if row is None:
            raise CapabilityError('workspace', 'unknown revision')
        return dict(zip(('revision', 'parent', 'state', 'error'), row))

    def manifest(self, revision: str) -> dict[str, Any]:
        status = self.status(revision)
        if status['state'] != 'PUBLISHED':
            raise CapabilityError('workspace', 'revision was never published')
        return json.loads((self.root / 'revisions' / revision / 'manifest.json').read_text())

    def rollback(self, revision: str, *, expected_revision: str | None = None) -> dict[str, Any]:
        """Repoint to a retained complete snapshot without deleting history."""
        with self._lock():
            return self._rollback(revision, expected_revision=expected_revision)

    def _rollback(self, revision, *, expected_revision):
        self.manifest(revision)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            head = conn.execute('SELECT revision FROM head WHERE singleton=1').fetchone()[0]
            if expected_revision is not None and head != expected_revision:
                raise CapabilityError('workspace', 'publication conflict; current revision changed')
            conn.execute('UPDATE head SET revision=?, epoch=epoch+1 WHERE singleton=1', (revision,))
        return {'workspace_revision': revision, 'previous_revision': head}

    def _settings_for(self, revision: str):
        settings = self.settings.model_copy(deep=True)
        settings.data.base_dir = self.root / 'revisions' / revision
        settings.kb.ingest_mode = 'direct'
        return settings

    def _snapshot_component_paths(self, settings, surface):
        storage = settings.data.storage_dir
        if surface == 'kb':
            return (storage / settings.kb.vector_store,)
        if surface == 'db':
            return tuple(storage / name for name in ('demo.db', 'demo.db-wal', 'demo.db-shm', 'demo.db-journal'))
        return (storage / 'graph.json',)

    def _snapshot_files(self, revision, surface=None):
        settings = self._settings_for(revision)
        files = {}
        paths = ((settings.data.data_dir, settings.data.storage_dir) if surface is None
                 else self._snapshot_component_paths(settings, surface))
        for entry in paths:
            for path in sorted(entry.rglob('*')) if entry.is_dir() else (entry,):
                if path.is_file():
                    with path.open('rb') as handle:
                        files[path.relative_to(settings.data.base_dir).as_posix()] = (
                            hashlib.file_digest(handle, 'sha256').hexdigest())
        return files

    def _building_snapshots(self, parent):
        with self._connect() as conn:
            return conn.execute(
                "SELECT id FROM revisions WHERE parent IS ? AND state='BUILDING' ORDER BY rowid DESC",
                (parent,),
            ).fetchall()

    def _ready_snapshot(self, parent, epoch, plan):
        for (revision,) in self._building_snapshots(parent):
            root = self._settings_for(revision).data.base_dir
            try:
                receipt = json.loads((root / 'snapshot-ready.json').read_text())
                if receipt['plan'] != plan:
                    continue
                manifest = json.loads((root / 'manifest.json').read_text())
                fingerprint = _digest(json.dumps(manifest, sort_keys=True).encode())
                if (manifest['workspace_revision'] == revision
                        and manifest['parent_epoch'] == epoch
                        and receipt['manifest'] == fingerprint
                        and receipt['files'] == self._snapshot_files(revision)):
                    return manifest
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def _resume_snapshot(self, parent, epoch, plan, selected):
        for (revision,) in self._building_snapshots(parent):
            root = self._settings_for(revision).data.base_dir
            path = root / 'snapshot-progress.json'
            if not path.is_file() or (root / 'manifest.json').exists():
                continue
            lock = FileLock(root / '.build.lock', timeout=0)
            try:
                lock.acquire()
            except (LockTimeout, FileNotFoundError):
                continue
            keep = False
            try:
                if self.status(revision)['state'] != 'BUILDING' or (root / 'manifest.json').exists():
                    continue
                progress = json.loads(path.read_text())
                if progress['plan'] != plan or progress['parent_epoch'] != epoch:
                    continue
                reusable = {}
                for surface in selected & progress['components'].keys():
                    receipt = progress['components'][surface]
                    if surface == 'kb' and not (
                        isinstance(receipt.get('dimension'), int) and receipt['dimension'] > 0
                    ):
                        continue
                    if (isinstance(receipt.get('binding'), dict) and receipt.get('files')
                            and receipt['files'] == self._snapshot_files(revision, surface)):
                        reusable[surface] = receipt
                if reusable:
                    keep = True
                    return revision, lock, reusable
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
            finally:
                if not keep:
                    lock.release()
        return None

    def _publish_snapshot(self, revision, parent, epoch):
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current, current_epoch = conn.execute('SELECT revision, epoch FROM head WHERE singleton=1').fetchone()
            if (current, current_epoch) == (revision, epoch + 1):
                return  # A concurrent retry published this same sealed candidate.
            if (current, current_epoch) != (parent, epoch):
                raise CapabilityError('workspace', 'publication conflict; another build published first')
            changed = conn.execute(
                "UPDATE revisions SET state='PUBLISHED' WHERE id=? AND state='BUILDING'", (revision,)
            ).rowcount
            if changed != 1:
                raise CapabilityError('workspace', 'publication conflict; candidate is no longer building')
            conn.execute('UPDATE head SET revision=?, epoch=epoch+1 WHERE singleton=1', (revision,))

    async def _system(self, settings, surfaces, *, embedding=None):
        from datamind.data_plane import build_data_plane

        builder = self._builder if self._builder is not None else build_data_plane
        if embedding is None:
            return await builder(settings, enable=set(surfaces))
        return await builder(settings, enable=set(surfaces), embedding=embedding)

    async def migrate(self):
        from datamind.workspace_migration import migrate
        return await migrate(self)

    def _load_checkpoint(self, path: Path, *, source_uri: str, dimension=None) -> _SourceCheckpoint | None:
        try:
            return _SourceCheckpoint.read(path, dimension=dimension)
        except ValidationError:
            # Recheck under the workspace lock before moving a shared cache entry.
            with self._lock():
                try:
                    return _SourceCheckpoint.read(path, dimension=dimension)
                except ValidationError as exc:
                    quarantined = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.corrupt')
                    path.replace(quarantined)
                    issue = exc.errors(include_input=False, include_url=False, include_context=False)[0]
                    logger.warning('workspace_checkpoint_corrupt', extra={
                        'source_uri': source_uri,
                        'checkpoint': str(path), 'quarantined': str(quarantined),
                        'reason': issue['msg'], 'field': issue['loc'],
                    })
                    return None

    def _inventory(self, source: Path) -> list[dict[str, str]]:
        if not source.is_dir():
            raise CapabilityError('workspace', 'workspace path must be a directory')
        inventory = []
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if path.is_relative_to(self.root) or any(part.startswith('.') for part in relative.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in DOCUMENT_EXTS | TABLE_EXTS:
                continue
            if not path.resolve().is_relative_to(source):
                raise CapabilityError('workspace', f'symlink escapes workspace: {relative}')
            sha = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''): sha.update(block)
            inventory.append({'relative_path': relative.as_posix(), 'uri': str(path.resolve()),
                              'version': 'sha256:' + sha.hexdigest()})
        return inventory

    async def update(self, path: str, *, surfaces=('kb', 'graph', 'db'), relation_types=None, max_triples_per_file=200, graph_sources=None) -> dict[str, Any]:
        """Publish a generation and report logical compute/materialization costs.

        Metrics are per attempt, including unsuccessful attempts. Provider usage
        is reported only when available; absent token usage is not a zero cost.
        """
        metrics = {"sources_total": 0, "sources_reused": 0, "sources_computed": 0,
                   "sources_completed": 0, "embedding_calls": 0, "embedding_texts": 0,
                   "graph_extraction_calls": 0, "parsed_table_files": 0,
                   "inventory_ms": 0., "setup_ms": 0., "source_copy_ms": 0.,
                   "checkpoint_read_ms": 0., "checkpoint_write_ms": 0.,
                   "source_compute_ms": 0., "materialization_ms": 0., "publication_ms": 0.}
        parent = current_context()
        ctx = replace(parent, extra=dict(parent.extra)) if parent else RequestContext.new(profile=self.settings.data.profile)
        ctx.extra.update(nested_model_usage=[], embedding_usage=[])
        started = time.monotonic()
        try:
            from datamind.workspace_incremental import incremental_enabled, update_incremental
            update = update_incremental if incremental_enabled(self, surfaces) else Workspace._update
            with bind_context(ctx):
                result = await update(self, path, surfaces=surfaces, relation_types=relation_types,
                    max_triples_per_file=max_triples_per_file, metrics=metrics, graph_sources=graph_sources)
            metrics["status"] = "published"
            if self.settings.workspace.auto_gc:
                try:metrics["retention_cleanup"] = await self.collect(dry_run=False,compact=self.retention['mode']!='all')
                except Exception as exc:metrics["cleanup_error"] = type(exc).__name__
            return {**result, "metrics": metrics}
        except BaseException as exc:
            metrics.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            metrics["total_ms"] = round((time.monotonic() - started) * 1000, 3)
            usage = ctx.extra.get("nested_model_usage", [])
            embed_usage = ctx.extra.get("embedding_usage", [])
            metrics["model_usage_records"] = usage
            metrics["embedding_usage_records"] = embed_usage
            metrics["reported_model_input_tokens"] = sum(x.get("input_tokens", 0) for x in usage) if usage else None
            metrics["reported_model_output_tokens"] = sum(x.get("output_tokens", 0) for x in usage) if usage else None
            metrics["reported_embedding_tokens"] = sum(x.get("total_tokens", 0) for x in embed_usage) if embed_usage else None
            if parent is not None:
                parent.extra.setdefault("nested_model_usage", []).extend(usage)
                parent.extra.setdefault("embedding_usage", []).extend(embed_usage)
            revision = metrics.get('workspace_revision')
            if revision:
                # GC cannot remove a revision between checking it and persisting
                # metrics; a metrics write must never resurrect a pruned directory.
                with self._lock():
                    report = self.root / 'revisions' / revision / 'build_metrics.json'
                    if report.parent.is_dir():
                        try: _atomic_json(report, metrics)
                        except OSError: metrics['metrics_persisted'] = False

    async def _update(self, path: str, *, surfaces, relation_types, max_triples_per_file, metrics, graph_sources=None) -> dict[str, Any]:
        """Prepare and publish one generation, retaining the previous one on failure.

        Cached complete-source results are reused. Index materialization is currently
        full-snapshot; compute reuse does not imply constant storage or write cost.
        """
        if not 1 <= max_triples_per_file <= 10000:
            raise ConfigError('max_triples_per_file must be between 1 and 10000')
        selected = set(surfaces)
        if not selected or not selected <= {'kb', 'graph', 'db'}:
            raise ConfigError('workspace surfaces must be a nonempty subset of kb, graph, db')
        if ('graph' in selected and self.settings.graph.backend != 'networkx') or (
            'db' in selected and (self.settings.db.dialect != 'sqlite' or self.settings.db.dsn)
        ):
            raise ConfigError('workspace snapshots currently require local NetworkX and SQLite without an external DSN')
        source = Path(path).expanduser().resolve()
        stage_started = time.monotonic()
        inventory = await asyncio.to_thread(self._inventory, source)
        metrics['inventory_ms'] = (time.monotonic() - stage_started) * 1000
        metrics['sources_total'] = len(inventory)
        external = None
        if graph_sources is not None:
            external = {}
            expected = {item['uri']: item for item in inventory
                        if Path(item['uri']).suffix.lower() in DOCUMENT_EXTS - TABLE_EXTS}
            for entry in graph_sources:
                uri = entry['uri']
                if uri not in expected or uri in external or entry['version'] != expected[uri]['version']:
                    raise ConfigError('Graph source must match the prepared inventory and content hash')
                triples = [GraphTriple(**value) for value in entry['triples']]
                if len(triples) > max_triples_per_file:
                    raise ConfigError('Too many graph triples for source')
                for triple in triples:
                    if relation_types is not None and triple.relation not in relation_types:
                        raise ConfigError('Graph relation is outside the requested vocabulary')
                    triple.source = uri
                    triple.evidence = [GraphEvidence(
                        source_id='file:' + _digest(uri.encode()), source_version=entry['version'],
                        locator={'uri': uri}, method='extracted', pipeline_version='codex-explicit-v1')]
                external[uri] = {'triples': [t.model_dump(mode='json') for t in triples],
                                 'truncated': bool(entry.get('truncated', False))}
            if set(external) != set(expected):
                raise ConfigError('Supply triples (including explicit empty lists) for every document source')
        pipeline = {'version': 'workspace-v1', 'surfaces': sorted(selected),
                    'embedding': self.settings.embedding.model_dump(mode='json', exclude={'api_key'}),
                    'model': self.settings.llm.fallback_model or self.settings.llm.model,
                    'chunk_size': self.settings.retrieval.chunk_size,
                    'chunk_overlap': self.settings.retrieval.chunk_overlap,
                    'relation_types': relation_types, 'max_triples_per_file': max_triples_per_file}
        pipeline['graph_extraction'] = 'codex-explicit-v1' if external is not None else 'legacy-model'
        context = current_context()
        applied_skills = list(context.extra.get('applied_skills', {}).values()) if context else []
        pipeline['applied_skills'] = applied_skills
        # Persist only the fingerprint; configuration can include private endpoints.
        pipeline_hash = _digest(json.dumps(pipeline, sort_keys=True, default=str).encode())
        plan = _digest(json.dumps([
            str(source), inventory, pipeline_hash, external,
            {surface: getattr(self.settings, surface).model_dump(mode='json') for surface in selected},
        ], sort_keys=True).encode())
        with self._connect() as conn:
            parent, parent_epoch = conn.execute('SELECT revision, epoch FROM head WHERE singleton=1').fetchone()
        started = time.monotonic()
        ready = await asyncio.to_thread(self._ready_snapshot, parent, parent_epoch, plan)
        if ready is not None:
            if await asyncio.to_thread(self._inventory, source) != inventory:
                raise CapabilityError('workspace', 'workspace changed during build; retry')
            revision = ready['workspace_revision']
            self._publish_snapshot(revision, parent, parent_epoch)
            metrics.update(workspace_revision=revision, resumed_snapshot_candidate=True,
                           sources_reused=len(inventory), sources_completed=len(inventory),
                           publication_ms=(time.monotonic() - started) * 1000)
            return {**ready, 'reused_sources': len(inventory)}
        def checkpoint_path(item, source_id):
            cache_key = _digest(json.dumps([source_id, item['version'], pipeline_hash,
                external.get(item['uri']) if external is not None else None], sort_keys=True).encode())
            return self.root / 'checkpoints' / (cache_key + '.json')

        resumed = self._resume_snapshot(parent, parent_epoch, plan, selected)
        if resumed is None:
            revision = 'revision-' + uuid.uuid4().hex
            root = self._settings_for(revision).data.base_dir
            root.mkdir(parents=True)
            build_lock = FileLock(root / '.build.lock', timeout=0)
            build_lock.acquire()
            try:
                with self._connect() as conn:
                    conn.execute('INSERT INTO revisions VALUES (?, ?, ?, NULL)', (revision, parent, 'BUILDING'))
            except BaseException:
                build_lock.release()
                raise
            receipts = {}
        else:
            revision, build_lock, receipts = resumed
            metrics['resumed_snapshot_candidate'] = True
        metrics['workspace_revision'] = revision
        settings = self._settings_for(revision)
        system = None
        reused = 0
        manifest_sources = []
        truncated_sources = []
        bindings = {}

        def save_progress():
            _atomic_json(settings.data.base_dir / 'snapshot-progress.json', {
                'plan': plan, 'parent_epoch': parent_epoch, 'components': receipts,
            })

        async def seal(surface, binding, **details):
            bindings[surface] = binding
            files = await asyncio.to_thread(self._snapshot_files, revision, surface)
            if files:
                receipts[surface] = {'files': files, 'binding': binding, **details}
                save_progress()

        try:
            from datamind.data_plane import build_data_plane

            if 'kb' in receipts:
                for item in inventory:
                    if Path(item['uri']).suffix.lower() not in DOCUMENT_EXTS:
                        continue
                    source_id = 'file:' + _digest(item['uri'].encode())
                    if self._load_checkpoint(checkpoint_path(item, source_id), source_uri=item['uri'],
                                             dimension=receipts['kb']['dimension']) is None:
                        del receipts['kb']
                        break
            restored = set(receipts)
            if resumed is not None:
                metrics['resumed_surfaces'] = sorted(restored)
                bindings.update((surface, value['binding']) for surface, value in receipts.items())
                save_progress()
                for surface in selected - restored:
                    for path in self._snapshot_component_paths(settings, surface):
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink(missing_ok=True)
            active = selected - restored
            embedding = None
            default_builder = self._builder is None or self._builder is build_data_plane
            if 'kb' in active and default_builder and settings.embedding.provider == 'huggingface':
                dimension = settings.embedding.dimension
                if dimension is None:
                    started = time.monotonic()
                    for item in inventory:
                        source_id = 'file:' + _digest(item['uri'].encode())
                        try:
                            cached = _SourceCheckpoint.read(checkpoint_path(item, source_id))
                        except ValidationError:
                            continue
                        if cached and cached.vectors:
                            dimension = len(cached.vectors[0])
                            break
                    metrics['checkpoint_read_ms'] += (time.monotonic() - started) * 1000
                if dimension is not None:
                    embedding = _SnapshotEmbedding(settings, dimension)
            stage_started = time.monotonic()
            system = await self._system(settings, active, embedding=embedding)
            metrics['setup_ms'] = (time.monotonic() - stage_started) * 1000
            ingest = system.services.ingest
            dimension = system.services.kb.embedding.dimension if 'kb' in active else receipts.get('kb', {}).get('dimension')
            (settings.data.data_dir / 'uploads').mkdir(parents=True, exist_ok=True)
            graph_triples = []
            last_source = {
                surface: max((i for i, item in enumerate(inventory)
                              if Path(item['uri']).suffix.lower() in extensions), default=len(inventory) - 1)
                for surface, extensions in (('kb', DOCUMENT_EXTS), ('db', TABLE_EXTS)) if surface in active
            }

            async def seal_kb():
                await seal('kb', {'provider': settings.kb.vector_store,
                    'path': str(settings.data.storage_dir.relative_to(settings.data.base_dir)),
                    'version': getattr(system.services.kb.vector_store, 'latest_version', None)},
                    dimension=dimension)

            async def seal_db():
                db = system.services.db
                db_path = Path(db.engine.url.database)
                await db.aclose()
                system.services.db = None
                await seal('db', {'provider': 'sqlite', 'path': str(db_path.relative_to(settings.data.base_dir))})

            for index, item in enumerate(inventory):
                stage_started = time.monotonic()
                original = Path(item['uri'])
                snapshot = settings.data.data_dir / 'uploads' / item['relative_path']
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                raw = await asyncio.to_thread(original.read_bytes)
                if 'sha256:' + _digest(raw) != item['version']:
                    raise CapabilityError('workspace', 'source changed during build; retry')
                snapshot.write_bytes(raw)
                metrics['source_copy_ms'] += (time.monotonic() - stage_started) * 1000
                source_id = 'file:' + _digest(item['uri'].encode())
                cache_path = checkpoint_path(item, source_id)
                stage_started = time.monotonic()
                checkpoint = self._load_checkpoint(cache_path, source_uri=item['uri'], dimension=dimension)
                if checkpoint is not None:
                    reused += 1
                    metrics['sources_reused'] += 1
                    metrics['checkpoint_read_ms'] += (time.monotonic() - stage_started) * 1000
                else:
                    prepared = {'chunks': [], 'vectors': [], 'triples': [], 'tables': [], 'truncated': False}
                    is_table = original.suffix.lower() in TABLE_EXTS
                    text = (await asyncio.to_thread(extract_document, snapshot)).text if original.suffix.lower() in DOCUMENT_EXTS else ''
                    if 'kb' in active and text:
                        parts = _split_text(text, chunk_size=settings.retrieval.chunk_size,
                                            chunk_overlap=settings.retrieval.chunk_overlap)
                        prepared['chunks'] = [
                            {'id': _hash(part, item['uri'], ordinal=i), 'text': part,
                             'metadata': {'source': item['uri'], 'source_id': source_id,
                                          'source_version': item['version'], '_chunk_ordinal': i}}
                            for i, part in enumerate(parts)
                        ]
                        if parts:
                            metrics['embedding_calls'] += 1
                            metrics['embedding_texts'] += len(parts)
                            prepared['vectors'] = await system.services.kb.embedding.embed_texts(parts)
                    if 'graph' in selected and external is not None:
                        prepared.update(external.get(item['uri'], {'triples': [], 'truncated': False}))
                    if 'graph' in selected and external is None and text and not is_table:
                        parts = _split_text(text, chunk_size=settings.retrieval.chunk_size, chunk_overlap=0)
                        for i, part in enumerate(parts, 1):
                            remaining = max_triples_per_file - len(prepared['triples'])
                            if remaining <= 0:
                                prepared['truncated'] = True
                                break
                            metrics['graph_extraction_calls'] += 1
                            triples = await ingest._extract_graph_triples(
                                text=part, max_triples=min(30, remaining), source=item['uri'],
                                source_version=item['version'], locator={'part': i},
                                relation_types=relation_types,
                            )
                            prepared['triples'].extend(t.model_dump(mode='json') for t in triples)
                            if len(triples) >= min(30, remaining):
                                prepared['truncated'] = True
                    if 'db' in selected and is_table:
                        metrics['parsed_table_files'] += 1
                        prepared['tables'] = await asyncio.to_thread(extract_tabular, snapshot)
                    metrics['source_compute_ms'] += (time.monotonic() - stage_started) * 1000
                    metrics['sources_computed'] += 1
                    stage_started = time.monotonic()
                    _atomic_json(cache_path, prepared, overwrite=False)
                    checkpoint = _SourceCheckpoint.model_validate_json(
                        cache_path.read_bytes(), strict=True, context={'dimension': dimension})
                    metrics['checkpoint_write_ms'] += (time.monotonic() - stage_started) * 1000
                stage_started = time.monotonic()
                if checkpoint.truncated:
                    truncated_sources.append(item['uri'])
                if checkpoint.chunks and 'kb' in active:
                    await system.services.kb.vector_store.add(
                        ids=[c.id for c in checkpoint.chunks], texts=[c.text for c in checkpoint.chunks],
                        embeddings=checkpoint.vectors, metadatas=[c.metadata for c in checkpoint.chunks])
                graph_triples.extend(checkpoint.triples)
                for sheet, columns, rows in checkpoint.tables:
                    if not rows or 'db' not in active:
                        continue
                    table = 'ws_' + _digest((source_id + sheet).encode())[:24]
                    safe_cols, safe_rows = ingest._read_from_records(rows, source_file=item['uri'])
                    await ingest._ingest_rows(table=table, rows=safe_rows, safe_cols=safe_cols,
                                              if_exists='replace', source_file=item['uri'])
                manifest_sources.append({**item, 'source_id': source_id,
                    'snapshot': str(snapshot.relative_to(settings.data.base_dir)),
                    'checkpoint': str(cache_path.relative_to(self.root))})
                metrics['sources_completed'] += 1
                if last_source.get('kb') == index:
                    await seal_kb()
                if last_source.get('db') == index:
                    await seal_db()
                metrics['materialization_ms'] += (time.monotonic() - stage_started) * 1000
            stage_started = time.monotonic()
            if 'graph' in active:
                await system.services.graph.store.upsert_triples(graph_triples)
                await system.services.graph.store.persist()
                graph_path = settings.data.storage_dir / 'graph.json'
                await seal('graph', {'provider': 'networkx',
                    'path': str(graph_path.relative_to(settings.data.base_dir)),
                    'sha256': _digest(graph_path.read_bytes())})
            if 'kb' in active:
                await system.services.kb.record_incremental_ingest()
                if 'kb' not in bindings:
                    await seal_kb()
            elif 'kb' in restored:
                write_manifest_atomic(settings.data.storage_dir / 'kb_index_manifest.json', build_index_manifest(
                    data_dir=settings.data.data_dir, embedding_provider=settings.embedding.provider,
                    embedding_model=settings.embedding.model, dimension=dimension,
                    chunk_size=settings.retrieval.chunk_size, chunk_overlap=settings.retrieval.chunk_overlap,
                ))
            if 'db' in active and 'db' not in bindings:
                await seal_db()
            await system.aclose()
            system = None
            metrics['materialization_ms'] += (time.monotonic() - stage_started) * 1000
            stage_started = time.monotonic()
            files = await asyncio.to_thread(self._snapshot_files, revision)
            if await asyncio.to_thread(self._inventory, source) != inventory:
                raise CapabilityError('workspace', 'workspace changed during build; retry')
            manifest = {'workspace_revision': revision, 'parent_revision': parent, 'source_root': str(source), 'parent_epoch': parent_epoch,
                        'surfaces': sorted(selected), 'pipeline_hash': pipeline_hash,
                        'sources': manifest_sources, 'reused_sources': reused,
                        'snapshots': bindings, 'truncated_sources': truncated_sources,
                        'applied_skills': applied_skills}
            _atomic_json(settings.data.base_dir / 'snapshot-ready.json', {
                'plan': plan,
                'manifest': _digest(json.dumps(manifest, sort_keys=True).encode()),
                'files': files,
            })
            _atomic_json(settings.data.base_dir / 'manifest.json', manifest)
            self._publish_snapshot(revision, parent, parent_epoch)
            metrics['publication_ms'] = (time.monotonic() - stage_started) * 1000
            return manifest
        except BaseException as exc:
            with self._connect() as conn:
                conn.execute("UPDATE revisions SET state=?, error=? WHERE id=? AND state='BUILDING'",
                             ('FAILED', f'{type(exc).__name__}: {exc}', revision))
            raise
        finally:
            try:
                if system is not None:
                    await system.aclose()
            finally:
                build_lock.release()

    @asynccontextmanager
    async def reader(self, revision: str | None = None, *, read_retries: int = 1, enable=None):
        token,revision=self.acquire_lease(revision)
        if revision is None:
            raise CapabilityError('workspace', 'no published revision')
        system=None
        heartbeat=asyncio.create_task(self.keepalive(token))
        try:
            manifest=self.manifest(revision)
            selected=set(manifest['surfaces']) if enable is None else set(enable)
            if not selected<=set(manifest['surfaces']):
                raise CapabilityError('workspace','requested surface is absent from the pinned revision')
            if manifest.get('format_version')==2:
                from datamind.workspace_incremental import reader_system
                system,settings=await reader_system(self,manifest,selected)
                root=self.root
            else:
                settings=self._settings_for(revision)
                if manifest.get('snapshots',{}).get('kb'):
                    settings.kb.vector_store=manifest['snapshots']['kb']['provider']
                if manifest.get('snapshots',{}).get('db'):
                    settings.db.dialect=manifest['snapshots']['db']['provider'];settings.db.dsn=None
                settings.graph.backend='networkx'
                system=await self._system(settings,selected)
                root=settings.data.base_dir
            reader=WorkspaceReader(system,manifest,root,read_retries=read_retries,
                timeout_s=settings.agent.wall_clock_timeout_s)
            reader._system_settings=self.settings
            reader._lease=(self,token)
            yield reader
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat,return_exceptions=True)
            try:
                if system is not None: await system.aclose()
            finally: self.release_lease(token)


    async def query(self, message: str, *, revision: str | None = None, read_retries: int = 1, **kwargs):
        async with self.reader(revision, read_retries=read_retries) as reader:
            return await reader.query(message, **kwargs)


class WorkspaceReader:
    """Read-only facade fixed to a retained generation, including raw evidence."""

    def __init__(self, system, manifest, root, *, read_retries=1, timeout_s=300.):
        if not 0 <= read_retries <= 5:
            raise ValueError("read_retries must be between 0 and 5")
        self._system = system
        self._timeout = timeout_s
        self.manifest = manifest
        self.revision = manifest['workspace_revision']
        mapping = {s['uri']: root / s['snapshot'] for s in manifest['sources']}
        tools = system.read_tools
        system.tools = tools
        self.tools = tools
        raw = tools.get('raw_file_read')
        original = raw.handler

        async def read_snapshot(path: str, **kwargs):
            resolved = mapping.get(path)
            if resolved is None:
                candidate = Path(path).expanduser().resolve()
                if candidate not in mapping.values():
                    raise CapabilityError('workspace', 'raw evidence must belong to the pinned revision')
                resolved = candidate
            return await original(path=str(resolved), **kwargs)
        tools.replace(replace(raw, handler=read_snapshot))

        inspect = tools.get('workspace_inspect')
        original_inspect = inspect.handler
        source_root = Path(manifest['source_root'])
        snapshot_root = root / 'data' / 'profiles' / system.profile / 'uploads'

        async def inspect_snapshot(path: str, **kwargs):
            candidate = Path(path).expanduser().resolve()
            if manifest.get('format_version')==2:
                if candidate!=source_root and not candidate.is_relative_to(source_root):
                    raise CapabilityError('workspace','inspection must belong to the pinned revision')
                items=[{'path':s['uri'],'relative_path':s['relative_path'],'sha256':s['version'].removeprefix('sha256:')}
                    for s in manifest['sources'] if Path(s['uri']).is_relative_to(candidate) or Path(s['uri'])==candidate]
                limit=kwargs.get('max_files',2000)
                return {'path':str(candidate),'files':items[:limit],'count':len(items[:limit]),'truncated':len(items)>limit,
                        'workspace_revision':self.revision}

            if candidate.is_relative_to(source_root):
                candidate = snapshot_root / candidate.relative_to(source_root)
            elif not candidate.is_relative_to(snapshot_root):
                raise CapabilityError('workspace', 'inspection must belong to the pinned revision')
            return await original_inspect(path=str(candidate), **kwargs)
        tools.replace(replace(inspect, handler=inspect_snapshot))

        if 'kb_search_at_version' in tools:
            historical = tools.get('kb_search_at_version')
            original_historical = historical.handler
            pinned = manifest['snapshots'].get('kb', {}).get('version')

            async def search_pinned(version: int, **kwargs):
                if version != pinned:
                    raise CapabilityError('workspace', 'select a workspace revision instead of another backend version')
                return await original_historical(version=version, **kwargs)
            tools.replace(replace(historical, handler=search_pinned))

        for name in tools.names():
            spec = tools.get(name)
            original_handler = spec.handler

            async def retry_read(_handler=original_handler, _name=name, **kwargs):
                return await recover_read(lambda: _handler(**kwargs), component=_name,
                                          retries=read_retries, manage_provider_retries=True)
            tools.replace(replace(spec, handler=retry_read))

    async def call(self, name: str, **kwargs):
        if hasattr(self,'_lease'):
            self._lease[0].renew_lease(self._lease[1])
        parent = current_context()
        if parent is not None:
            parent.extra.setdefault('recovery_events', [])
        context = replace(parent, extra=dict(parent.extra)) if parent else RequestContext.new(profile=self._system.profile)
        context.extra['workspace_revision'] = self.revision
        context.extra['query_deadline'] = time.monotonic() + self._timeout
        with bind_context(context):
            return await self._system.call(name, kwargs, context=context, read_only=True)

    async def query(self, message: str, **kwargs):
        context = RequestContext.new(profile=self._system.profile)
        context.extra['workspace_revision'] = self.revision
        context.extra['query_deadline'] = time.monotonic() + self._timeout
        with bind_context(context):
            async with asyncio.timeout(self._timeout):
                from datamind.host import CodexHost
                settings = self._system_settings
                host = CodexHost(settings, revision=self.revision, runtime_env=kwargs.pop('runtime_env', None))
                try:
                    result = await host.query(message, **kwargs)
                finally:
                    await host.aclose()
        recovery = result.get('recovery') or recovery_summary(context.extra.get('recovery_events', []))
        return {**result, 'workspace_revision': self.revision,
                'truncated_sources': self.manifest.get('truncated_sources', []),
                'workspace_read_retries': recovery['retries'],
                'recovery': recovery}
