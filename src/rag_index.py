"""
rag_index.py — Build and incrementally maintain the ChromaDB vector index.

What gets indexed:
  - summary.json  → one "summary" chunk per meeting (high-level, fast retrieval)
  - transcript.json → one chunk per speaker turn (fine-grained, who-said-what retrieval)

Metadata stored per chunk (enables filtering later):
  meeting_id, title, sentiment, call_type (guessed), date, date_int, speaker,
  tenant_id  ← NEW: multi-tenant isolation

PII handling:
  Text is scrubbed before embedding (emails, phones, SSNs, credit cards, IPs).
  Set PII_SCRUB_NAMES=true to also redact PERSON/ORG via spaCy NER.
  Set ANONYMIZE_SPEAKERS=true to replace speaker names with Speaker_1, Speaker_2…

Temporal / versioning metadata (also on every chunk):
  account_id, account_name      stable customer id from the email domain (src/accounts.py)
  occurred_at, occurred_ts      when the conversation happened
  conversation_sequence         n-th conversation with this account (1 = first)
  source_version, is_current_revision, supersedes_version,
  source_updated_at, content_hash, index_version, chunking_version

Content-aware upsert (the SQLite manifest, src/manifest.py, is the source of truth):
  new meeting id                     → version 1, inserted as a new timeline event
  same id + same content hash        → no action
  same id + changed content hash     → version N+1 inserted and marked current;
                                       older vectors kept for audit, is_current_revision=False
  meeting removed from the source    → tombstoned; its vectors deactivated

Usage:
  python -m src.rag_index                              # incremental update
  python -m src.rag_index --reset                      # wipe index + manifest and rebuild
  python -m src.rag_index --tenant acme               # tag chunks with tenant_id=acme
"""

import argparse
import json
import os
import re
from pathlib import Path

import hashlib
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.accounts import AccountRegistry, REGISTRY_PATH
from src.manifest import Manifest
from src.pii_scrub import scrub_text, build_speaker_map

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/data/dataset"))
CHROMA_DIR  = Path(os.getenv("CHROMA_DIR",  "/app/data/chroma"))
EMBED_MODEL  = "text-embedding-3-small"
CHUNKING_VERSION = "speaker-turn-v2"   # bump when chunk boundaries or chunk text change
COLLECTION = os.getenv("RAG_COLLECTION", "transcripts")

# Default tenant — override via --tenant CLI arg or TENANT_ID env var.
# In a real deployment each customer / business unit gets its own tenant_id
# and the API gateway injects it into every request so users only see their data.
DEFAULT_TENANT = os.getenv("TENANT_ID", "aegis")   # the employees in src/auth.py belong to tenant "aegis"


def _get_embedding_function():
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    print(f"  Using OpenAI embeddings ({EMBED_MODEL})")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)


def _guess_call_type(title: str) -> str:
    """Heuristic call-type classification from the meeting title."""
    t = title.lower()
    if re.search(r"support case|ticket|issue|help|error|bug|slow|outage", t):
        return "support"
    if re.search(r"aegis\s*/|renewal|contract|qbr|onboarding|demo|pilot", t):
        return "external"
    return "internal"


def _classify_access_level(title: str) -> str:
    """
    Assign an access_level to a meeting based on its title.
    This drives RBAC — only roles that include this level in their policy
    will be able to retrieve this chunk.

    Levels (from broadest to most restricted):
      public      — All Hands, company-wide announcements
      customer    — Customer-facing calls (Aegis / <Customer> meetings)
      support     — Support cases (#XXXX tickets)
      engineering — Internal eng meetings (standups, sprints, retros, roadmaps)
      incident    — Outage war rooms, escalations, urgent incidents
      compliance  — Internal SOC 2 / audit / Comply v2 release meetings
      executive   — Strategic / competitive / financial (win/loss, all-hands strategy)
    """
    t = title.lower()

    # Company-wide meetings — visible to every role
    if re.search(r"\ball hands\b", t):
        return "public"

    # Incidents first — highest operational urgency
    if re.search(r"incident|urgent|escalation|war room|outage.*root cause|"
                 r"outage.*post.incident|outage.*remediation|outage.*escalation|"
                 r"outage.*bridge|pipeline failure|detect outage|"
                 r"loss of threat visibility|dashboard down", t):
        return "incident"

    # Support cases
    if re.search(r"support case\s*#\d+", t):
        return "support"

    # Customer-facing calls
    if re.search(r"^aegis\s*/", t):
        return "customer"

    # Internal compliance / audit / product compliance releases
    if re.search(r"soc\s*2|iso\s*27001|comply\s*v2|audit prep|audit preparation|"
                 r"pci\s*dss|hipaa|gdpr|compliance review", t):
        return "compliance"

    # Engineering internal
    if re.search(r"standup|sprint|retro|roadmap|product sync|design review|"
                 r"scalability|reliability|launch readiness|launch day|"
                 r"adoption metrics|detect team|identity team|protect team|"
                 r"engineering|deployment plan|detect reliability", t):
        return "engineering"

    # Executive / strategic
    if re.search(r"win.?loss|competitive|strategy|30.day review", t):
        return "executive"

    # Everything else defaults to internal/engineering visibility
    return "engineering"


def _load_summary_chunk(meeting_dir: Path, tenant_id: str) -> tuple[str, dict] | None:
    """Return (text, metadata) for the summary chunk, or None if files missing."""
    info_path    = meeting_dir / "meeting-info.json"
    summary_path = meeting_dir / "summary.json"
    if not info_path.exists() or not summary_path.exists():
        return None

    info    = json.loads(info_path.read_text())
    summary = json.loads(summary_path.read_text())

    meeting_id = info.get("meetingId", meeting_dir.name)
    title      = info.get("title", "Unknown")
    date       = info.get("startTime", "")[:10]  # YYYY-MM-DD (display only)
    date_int   = int(date.replace("-", "")) if date else 0  # YYYYMMDD int for $gte/$lte filtering
    sentiment  = summary.get("overallSentiment", "unknown")
    topics     = ", ".join(summary.get("topics", []))
    actions    = "\n".join(f"- {a}" for a in summary.get("actionItems", []))
    key_moments = "\n".join(
        f"- [{m.get('type','?')}] {m.get('text','')}"
        for m in summary.get("keyMoments", [])
    )

    raw_text = (
        f"Meeting: {title}\n"
        f"Date: {date}  |  Sentiment: {sentiment}\n"
        f"Topics: {topics}\n\n"
        f"Summary:\n{summary.get('summary', '')}\n\n"
        f"Action Items:\n{actions}\n\n"
        f"Key Moments:\n{key_moments}"
    ).strip()

    # ── PII scrub before embedding ────────────────────────────────────────────
    text = scrub_text(raw_text)

    metadata = {
        "chunk_type":   "summary",
        "meeting_id":   meeting_id,
        "title":        title,
        "date":         date,                      # "YYYY-MM-DD" string for display
        "date_int":     date_int,                  # YYYYMMDD int for $gte/$lte filtering
        "sentiment":    sentiment,
        "call_type":    _guess_call_type(title),
        "topics":       topics,
        "tenant_id":    tenant_id,                 # multi-tenant isolation
        "access_level": _classify_access_level(title),  # RBAC access control
    }
    return text, metadata


def _load_transcript_chunks(meeting_dir: Path, info: dict, tenant_id: str) -> list[tuple[str, dict]]:
    """Return list of (text, metadata) — one per speaker turn."""
    tx_path = meeting_dir / "transcript.json"
    if not tx_path.exists():
        return []

    transcript = json.loads(tx_path.read_text())
    sentences  = transcript.get("data", [])
    if not sentences:
        return []

    meeting_id = info.get("meetingId", meeting_dir.name)
    title      = info.get("title", "Unknown")
    date       = info.get("startTime", "")[:10]
    date_int   = int(date.replace("-", "")) if date else 0
    call_type  = _guess_call_type(title)

    # Build speaker map for optional anonymisation (ANONYMIZE_SPEAKERS=true)
    all_speakers = list({s.get("speaker_name", "") for s in sentences if s.get("speaker_name")})
    speaker_map  = build_speaker_map(all_speakers)

    # Group consecutive sentences by the same speaker into one chunk
    chunks: list[tuple[str, dict]] = []
    current_speaker = None
    current_lines: list[str] = []
    chunk_start = 0.0

    def flush(speaker, lines, start):
        if not lines:
            return
        raw = f"[{speaker}]: " + " ".join(lines)
        # ── PII scrub (pass speaker_map so names are anonymised consistently)
        text = scrub_text(raw, speaker_map=speaker_map)
        display_speaker = speaker_map.get(speaker, speaker)  # anonymised if map active
        meta = {
            "chunk_type":   "transcript",
            "meeting_id":   meeting_id,
            "title":        title,
            "date":         date,
            "date_int":     date_int,
            "speaker":      display_speaker,
            "call_type":    call_type,
            "time_start":   str(round(start, 1)),
            "tenant_id":    tenant_id,                      # multi-tenant isolation
            "access_level": _classify_access_level(title),  # RBAC access control
        }
        chunks.append((text, meta))

    for sent in sentences:
        speaker = sent.get("speaker_name", "Unknown")
        text    = sent.get("sentence", "").strip()
        t       = sent.get("time", 0.0)
        if not text:
            continue
        if speaker != current_speaker:
            flush(current_speaker, current_lines, chunk_start)
            current_speaker = speaker
            current_lines   = [text]
            chunk_start     = t
        else:
            current_lines.append(text)

    flush(current_speaker, current_lines, chunk_start)
    return chunks


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def indexing_settings() -> dict:
    """
    Everything OTHER than the source files that changes what a chunk contains or how it is
    tagged. If any of it changes, existing vectors are stale even though content_hash() is not.
    (The embedding model is tracked per collection instead: vectors from two models cannot
    share one collection.)
    """
    import inspect
    from src import pii_scrub
    code = inspect.getsource(_classify_access_level) + inspect.getsource(_guess_call_type)
    return {
        "chunking_version": CHUNKING_VERSION,
        "anonymize_speakers": _env_flag("ANONYMIZE_SPEAKERS"),
        "pii_scrub_names": _env_flag("PII_SCRUB_NAMES"),
        "pii_patterns": hashlib.sha256("|".join(f"{l}:{p.pattern}" for l, p in pii_scrub._PATTERNS)
                                       .encode()).hexdigest()[:12],
        "tagging_rules": hashlib.sha256(code.encode()).hexdigest()[:12],   # RBAC access_level + call_type
    }


def meeting_fingerprint(settings: dict, account_id: str, access_level: str) -> str:
    """Per-meeting indexing fingerprint: global settings + the tags this meeting gets."""
    blob = json.dumps({**settings, "account_id": account_id, "access_level": access_level}, sort_keys=True)
    return "fp:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def content_hash(meeting_dir: Path) -> str:
    """sha256 over the canonical JSON of every source file that feeds the chunks."""
    h = hashlib.sha256()
    for name in ("meeting-info.json", "summary.json", "transcript.json"):
        p = meeting_dir / name
        h.update(name.encode())
        if p.exists():
            h.update(json.dumps(json.loads(p.read_text()), sort_keys=True, separators=(",", ":")).encode())
    return "sha256:" + h.hexdigest()


def _source_updated_at(meeting_dir: Path) -> str:
    mtimes = [p.stat().st_mtime for p in meeting_dir.glob("*.json")]
    return datetime.fromtimestamp(max(mtimes) if mtimes else 0, tz=timezone.utc).isoformat()


def chunk_id(tenant_id: str, meeting_id: str, version: int, kind: str, i: int | None = None) -> str:
    base = f"{tenant_id}::{meeting_id}::v{version}"
    return f"{base}::summary" if kind == "summary" else f"{base}::tx::{i}"


def _occurred(info: dict) -> tuple[str, int, int]:
    raw = info.get("startTime", "") or ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "", 0, 0
    return dt.isoformat(), int(dt.timestamp()), int(dt.strftime("%Y%m%d"))


def _collection(reset_all: bool):
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset_all:
        try:
            client.delete_collection(COLLECTION)
            print(f"  Collection '{COLLECTION}' deleted (all tenants).")
        except Exception:
            pass
    return client.get_or_create_collection(name=COLLECTION, embedding_function=_get_embedding_function(),
                                           metadata={"hnsw:space": "cosine"})


def _set_metadata(collection, where: dict, changes: dict) -> int:
    """Merge `changes` into the metadata of every chunk matching `where`."""
    got = collection.get(where=where, include=["metadatas"])
    if not got["ids"]:
        return 0
    collection.update(ids=got["ids"], metadatas=[{**(m or {}), **changes} for m in got["metadatas"]])
    return len(got["ids"])


def _w(tenant_id: str, meeting_id: str, *extra: dict) -> dict:
    return {"$and": [{"tenant_id": {"$eq": tenant_id}}, {"meeting_id": {"$eq": meeting_id}}, *extra]}


def _tenant_has_vectors(collection, tenant_id: str) -> bool:
    return bool(collection.get(where={"tenant_id": {"$eq": tenant_id}}, limit=1, include=[])["ids"])


def build_index(reset: bool = False, tenant_id: str = DEFAULT_TENANT, *, reset_all: bool = False,
                dataset_dir: Path | None = None, collection=None, manifest: Manifest | None = None,
                registry: AccountRegistry | None = None, today: str | None = None,
                batch_size: int = 500) -> dict:
    """
    Incrementally sync one tenant's slice of the vector index with the dataset.

      reset      delete THIS tenant's vectors + manifest rows, then rebuild them (other tenants untouched)
      reset_all  delete the whole collection + manifest (every tenant), then rebuild this tenant

    Writes are two-phase (see src/manifest.py) and idempotent: re-running after a crash
    finishes whatever the crash interrupted. `collection` / `manifest` / `registry` are
    injectable for tests.
    """
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    production = collection is None          # tests inject fakes and must not touch data/chroma
    manifest = manifest or Manifest(CHROMA_DIR / "manifest.sqlite")
    if reset_all:
        manifest.reset_all()
    collection = collection if collection is not None else _collection(reset_all)
    registry = registry or AccountRegistry.load(REGISTRY_PATH)

    # One embedding model per collection: mixing vector spaces silently breaks search.
    known = manifest.collection_embed_model(COLLECTION)
    if known and known != EMBED_MODEL:
        raise SystemExit(f"Collection '{COLLECTION}' holds {known} vectors but EMBED_MODEL is {EMBED_MODEL}. "
                         f"Build a new collection (RAG_COLLECTION=transcripts_v2 python -m src.rag_index), evaluate it, "
                         f"then switch RAG_COLLECTION — or rebuild everything with --reset-all.")
    if reset:
        n = len(collection.get(where={"tenant_id": {"$eq": tenant_id}}, include=[])["ids"])
        collection.delete(where={"tenant_id": {"$eq": tenant_id}})
        manifest.reset(tenant_id)
        print(f"  Tenant '{tenant_id}' reset: {n} vectors deleted (other tenants untouched).")
    elif not manifest.has_tenant(tenant_id) and _tenant_has_vectors(collection, tenant_id):
        raise SystemExit(f"Vectors for tenant '{tenant_id}' exist but the manifest has no record of them "
                         f"(legacy/unversioned index). Run once with --reset --tenant {tenant_id}.")
    if not known:
        manifest.set_collection_embed_model(COLLECTION, EMBED_MODEL)

    settings = indexing_settings()
    index_version = manifest.start_run(tenant_id, EMBED_MODEL, CHUNKING_VERSION, today=today)
    print(f"  Tenant: {tenant_id}   index_version: {index_version}   chunking: {CHUNKING_VERSION}")
    try:
        stats = _sync(collection, manifest, registry, dataset_dir, tenant_id, index_version, settings,
                      production, batch_size)
    except BaseException as e:
        manifest.fail_run(index_version, f"{type(e).__name__}: {e}")
        raise
    total = collection.count()
    manifest.finish_run(index_version, total_chunks=total, **stats)
    if production:
        _write_manifest(total, stats, tenant_id, reset or reset_all, index_version, settings)
    print(f"\n✅  Index {index_version} — {total} chunks | " + ", ".join(f"{k} {v}" for k, v in stats.items())
          + f" (tenant={tenant_id})")
    return {"index_version": index_version, "total_chunks": total, **stats}


def _sync(collection, manifest: Manifest, registry: AccountRegistry, dataset_dir: Path, tenant_id: str,
          index_version: str, settings: dict, production: bool, batch_size: int) -> dict:
    from src.accounts import external_domains

    # ── Pass 1: what is in the source, and what has to happen to each meeting? ─
    meeting_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    print(f"  Found {len(meeting_dirs)} meetings in {dataset_dir}")
    plan: list[dict] = []
    seen: set[str] = set()
    new_domains = 0
    for d in meeting_dirs:
        info_path = d / "meeting-info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        mid = info.get("meetingId", d.name)
        seen.add(mid)
        acct = registry.for_meeting(info)
        if acct is None:
            doms = external_domains(info)
            if len(doms) == 1:                      # unseen customer domain → register it
                acct = registry.add_domain(doms[0])
                new_domains += 1
        title = info.get("title", "Unknown")
        access_level = _classify_access_level(title)
        occurred_at, occurred_ts, date_int = _occurred(info)
        h = content_hash(d)
        fp = meeting_fingerprint(settings, acct.account_id if acct else "", access_level)
        cur = manifest.current_version_row(tenant_id, mid)      # committed + current only

        if cur and cur["content_hash"] == h:
            healthy = len(collection.get(where=_w(tenant_id, mid, {"source_version": {"$eq": cur["version"]}},
                                                  {"is_current_revision": {"$eq": True}}), include=[])["ids"]) \
                == cur["chunk_count"]
            if cur["fingerprint"] == fp and healthy:
                action = "unchanged"
            else:
                action = "reembedded" if cur["fingerprint"] != fp else "repaired"
            version = cur["version"]
        else:
            action = "updated" if cur else "inserted"
            version = manifest.pending_version(tenant_id, mid, h) or manifest.next_version(tenant_id, mid)

        manifest.upsert_meeting(tenant_id, mid, account_id=acct.account_id if acct else "", title=title,
                                access_level=access_level, occurred_at=occurred_at, occurred_ts=occurred_ts,
                                date_int=date_int)
        plan.append({"dir": d, "info": info, "meeting_id": mid, "account": acct, "action": action, "hash": h,
                     "fingerprint": fp, "version": version, "prev_version": cur["version"] if cur else None,
                     "occurred_at": occurred_at, "occurred_ts": occurred_ts})

    for acct in registry.accounts.values():
        manifest.upsert_account(tenant_id, acct.account_id, acct.name, acct.aliases)
    if new_domains and production:
        registry.save(REGISTRY_PATH)
        print(f"  ⚠ {new_domains} new customer domain(s) auto-registered — review {REGISTRY_PATH}")

    # Deleted from source: deactivate vectors FIRST, then tombstone (a crash in between just repeats it)
    deleted = sorted(manifest.active_meeting_ids(tenant_id) - seen)
    for mid in deleted:
        _set_metadata(collection, _w(tenant_id, mid), {"is_current_revision": False, "deleted": True})
        manifest.tombstone(tenant_id, mid)

    counts = {"inserted": 0, "updated": 0, "reembedded": 0, "repaired": 0, "unchanged": 0, "deleted": len(deleted)}
    pending_writes: list[tuple] = []        # (ids, texts, metas, plan_item)

    def flush():
        """Write the batch, then switch each meeting over and commit it in the manifest."""
        if not pending_writes:
            return
        ids = [i for w in pending_writes for i in w[0]]
        collection.upsert(ids=ids, documents=[t for w in pending_writes for t in w[1]],
                          metadatas=[m for w in pending_writes for m in w[2]])
        for new_ids, _, metas, p in pending_writes:
            mid, v = p["meeting_id"], p["version"]
            # drop chunks of this version that the new chunking no longer produces
            stale = [i for i in collection.get(where=_w(tenant_id, mid, {"source_version": {"$eq": v}}),
                                               include=[])["ids"] if i not in set(new_ids)]
            if stale:
                collection.delete(ids=stale)
            if p["action"] in ("inserted", "updated"):
                # switch: older revisions off, this one on, THEN commit (crash before commit → redone next run)
                _set_metadata(collection, _w(tenant_id, mid, {"source_version": {"$ne": v}}),
                              {"is_current_revision": False})
                _set_metadata(collection, _w(tenant_id, mid, {"source_version": {"$eq": v}}),
                              {"is_current_revision": True})
                manifest.commit_version(tenant_id, mid, v)
            else:  # reembedded / repaired: same revision, written in place as current
                manifest.update_fingerprint(tenant_id, mid, v, p["fingerprint"], len(new_ids),
                                            index_version, CHUNKING_VERSION)
        pending_writes.clear()

    # Sequences are computed over committed meetings; new ones get theirs after commit (see below).
    for p in plan:
        mid, version = p["meeting_id"], p["version"]
        counts[p["action"]] += 1
        if p["action"] == "unchanged":
            continue

        acct = p["account"]
        is_new_revision = p["action"] in ("inserted", "updated")
        temporal = {
            "account_id": acct.account_id if acct else "",
            "account_name": acct.name if acct else "",
            "occurred_at": p["occurred_at"],
            "occurred_ts": p["occurred_ts"],
            "conversation_sequence": 0,                    # filled in after commit
            "source_version": version,
            "is_current_revision": not is_new_revision,    # new revisions stay invisible until switched on
            "supersedes_version": p["prev_version"] or 0,
            "source_updated_at": _source_updated_at(p["dir"]),
            "content_hash": p["hash"],
            "index_fingerprint": p["fingerprint"],
            "index_version": index_version,
            "chunking_version": CHUNKING_VERSION,
        }
        chunks = []
        summary = _load_summary_chunk(p["dir"], tenant_id)
        if summary:
            chunks.append((chunk_id(tenant_id, mid, version, "summary"), *summary))
        for i, (text, meta) in enumerate(_load_transcript_chunks(p["dir"], p["info"], tenant_id)):
            chunks.append((chunk_id(tenant_id, mid, version, "tx", i), text, meta))

        if is_new_revision:
            manifest.begin_version(tenant_id, mid, version=version, content_hash=p["hash"],
                                   fingerprint=p["fingerprint"], source_updated_at=temporal["source_updated_at"],
                                   chunk_count=len(chunks), index_version=index_version,
                                   chunking_version=CHUNKING_VERSION)
        pending_writes.append(([c[0] for c in chunks], [c[1] for c in chunks],
                               [{**c[2], **temporal} for c in chunks], p))
        if sum(len(w[0]) for w in pending_writes) >= batch_size:
            flush()
    flush()

    # conversation_sequence over committed meetings; push to every current chunk whose value differs
    manifest.recompute_sequences(tenant_id)
    for p in plan:
        row = manifest.get_meeting(tenant_id, p["meeting_id"]) or {}
        seq = row.get("conversation_sequence", 0)
        cur = _w(tenant_id, p["meeting_id"], {"is_current_revision": {"$eq": True}},
                 {"conversation_sequence": {"$ne": seq}})
        _set_metadata(collection, cur, {"conversation_sequence": seq})
    return counts


def _write_manifest(total_chunks: int, counts: dict, tenant_id: str, reset: bool,
                    index_version: str, settings: dict) -> None:
    """
    data/chroma/index_manifest.json — read by RAGPipeline (index_version, stamped on
    every trace and eval run) and by src/monitor.py (index freshness alert).
    """
    if not CHROMA_DIR.exists():
        return
    manifest = {
        "index_version": index_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "embed_model": EMBED_MODEL,
        "indexing_settings": settings,
        "total_chunks": total_chunks,
        "last_tenant": tenant_id,
        "reset": reset,
        "last_run": counts,
    }
    (CHROMA_DIR / "index_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally sync a tenant's vectors with the dataset")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant ID to tag all chunks with")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--reset", action="store_true", help="Delete and rebuild THIS tenant only")
    g.add_argument("--reset-all", action="store_true",
                   help="Delete the whole collection + manifest (ALL tenants), then rebuild this tenant")
    args = parser.parse_args()
    build_index(reset=args.reset, reset_all=args.reset_all, tenant_id=args.tenant)
