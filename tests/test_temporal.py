"""
Account + temporal behaviour, end to end:
  real indexer (src/rag_index.build_index) → SQLite manifest → FakeCollection → RAGPipeline.

A tiny synthetic dataset is written to tmp_path so we can also exercise what the
real dataset never contains: corrected transcripts, deleted meetings, late-arriving
earlier calls, and look-alike customer names.
"""

import json
import shutil
from datetime import date

import pytest

from src.accounts import AccountRegistry, build_registry
from src.manifest import Manifest
from src.query_understanding import understand
from src.rag_index import build_index
from src.rag_pipeline import Principal, RAGConfig, RAGPipeline
from src.telemetry import Tracer, read_jsonl
from tests.fakes import FakeCollection, FakeLLM

T = "t1"
REF = date(2026, 10, 13)


def write_meeting(root, mid, title, start, emails, summary, turns):
    d = root / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meeting-info.json").write_text(json.dumps(
        {"meetingId": mid, "title": title, "startTime": start, "allEmails": emails}))
    (d / "summary.json").write_text(json.dumps(
        {"summary": summary, "overallSentiment": "neutral", "topics": [], "actionItems": [], "keyMoments": []}))
    (d / "transcript.json").write_text(json.dumps(
        {"data": [{"speaker_name": s, "sentence": t, "time": float(i * 10)} for i, (s, t) in enumerate(turns)]}))
    return d


@pytest.fixture
def world(tmp_path):
    ds = tmp_path / "dataset"
    A = ["kim@aegiscloud.com", "eve@alphahealth.com"]
    write_meeting(ds, "A1", "Aegis / Alpha Health - Renewal Kickoff", "2026-02-10T10:00:00Z", A,
                  "Alpha Health renewal status: pricing proposal pending, decision expected in March.",
                  [("Eve", "Our renewal status is open, we want a pricing proposal."), ("Kim", "We will send pricing.")])
    write_meeting(ds, "A2", "Support Case #12 - Alpha Health Outage", "2026-03-12T10:00:00Z", A,
                  "Alpha Health reported an outage and loss of visibility; escalated.",
                  [("Eve", "We have an outage and zero visibility right now."), ("Kim", "Escalating now.")])
    write_meeting(ds, "A3", "Aegis / Alpha Health - Renewal Signed", "2026-04-20T10:00:00Z", A,
                  "Alpha Health renewal status: signed with flat pricing and a seat expansion.",
                  [("Eve", "Renewal status is final, we signed with flat pricing."), ("Kim", "Great, seat expansion added.")])
    write_meeting(ds, "I1", "Aegis / Ironclad Financial - Check-in", "2026-03-01T10:00:00Z",
                  ["kim@aegiscloud.com", "rob@ironclad.com"],
                  "Ironclad Financial renewal conditions: post-mortem and SLA uptime terms.",
                  [("Rob", "Our renewal conditions are a post-mortem and SLA uptime terms.")])
    write_meeting(ds, "W1", "Aegis / Ironworks Corp - Vendor Comparison", "2026-03-02T10:00:00Z",
                  ["kim@aegiscloud.com", "cat@ironworks.com"],
                  "Ironworks Corp renewal conditions: evaluating a competitor, wants RCA.",
                  [("Cat", "Our renewal conditions depend on the competitor evaluation.")])
    write_meeting(ds, "N1", "Weekly Engineering Standup", "2026-03-05T10:00:00Z", ["kim@aegiscloud.com"],
                  "Internal standup about Alpha Health outage remediation.",
                  [("Kim", "Alpha Health outage remediation is on track.")])

    registry = build_registry(ds)
    manifest = Manifest(":memory:")
    coll = FakeCollection()

    def index(**kw):
        return build_index(tenant_id=T, dataset_dir=ds, collection=coll, manifest=manifest,
                           registry=registry, today="2026-10-13", **kw)

    def pipeline(**cfg):
        config = RAGConfig(top_k=cfg.pop("top_k", 6), neighbor_turns=0, index_version="test",
                           reference_date="2026-10-13", **cfg)
        return RAGPipeline(config, collection=coll, llm=FakeLLM("ok [1]"), manifest=manifest,
                           tracer=Tracer(path=str(tmp_path / "traces.jsonl"), sample_rate=1.0))

    index()
    return {"ds": ds, "coll": coll, "manifest": manifest, "registry": registry, "index": index,
            "pipeline": pipeline, "tmp": tmp_path}


def meetings(r):
    return [c.meeting_id for c in r.chunks]


# ── Accounts ──────────────────────────────────────────────────────────────────

def test_accounts_come_from_email_domain(world):
    reg: AccountRegistry = world["registry"]
    assert set(reg.accounts) == {"alpha-health", "ironclad-financial", "ironworks-corp"}
    assert reg.for_meeting({"allEmails": ["x@aegiscloud.com"]}) is None   # internal meeting


def test_similar_names_resolve_separately(world):
    accts = world["manifest"].accounts(T)
    assert understand("What did Ironclad want?", accts, reference_date=REF).account_id == "ironclad-financial"
    assert understand("What did Ironworks want?", accts, reference_date=REF).account_id == "ironworks-corp"
    both = understand("Compare Ironclad and Ironworks", accts, reference_date=REF)
    assert both.account_id is None and both.account_resolution == "multiple"


# ── Indexing / versioning ─────────────────────────────────────────────────────

def test_first_index_metadata(world):
    m = world["manifest"]
    assert [x["meeting_id"] for x in m.timeline(T, "alpha-health")] == ["A1", "A2", "A3"]
    meta = world["coll"].store["t1::A3::v1::summary"][1]
    assert meta["account_id"] == "alpha-health" and meta["conversation_sequence"] == 3
    assert meta["is_current_revision"] is True and meta["source_version"] == 1
    assert meta["content_hash"].startswith("sha256:") and meta["index_version"] == "2026-10-13.1"
    assert meta["chunking_version"] == "speaker-turn-v2"


def test_unchanged_reindex_is_a_noop(world):
    before = world["coll"].count()
    stats = world["index"]()
    assert stats["inserted"] == 0 and stats["updated"] == 0 and stats["unchanged"] == 6
    assert world["coll"].count() == before
    assert stats["index_version"] == "2026-10-13.2"


def test_changed_transcript_creates_new_current_revision(world):
    write_meeting(world["ds"], "A3", "Aegis / Alpha Health - Renewal Signed", "2026-04-20T10:00:00Z",
                  ["kim@aegiscloud.com", "eve@alphahealth.com"],
                  "Alpha Health renewal status: signed with flat pricing, seat expansion 20 percent (corrected).",
                  [("Eve", "Renewal status is final, flat pricing, corrected seat expansion 20 percent.")])
    stats = world["index"]()
    assert stats["updated"] == 1
    store = world["coll"].store
    assert store["t1::A3::v2::summary"][1]["is_current_revision"] is True
    assert store["t1::A3::v2::summary"][1]["supersedes_version"] == 1
    assert store["t1::A3::v1::summary"][1]["is_current_revision"] is False      # kept for audit
    vs = world["manifest"].versions(T, "A3")
    assert [(v["version"], v["is_current"], v["supersedes_version"]) for v in vs] == [(1, 0, None), (2, 1, 1)]

    p = world["pipeline"]()
    r = p.retrieve("What is Alpha Health's current renewal status?", Principal(T))
    assert all(c.is_current_revision for c in r.chunks), "superseded revision leaked into a normal answer"
    assert any(c.chunk_id.startswith("t1::A3::v2") for c in r.chunks)

    r = p.retrieve("Use the original transcript version of the Alpha Health renewal signed call", Principal(T))
    assert r.chunks and all(c.source_version == 1 for c in r.chunks)


def test_deleted_meeting_is_tombstoned(world):
    shutil.rmtree(world["ds"] / "A2")
    stats = world["index"]()
    assert stats["deleted"] == 1
    assert [x["meeting_id"] for x in world["manifest"].timeline(T, "alpha-health")] == ["A1", "A3"]
    assert all(m["is_current_revision"] is False for cid, (t, m) in world["coll"].store.items() if "::A2::" in cid)
    r = world["pipeline"]().retrieve("Alpha Health outage zero visibility", Principal(T))
    assert "A2" not in meetings(r)


def test_late_arriving_earlier_call_renumbers_sequence(world):
    write_meeting(world["ds"], "A0", "Aegis / Alpha Health - Intro", "2026-01-15T10:00:00Z",
                  ["kim@aegiscloud.com", "eve@alphahealth.com"], "Alpha Health intro call.", [("Eve", "Hello.")])
    world["index"]()
    store = world["coll"].store
    assert store["t1::A0::v1::summary"][1]["conversation_sequence"] == 1
    assert store["t1::A3::v1::summary"][1]["conversation_sequence"] == 4      # was 3


# ── Retrieval modes ───────────────────────────────────────────────────────────

def test_latest_prefers_newest_relevant_meeting(world):
    r = world["pipeline"]().retrieve("What is Alpha Health's current renewal status?", Principal(T))
    assert r.plan["mode"] == "latest"
    assert r.primary_meeting_id == "A3" and meetings(r)[0] == "A3"
    assert "newest relevant meeting" in r.plan_note


def test_as_of_never_uses_future_meetings(world):
    r = world["pipeline"]().retrieve("As of March 1, 2026, what was Alpha Health's renewal status?", Principal(T))
    assert r.plan["mode"] == "as_of" and r.chunks
    assert all(c.date_int <= 20260301 for c in r.chunks)
    assert set(meetings(r)) == {"A1"}


def test_explicit_period_has_no_recency_boost(world):
    r = world["pipeline"]().retrieve("What did Alpha Health say in March 2026?", Principal(T))
    assert r.plan["mode"] == "historical"
    assert set(meetings(r)) == {"A2"}


def test_previous_call(world):
    r = world["pipeline"]().retrieve("What was discussed in Alpha Health's call before their most recent one?",
                                     Principal(T))
    assert r.plan["mode"] == "previous" and set(meetings(r)) == {"A2"}


def test_comparison_gets_both_calls_oldest_first(world):
    r = world["pipeline"]().retrieve("What changed for Alpha Health since the previous call?", Principal(T))
    assert r.plan["mode"] == "comparison"
    ms = meetings(r)
    assert set(ms) == {"A2", "A3"} and ms.index("A2") < ms.index("A3")


def test_history_is_chronological(world):
    r = world["pipeline"]().retrieve("Summarize the Alpha Health account history", Principal(T))
    ms = list(dict.fromkeys(meetings(r)))
    assert ms == ["A1", "A2", "A3"]


def test_exact_meeting_id(world):
    write_meeting(world["ds"], "01KQ85D10E82A26A8F754FB8", "Aegis / Alpha Health - Review", "2026-02-20T10:00:00Z",
                  ["kim@aegiscloud.com", "eve@alphahealth.com"], "Alpha Health review.", [("Kim", "Review.")])
    world["index"]()
    r = world["pipeline"]().retrieve("In meeting 01KQ85D10E82A26A8F754FB8, what happened?", Principal(T))
    assert r.plan["mode"] == "meeting" and set(meetings(r)) == {"01KQ85D10E82A26A8F754FB8"}
    # the model must be told which meeting the id refers to (golden-v3 q030 abstained without it)
    assert '01KQ85D10E82A26A8F754FB8 = "Aegis / Alpha Health - Review" (2026-02-20)' in r.plan_note


def test_account_question_does_not_mix_similar_customer(world):
    r = world["pipeline"]().retrieve("What renewal conditions did Ironclad Financial set?", Principal(T))
    assert r.chunks and {c.account_id for c in r.chunks} <= {"ironclad-financial", ""}


def test_timeline_respects_rbac(world):
    """sales can't read support cases → 'previous call' must skip the support case, not leak it."""
    r = world["pipeline"]().retrieve("What was discussed in Alpha Health's call before their most recent one?",
                                     Principal(T, role="sales"))
    assert r.timeline_meeting_ids == ["A1", "A3"]
    assert set(meetings(r)) == {"A1"}


# ── Telemetry ─────────────────────────────────────────────────────────────────

def test_trace_records_temporal_plan_and_checks(world):
    p = world["pipeline"]()
    p.answer("What is Alpha Health's current renewal status?", Principal(T))
    t = read_jsonl(world["tmp"] / "traces.jsonl")[-1]
    assert t["temporal_mode"] == "latest"
    assert t["temporal"]["account_id"] == "alpha-health"
    assert t["temporal"]["latest_is_newest"] is True
    assert t["checks"]["as_of_leakage"] == [] and t["checks"]["account_mixing"] == []
    assert t["checks"]["stale_revision"] == []


def test_checks_flag_leakage_mixing_and_stale_revision(world):
    from src.telemetry import deterministic_checks
    p = world["pipeline"]()
    res = p.answer("As of March 1, 2026, what was Alpha Health's renewal status?", Principal(T))
    # Simulate a retrieval bug: inject a future, other-account, superseded chunk
    bad = res.retrieval.chunks[0]
    bad.metadata = {**bad.metadata, "date_int": 20260420, "account_id": "ironworks-corp", "is_current_revision": False}
    c = deterministic_checks(Principal(T), res)
    assert c["as_of_leakage"] and c["account_mixing"] and c["stale_revision"] and c["severity"] == "warn"


# ── Golden set labels stay consistent with the parser ─────────────────────────

def test_golden_v3_labels_match_query_understanding():
    from pathlib import Path
    accts = json.loads(Path("data/accounts.json").read_text())["accounts"]
    for g in json.loads(Path("data/evals/golden-v3.json").read_text()):
        plan = understand(g["question"], accts, g.get("filters"), reference_date=REF)
        if "expected_account_id" in g:
            assert plan.account_id == g["expected_account_id"], g["id"]
        if "expected_temporal_mode" in g:
            assert plan.mode == g["expected_temporal_mode"], g["id"]
