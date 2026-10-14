"""
Regression tests for the production-readiness review (2026-10-13):

  1. tenant comes only from the verified token
  2. --reset is tenant-scoped
  3. index updates are failure-safe (two-phase, idempotent re-runs, repair)
  4. indexing-settings changes trigger re-embedding; embedding-model changes are refused
  5. declared Python version matches the code
"""

import re
import time
from pathlib import Path

import jwt
import pytest

from src import rag_index
from src.accounts import build_registry
from src.manifest import Manifest
from src.rag_index import build_index
from src.rag_pipeline import RAGConfig, RAGPipeline
from tests.fakes import FakeCollection, FakeLLM
from tests.test_temporal import write_meeting

SECRET = "x" * 48


@pytest.fixture
def token_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "token")
    monkeypatch.setenv("JWT_SECRET", SECRET)


# ── 1. Tenant only from the verified token ────────────────────────────────────

def test_tenant_flag_cannot_override_token(token_mode):
    from src.auth import issue_token
    from src.rag_chat import resolve_principal
    tok = issue_token("vanessa.cruz@aegis.com")
    p = resolve_principal(tok, None, None)
    assert (p.tenant_id, p.role) == ("aegis", "sales")
    assert resolve_principal(tok, None, "aegis").tenant_id == "aegis"          # matching flag is fine
    with pytest.raises(PermissionError, match="cannot be overridden"):
        resolve_principal(tok, None, "attacker-selected-tenant")


def test_forged_or_incomplete_tokens_rejected(token_mode):
    from src.auth import JWT_AUDIENCE, JWT_ISSUER, verify_token
    now = int(time.time())
    base = {"sub": "vanessa.cruz@aegis.com", "role": "sales", "tenant_id": "aegis", "iat": now,
            "exp": now + 60, "iss": JWT_ISSUER, "aud": JWT_AUDIENCE}
    forged = jwt.encode({**base, "tenant_id": "other"}, "y" * 48, algorithm="HS256")   # wrong key
    with pytest.raises(PermissionError):
        verify_token(forged)
    no_tenant = jwt.encode({k: v for k, v in base.items() if k != "tenant_id"}, SECRET, algorithm="HS256")
    with pytest.raises(PermissionError):
        verify_token(no_tenant)
    bad_role = jwt.encode({**base, "role": "superuser"}, SECRET, algorithm="HS256")
    with pytest.raises(PermissionError, match="unknown role"):
        verify_token(bad_role)
    wrong_aud = jwt.encode({**base, "aud": "someone-else"}, SECRET, algorithm="HS256")
    with pytest.raises(PermissionError):
        verify_token(wrong_aud)
    assert verify_token(jwt.encode(base, SECRET, algorithm="HS256"))["tenant_id"] == "aegis"


def test_email_login_and_anonymous_access_are_dev_only(token_mode, monkeypatch):
    from src.rag_chat import resolve_principal
    with pytest.raises(PermissionError, match="development simulation"):
        resolve_principal(None, "vanessa.cruz@aegis.com", None)
    with pytest.raises(PermissionError, match="Authentication required"):
        resolve_principal(None, None, "aegis")
    monkeypatch.setenv("AUTH_MODE", "dev")
    assert resolve_principal(None, "vanessa.cruz@aegis.com", None).tenant_id == "aegis"
    anon = resolve_principal(None, None, "aegis")
    assert anon.role is None and anon.tenant_id == "aegis"


def test_placeholder_secret_refused_outside_dev(monkeypatch):
    from src.auth import issue_token
    monkeypatch.setenv("AUTH_MODE", "token")
    for bad in ("", "change-me-in-production", "short"):
        monkeypatch.setenv("JWT_SECRET", bad)
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            issue_token("vanessa.cruz@aegis.com")


def test_simulated_login_token_is_not_a_real_credential(token_mode):
    from src.auth import _EPHEMERAL_SECRET, issue_token, verify_token
    sim = issue_token("vanessa.cruz@aegis.com", _key=_EPHEMERAL_SECRET)
    with pytest.raises(PermissionError):
        verify_token(sim)                     # rejected by the real verifier


def test_eval_harness_refuses_tenant_mismatch(token_mode):
    from src.eval_run import principal_for
    item = {"id": "qX", "as_user": "vanessa.cruz@aegis.com"}
    assert principal_for(item, "aegis").role == "sales"
    with pytest.raises(SystemExit, match="belongs to tenant 'aegis'"):
        principal_for(item, "default")


# ── shared synthetic dataset ──────────────────────────────────────────────────

def make_dataset(root: Path):
    E = ["kim@aegiscloud.com", "eve@alphahealth.com"]
    write_meeting(root, "A1", "Aegis / Alpha Health - Kickoff", "2026-02-10T10:00:00Z", E, "Alpha kickoff.",
                  [("Eve", "Kickoff renewal pricing."), ("Kim", "Sending pricing.")])
    write_meeting(root, "A2", "Aegis / Alpha Health - Renewal", "2026-04-10T10:00:00Z", E, "Alpha renewal signed.",
                  [("Eve", "Renewal signed flat pricing."), ("Kim", "Great.")])
    return root


@pytest.fixture
def env(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    return {"ds": ds, "coll": FakeCollection(), "manifest": Manifest(":memory:"), "reg": build_registry(ds)}


def run(env, tenant="t1", **kw):
    return build_index(tenant_id=tenant, dataset_dir=env["ds"], collection=env["coll"], manifest=env["manifest"],
                       registry=env["reg"], today="2026-10-13", **kw)


def tenant_ids(coll, tenant):
    return {cid for cid, (t, m) in coll.store.items() if m["tenant_id"] == tenant}


# ── 2. --reset is tenant-scoped ───────────────────────────────────────────────

def test_reset_only_touches_one_tenant(env):
    run(env, "t1")
    run(env, "t2")
    t2_before = tenant_ids(env["coll"], "t2")
    stats = run(env, "t1", reset=True)
    assert stats["inserted"] == 2
    assert tenant_ids(env["coll"], "t2") == t2_before, "other tenant's vectors were touched"
    assert env["manifest"].has_tenant("t2")
    again = run(env, "t2")                           # t2 manifest and vectors still agree
    assert again["unchanged"] == 2 and again["repaired"] == 0


def test_unmanifested_tenant_vectors_are_refused(env):
    run(env, "t1")
    env["manifest"].reset("t1")                      # vectors exist, manifest forgot them
    with pytest.raises(SystemExit, match="--reset --tenant t1"):
        run(env, "t1")


def test_index_version_numbers_are_never_reused(env):
    run(env, "t1")
    run(env, "t2")
    run(env, "t1", reset=True)
    versions = [r[0] for r in env["manifest"].conn.execute("SELECT index_version FROM index_runs")]
    assert len(versions) == len(set(versions)) == 3


# ── 3. Failure-safe updates ───────────────────────────────────────────────────

def test_crash_during_first_index_leaves_nothing_half_committed(env):
    env["coll"].fail_upsert_after = 0
    with pytest.raises(ConnectionError):
        run(env)
    m = env["manifest"]
    assert m.last_run("t1")["status"] == "failed"
    assert m.timeline("t1", "alpha-health") == []                 # nothing claims to be served
    assert all(v["status"] == "pending" for mid in ("A1", "A2") for v in m.versions("t1", mid))

    env["coll"].fail_upsert_after = None
    stats = run(env)
    assert stats["inserted"] == 2
    assert [v["version"] for v in m.versions("t1", "A1")] == [1]  # pending version reused, not v2
    assert run(env)["unchanged"] == 2


def test_crash_during_update_keeps_serving_old_revision(env):
    run(env)
    write_meeting(env["ds"], "A2", "Aegis / Alpha Health - Renewal", "2026-04-10T10:00:00Z",
                  ["kim@aegiscloud.com", "eve@alphahealth.com"], "Alpha renewal signed (corrected).",
                  [("Eve", "Renewal signed, corrected terms.")])
    env["coll"].fail_upsert_after = env["coll"].upserts            # next upsert fails
    with pytest.raises(ConnectionError):
        run(env)
    m, store = env["manifest"], env["coll"].store
    assert m.current_version_row("t1", "A2")["version"] == 1       # v1 still the committed revision
    current = [cid for cid, (t, md) in store.items() if "::A2::" in cid and md["is_current_revision"]]
    assert current and all("::v1::" in c for c in current)

    env["coll"].fail_upsert_after = None
    assert run(env)["updated"] == 1
    assert m.current_version_row("t1", "A2")["version"] == 2       # pending v2 finished, not v3
    assert all("::v2::" in cid for cid, (t, md) in store.items() if "::A2::" in cid and md["is_current_revision"])


def test_missing_vectors_are_repaired(env):
    run(env)
    victim = next(cid for cid in env["coll"].store if "::A1::" in cid and "::tx::" in cid)
    env["coll"].delete(ids=[victim])
    stats = run(env)
    assert stats["repaired"] == 1 and victim in env["coll"].store


# ── 4. Settings fingerprint and embedding model ───────────────────────────────

def test_changed_indexing_settings_trigger_reembed(env, monkeypatch):
    run(env)
    monkeypatch.setenv("ANONYMIZE_SPEAKERS", "true")
    stats = run(env)
    assert stats["reembedded"] == 2 and stats["unchanged"] == 0
    tx = [t for cid, (t, m) in env["coll"].store.items() if "::tx::" in cid]
    assert tx and all("Speaker_" in t for t in tx)
    assert [v["version"] for v in env["manifest"].versions("t1", "A1")] == [1]   # same revision, re-embedded
    assert run(env)["unchanged"] == 2


def test_changed_access_rules_trigger_reembed(env, monkeypatch):
    run(env)
    original = rag_index.meeting_fingerprint
    monkeypatch.setattr(rag_index, "meeting_fingerprint", lambda s, a, l: original({**s, "tagging_rules": "v2"}, a, l))
    assert run(env)["reembedded"] == 2


def test_embedding_model_change_is_refused(env, monkeypatch):
    run(env)
    monkeypatch.setattr(rag_index, "EMBED_MODEL", "text-embedding-3-large")
    with pytest.raises(SystemExit, match="new collection"):
        run(env)


def test_pipeline_refuses_collection_built_with_other_model(env):
    run(env)
    with pytest.raises(ValueError, match="built with text-embedding-3-small"):
        RAGPipeline(RAGConfig(embed_model="text-embedding-3-large"), collection=env["coll"],
                    llm=FakeLLM(), manifest=env["manifest"])


# ── 5. Packaging ──────────────────────────────────────────────────────────────

def test_declared_python_version_matches_code():
    assert re.search(r'^requires-python\s*=\s*">=3\.10"', Path("pyproject.toml").read_text(), re.M)
    if Path("setup.cfg").exists():          # not copied into the Docker image
        assert "python_requires = >=3.10" in Path("setup.cfg").read_text()
