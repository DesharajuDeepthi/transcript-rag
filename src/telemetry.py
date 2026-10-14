"""
telemetry.py — privacy-safe per-request traces, cost accounting and the
real-time deterministic checks that run on every answer.

Each call to RAGPipeline.answer() appends one JSON line to TRACE_PATH
(default outputs/traces/traces.jsonl). A trace contains what is needed to
debug, monitor and evaluate a request — and nothing more:

  • the query and answer are PII-redacted before they are written
  • the user is stored as a salted hash, never as an email
  • retrieved text is NOT stored; only chunk ids (online_eval re-fetches them)

Deterministic checks (no LLM, run inline, cheap):
  tenant_violation     a returned chunk belongs to another tenant         → page
  rbac_violation       a returned chunk is outside the caller's role       → page
  pii_in_answer        SSN / card number in the answer → page; email/phone → warn
  invalid_citations    answer cites [n] that was not in the context        → warn
  injection_in_context retrieved text contains "ignore previous instructions"-style text → warn
  empty_retrieval      nothing retrieved (tracked as a rate, not an alert)
  as_of_leakage        an "as of" / period question got a source from outside its window → warn
  account_mixing       an account question got a source from a different customer       → warn
  stale_revision       a superseded transcript revision was used without being asked for → warn
  truncated            the answer hit the output-token cap even after one retry        → warn

In production, ship these lines to your log pipeline (Datadog, CloudWatch,
OpenTelemetry …) with tenant-separated storage and a retention limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.pii_scrub import detect_pii, scrub_text

if TYPE_CHECKING:
    from src.rag_pipeline import Principal, RAGResult

log = logging.getLogger("rag.telemetry")

TRACE_PATH = os.getenv("TRACE_PATH", "outputs/traces/traces.jsonl")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "outputs/traces/feedback.jsonl")
ONLINE_EVAL_SAMPLE_RATE = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.10"))
USER_HASH_SALT = os.getenv("TELEMETRY_SALT", "rotate-me")
MAX_STORED_CHARS = 4000

# ── Pricing ───────────────────────────────────────────────────────────────────
# USD per million tokens. Source: platform.claude.com/docs/en/about-claude/pricing
# and OpenAI embeddings pricing, checked 2026-10-13. Update when prices change.
# Haiku 5.5 is tiered by prompt length: prompts over 100K input tokens cost more.
PRICING_USD_PER_MTOK: dict[str, dict] = {
    "claude-haiku-5-5": {"tiers": [(100_000, 0.10, 0.50), (None, 0.50, 2.50)]},
    "text-embedding-3-small": {"embed": 0.02},
}

SEVERE_PII = {"SSN", "CREDIT_CARD"}

INJECTION_PATTERNS = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) instructions"
    r"|disregard (?:all |any |the )?(?:previous|prior|above|system) "
    r"|you are now (?:a|an|in) "
    r"|reveal (?:your|the) (?:system )?prompt"
    r"|<\s*/?\s*system\s*>",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 chars/token) for things the API doesn't report, e.g. query embeddings."""
    return math.ceil(len(text or "") / 4)


def estimate_cost_usd(chat_model: str, input_tokens: int, output_tokens: int,
                      embed_model: str | None = None, embed_tokens: int = 0) -> float | None:
    price = PRICING_USD_PER_MTOK.get(chat_model)
    if price is None:
        return None
    cost = 0.0
    for limit, p_in, p_out in price["tiers"]:
        if limit is None or input_tokens <= limit:
            cost += input_tokens * p_in / 1e6 + output_tokens * p_out / 1e6
            break
    if embed_model and embed_tokens:
        cost += embed_tokens * PRICING_USD_PER_MTOK.get(embed_model, {}).get("embed", 0.0) / 1e6
    return round(cost, 8)


# ── Query classification (for segmenting metrics) ─────────────────────────────

_CATEGORY_RULES = [
    ("aggregation", r"\bhow many\b|\bcount\b|\bwhich (?:meetings|customers|accounts)\b|\blist (?:all|every)\b"
                    r"|\bacross\b|\btrend|\bmost\b|\bcompare\b"),
    ("action_items", r"action item|follow[- ]?up|agree(?:d)? to|\bowner\b|by when|next steps|commit"),
    ("date", r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b20\d\d-\d\d"
             r"|last (?:week|month|quarter)|\bq[1-4]\b"),
    ("who", r"^\s*who\b|\bwho (?:said|raised|owns|asked)\b"),
    ("why", r"^\s*why\b|\breason\b|\bcause\b"),
]


def classify_query(question: str, filters: dict | None = None) -> str:
    if filters and (filters.get("date_from") or filters.get("date_to")):
        return "date"
    q = (question or "").lower()
    for name, pattern in _CATEGORY_RULES:
        if re.search(pattern, q):
            return name
    return "lookup"


# ── Deterministic checks ──────────────────────────────────────────────────────

def deterministic_checks(principal: "Principal", result: "RAGResult") -> dict:
    from src.roles import get_allowed_levels

    chunks = result.retrieval.chunks
    allowed = set(get_allowed_levels(principal.role)) if principal.role else None

    tenant_bad = [c.chunk_id for c in chunks if c.tenant_id and c.tenant_id != principal.tenant_id]
    rbac_bad = [c.chunk_id for c in chunks if allowed is not None and c.access_level not in allowed]
    pii_types = sorted({p["type"] for p in detect_pii(result.answer or "")})
    injected = [c.rank for c in chunks if INJECTION_PATTERNS.search(c.text or "")]

    plan = result.retrieval.plan or {}
    hi = min(x for x in (plan.get("as_of"), plan.get("date_to")) if x) if (plan.get("as_of") or plan.get("date_to")) else None
    lo = plan.get("date_from")
    leak = [c.rank for c in chunks if c.date_int and ((hi and c.date_int > hi) or (lo and c.date_int < lo))]
    acct = plan.get("account_id")
    mixing = [c.rank for c in chunks if acct and c.account_id not in (acct, "")]
    stale = [c.rank for c in chunks if plan.get("mode") != "revision" and not c.is_current_revision]

    checks = {
        "truncated": bool(getattr(result, "truncated", False)),
        "as_of_leakage": leak,
        "account_mixing": mixing,
        "stale_revision": stale,
        "tenant_violation": tenant_bad,
        "rbac_violation": rbac_bad,
        "pii_in_answer": pii_types,
        "invalid_citations": result.citations.get("invalid", []),
        "injection_in_context": injected,
        "empty_retrieval": not chunks,
    }
    if tenant_bad or rbac_bad or SEVERE_PII & set(pii_types):
        checks["severity"] = "page"
    elif pii_types or checks["invalid_citations"] or injected or leak or mixing or stale or checks["truncated"]:
        checks["severity"] = "warn"
    else:
        checks["severity"] = "ok"
    return checks


def _in_sample(request_id: str, rate: float) -> bool:
    """Deterministic sampling: the same request id always gets the same decision."""
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 10_000
    return bucket < rate * 10_000


def _user_hash(user: str | None) -> str | None:
    if not user:
        return None
    return hashlib.sha256((USER_HASH_SALT + user.lower()).encode()).hexdigest()[:16]


def _redact(text: str) -> str:
    return scrub_text(text or "")[:MAX_STORED_CHARS]


# ── Sink ──────────────────────────────────────────────────────────────────────

class Tracer:
    """Appends one JSON line per request. Thread-safe; failures never break the request."""

    def __init__(self, path: str | None = None, sample_rate: float | None = None,
                 feedback_path: str | None = None):
        self.path = Path(path or TRACE_PATH)
        self.feedback_path = Path(feedback_path) if feedback_path else (
            Path(FEEDBACK_PATH) if path is None else self.path.with_name("feedback.jsonl"))
        self.sample_rate = ONLINE_EVAL_SAMPLE_RATE if sample_rate is None else sample_rate
        self._lock = threading.Lock()

    def record(self, result: "RAGResult", principal: "Principal", filters: dict,
               source: str = "chat", error: str | None = None) -> dict:
        cfg = result.config
        checks = deterministic_checks(principal, result)
        embed_tokens = estimate_tokens(result.question)
        trace = {
            "type": "request",
            "request_id": result.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,                         # chat | eval | api …
            "tenant_id": principal.tenant_id,
            "role": principal.role or "dev",
            "user_hash": _user_hash(principal.user),
            "query_redacted": _redact(result.question),
            "query_category": classify_query(result.question, filters),
            "filters": filters,
            "where": result.retrieval.where,
            "temporal_mode": (result.retrieval.plan or {}).get("mode") or "none",
            "retrieved_chunk_ids": [c.chunk_id for c in result.retrieval.chunks],
            "retrieved_meeting_ids": [c.meeting_id for c in result.retrieval.chunks],
            "source_windows": [c.window_ids for c in result.retrieval.chunks],
            "retrieval_scores": [c.relevance_pct for c in result.retrieval.chunks],
            "retrieval_error": result.retrieval.error,
            "temporal": {
                "mode": (result.retrieval.plan or {}).get("mode"),
                "account_id": (result.retrieval.plan or {}).get("account_id"),
                "account_resolution": (result.retrieval.plan or {}).get("account_resolution"),
                "as_of": (result.retrieval.plan or {}).get("as_of"),
                "date_from": (result.retrieval.plan or {}).get("date_from"),
                "date_to": (result.retrieval.plan or {}).get("date_to"),
                "primary_meeting_id": result.retrieval.primary_meeting_id,
                "newest_eligible_meeting_id": (result.retrieval.timeline_meeting_ids or [None])[-1],
                # latest-type answers: was the newest eligible call the one treated as current?
                "latest_is_newest": (result.retrieval.primary_meeting_id ==
                                     (result.retrieval.timeline_meeting_ids or [None])[-1])
                if (result.retrieval.plan or {}).get("mode") in ("latest", "as_of", "account_overview")
                and result.retrieval.timeline_meeting_ids else None,
                "reasons": (result.retrieval.plan or {}).get("reasons", []),
            },
            "index_version": cfg.get("index_version"),
            "prompt_version": cfg.get("prompt_version"),
            "prompt_sha": cfg.get("prompt_sha"),
            "embedding_model": cfg.get("embed_model"),
            "generation_model": cfg.get("chat_model"),
            "top_k": cfg.get("top_k"),
            "citations": result.citations,
            "abstained": result.abstained,
            "answer_redacted": _redact(result.answer),
            "latency_ms": result.latency_ms,
            "tokens": {**result.usage, "embedding_est": embed_tokens},
            "cost_usd": estimate_cost_usd(cfg.get("chat_model"), result.usage["input_tokens"],
                                          result.usage["output_tokens"], cfg.get("embed_model"),
                                          embed_tokens),
            "checks": checks,
            "sampled_for_online_eval": bool(
                not error and result.retrieval.chunks
                and (checks["severity"] != "ok" or _in_sample(result.request_id, self.sample_rate))
            ),
            "error": error,
        }
        if checks["severity"] == "page":
            log.error("PAGE: security check failed for request %s: %s", result.request_id,
                      {k: v for k, v in checks.items() if v and k != "severity"})
        self._append(self.path, trace)
        return trace

    def record_feedback(self, request_id: str, rating: str, comment: str = "",
                        principal: "Principal | None" = None) -> dict:
        """rating: 'up' | 'down' | 'incorrect'. Written next to the traces for the monitor."""
        row = {
            "type": "feedback",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "comment": _redact(comment),
            "tenant_id": principal.tenant_id if principal else None,
            "role": (principal.role or "dev") if principal else None,
        }
        self._append(self.feedback_path, row)
        return row

    def _append(self, path: Path, row: dict) -> None:
        try:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
        except OSError as e:  # telemetry must never take the request down
            log.warning("could not write trace to %s: %s", path, e)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows
