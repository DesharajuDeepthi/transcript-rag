"""
query_understanding.py — work out WHICH account and WHEN a question is about,
before anything is retrieved.

Deterministic rules (no LLM call): fast, free, testable, and every decision is
explained in `plan.reasons` and logged with the trace. The retrieval layer then
ENFORCES the plan with filters; the prompt only explains it to the model.

  "What is Bridgeport's current renewal status?"     → latest       account=bridgeport-health
  "What did Bridgeport say in February?"             → historical   2026-02-01 … 2026-02-28, no recency boost
  "What did we know as of March 1?"                  → as_of        date_to=2026-03-01
  "What changed since the previous call?"            → comparison   latest two calls, oldest → newest
  "What was discussed in the call before that?"      → previous     second-newest call
  "Summarize the account history"                    → all_history  every call, oldest → newest
  "What happened in meeting 01KQ…?"                   → meeting      exact meeting_id
  "Use the original transcript version"              → revision     source_version=1, inactive revisions allowed
  "What is happening with Bridgeport?"               → account_overview  latest 3 calls
  "Why did Coastal pick Protect?"                    → account      semantic search inside the account
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from src.accounts import normalize

MODES = ("none", "account", "latest", "previous", "as_of", "historical", "comparison",
         "all_history", "account_overview", "meeting", "revision")

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_MON = r"(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"

MEETING_ID_RE = re.compile(r"\b(01[0-9A-HJKMNP-TV-Z]{22})\b")

_RE = {
    "revision": re.compile(r"\b(?:original (?:transcript|version|recording)|first version|earlier revision|"
                           r"previous revision|old(?:er)? version|before (?:it was|the transcript was) "
                           r"(?:corrected|updated|edited)|version (\d+)|revision (\d+)|v(\d+) of)\b", re.I),
    "comparison": re.compile(r"\b(?:what (?:has )?changed|what is different|what's different|how (?:has|have|did) .{0,40}"
                             r"chang|since (?:the|our|their) (?:previous|last|prior) (?:call|meeting|conversation)|"
                             r"compared? (?:to|with) (?:the |our |their )?(?:previous|last|prior|earlier)|"
                             r"difference between)\b", re.I),
    "previous": re.compile(r"\b(?:previous (?:call|meeting|conversation)|prior (?:call|meeting|conversation)|"
                           r"(?:call|meeting|conversation) before (?:the |their |our )?(?:latest|last|most recent|that)|"
                           r"before (?:the |their |our )?(?:most recent|latest|last) (?:call|meeting|conversation)|"
                           r"second[- ]most[- ]recent|penultimate)\b", re.I),
    "history": re.compile(r"\b(?:history|timeline|over time|chronolog\w*|all (?:of )?(?:the |their |our )?"
                          r"(?:calls|meetings|conversations)|every (?:call|meeting|conversation)|evolv\w*|"
                          r"evolution|journey|so far)\b", re.I),
    "overview": re.compile(r"\b(?:what(?:'s| is| has been) (?:happening|going on)|update on|catch me up|"
                           r"where (?:do|are) we (?:stand|with)|how(?:'s| is| are) .{0,40} doing|overview of)\b", re.I),
    "latest": re.compile(r"\b(?:current(?:ly)?|latest|most recent(?:ly)?|newest|right now|now|today|these days|"
                         r"at the moment|at present|status|still|last (?:call|meeting|conversation)|recent(?:ly)?)\b",
                         re.I),
    "as_of": re.compile(r"\b(?:as of|as at|by the end of|at the time of|what did we know (?:on|by|as of|at)|"
                        r"what was known (?:on|by|as of))\s*$", re.I),
    "before": re.compile(r"\b(?:before|prior to|until|up to)\s*$", re.I),
    "since": re.compile(r"\b(?:since|from|starting|beginning)\s*$", re.I),
    "after": re.compile(r"\bafter\s*$", re.I),
}


@dataclass
class DateSpan:
    start: date
    end: date
    text: str
    pos: int
    end_pos: int
    kind: str = "day"          # day | month | quarter | year


@dataclass
class TemporalPlan:
    mode: str = "none"
    account_id: str | None = None
    account_name: str | None = None
    account_resolution: str = "none"          # resolved | multiple | none
    account_candidates: list[str] = field(default_factory=list)
    date_from: int | None = None              # YYYYMMDD, inclusive
    date_to: int | None = None                # YYYYMMDD, inclusive
    as_of: int | None = None                  # YYYYMMDD, inclusive
    meeting_id: str | None = None
    source_version: int | None = None
    recency_boost: bool = False
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return asdict(self)


def _year_for(month: int, ref: date) -> int:
    return ref.year if month <= ref.month else ref.year - 1


def _month_span(y: int, m: int, part: str | None = None) -> tuple[date, date]:
    last = calendar.monthrange(y, m)[1]
    if part == "early":
        return date(y, m, 1), date(y, m, 10)
    if part in ("mid", "middle"):
        return date(y, m, 11), date(y, m, 20)
    if part == "late":
        return date(y, m, 21), date(y, m, last)
    return date(y, m, 1), date(y, m, last)


def find_dates(text: str, ref: date) -> list[DateSpan]:
    """All dates / periods mentioned, in order. Day-level matches win over month-level ones."""
    spans: list[DateSpan] = []
    taken: list[tuple[int, int]] = []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    def add(s, e, m, kind="day"):
        if free(m.start(), m.end()):
            spans.append(DateSpan(s, e, m.group(0), m.start(), m.end(), kind))
            taken.append((m.start(), m.end()))

    for m in re.finditer(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text):
        d = date(int(m[1]), int(m[2]), int(m[3]))
        add(d, d, m)
    for m in re.finditer(rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", text, re.I):
        mon = _MONTHS[m[1].lower().rstrip(".")]
        y = int(m[3]) if m[3] else _year_for(mon, ref)
        try:
            d = date(y, mon, int(m[2]))
        except ValueError:
            continue
        add(d, d, m)
    for m in re.finditer(r"\bQ([1-4])(?:\s+(20\d\d))?\b", text, re.I):
        q = int(m[1])
        y = int(m[2]) if m[2] else (ref.year if (q - 1) * 3 + 1 <= ref.month else ref.year - 1)
        add(date(y, (q - 1) * 3 + 1, 1), _month_span(y, q * 3)[1], m, "quarter")
    for m in re.finditer(rf"\b(?:(early|mid|middle|late)[- ](?:in )?)?({_MON})(?:\s+(?:of\s+)?(20\d\d))?\b", text, re.I):
        word = m[2].lower().rstrip(".")
        if word in ("may", "mar", "march") and not m[1] and not m[3]:
            # "may"/"march" are also ordinary words; require a month-ish context
            before = text[max(0, m.start() - 12):m.start()].lower()
            if not re.search(r"\b(?:in|during|of|since|before|after|by|until|through|from|early|mid|late)\s*$", before):
                continue
        mon = _MONTHS[word]
        y = int(m[3]) if m[3] else _year_for(mon, ref)
        s, e = _month_span(y, mon, (m[1] or "").lower() or None)
        add(s, e, m, "month")
    for m in re.finditer(r"\b(?:in |during )?(20\d\d)\b", text):
        y = int(m[1])
        add(date(y, 1, 1), date(y, 12, 31), m, "year")
    return sorted(spans, key=lambda s: s.pos)


# Words showing a date refers to WHEN a conversation happened (not what it was about)
_CONVO = re.compile(r"\b(?:meeting|meetings|call|calls|conversation|conversations|discuss\w*|sa(?:y|id)|mention\w*|"
                    r"talk\w*|told|tell|raised?|spoke|heard|hear|know|known|standup|review|session)\b", re.I)
_TOPIC_AFTER = re.compile(r"^\s*(?:planning|plan|roadmap|goals?|targets?|forecast|okrs?|pipeline|budget|priorities|"
                          r"objectives|results|earnings|revenue)\b", re.I)
_TOPIC_BEFORE = re.compile(r"\b(?:for|fiscal(?: year)?|fy|its|their|our|the)\s*$", re.I)
# "the March outage" names an EVENT (later meetings discuss it) — not a meeting-date filter
_EVENT_AFTER = re.compile(r"^\s*(?:[A-Z][a-z]+\s+){0,2}(?:outage|incident|launch|release|downtime|breach|failure|"
                          r"escalation|patch|bug|price increase)\b")


def is_temporal(span: DateSpan, text: str) -> bool:
    """Day/month mentions are meeting-time filters; years/quarters only with conversational context."""
    before, after = text[:span.pos], text[span.end_pos:]
    if span.kind in ("day", "month"):
        return not _EVENT_AFTER.search(after)
    if _TOPIC_AFTER.search(after) or _TOPIC_BEFORE.search(before):
        return False
    if any(_RE[k].search(before) for k in ("as_of", "before", "since", "after")):
        return True
    return bool(_CONVO.search(text))


def _di(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def resolve_account(question: str, accounts: list[dict]) -> tuple[str | None, list[str], str]:
    """Longest alias match on normalized text. Returns (account_id, candidates, status)."""
    q = f" {normalize(question)} "
    hits: dict[str, int] = {}
    for a in accounts:
        for alias in a.get("aliases", []) + [normalize(a["name"])]:
            if alias and f" {alias} " in q:
                hits[a["account_id"]] = max(hits.get(a["account_id"], 0), len(alias))
    if not hits:
        return None, [], "none"
    if len(hits) == 1:
        return next(iter(hits)), list(hits), "resolved"
    return None, sorted(hits, key=lambda k: -hits[k]), "multiple"


def understand(question: str, accounts: list[dict] | None = None, filters: dict | None = None,
               reference_date: date | None = None) -> TemporalPlan:
    ref = reference_date or date.today()
    plan = TemporalPlan()
    q = question or ""

    # ── account ───────────────────────────────────────────────────────────────
    aid, cands, status = resolve_account(q, accounts or [])
    plan.account_resolution, plan.account_candidates = status, cands
    if aid:
        plan.account_id = aid
        plan.account_name = next((a["name"] for a in accounts or [] if a["account_id"] == aid), None)
        plan.reasons.append(f"account '{plan.account_name}' matched by name/alias")
    elif status == "multiple":
        plan.reasons.append(f"several accounts named ({', '.join(cands)}); not filtering by account")

    # ── dates ─────────────────────────────────────────────────────────────────
    all_spans = find_dates(q, ref)
    spans = [s for s in all_spans if is_temporal(s, q)]
    for s in all_spans:
        if s not in spans:
            plan.reasons.append(f"'{s.text}' treated as topic, not meeting date")
    as_of = None
    rng_from = rng_to = None
    if len(spans) >= 2 and re.search(r"\b(?:between|from)\b", q[:spans[0].pos], re.I) and \
            re.search(r"^\s*(?:and|to|through|thru|until|till|-|–)\s*$", q[spans[0].end_pos:spans[1].pos], re.I):
        rng_from, rng_to = spans[0].start, spans[1].end
        plan.reasons.append(f"date range '{spans[0].text}' – '{spans[1].text}'")
    elif spans:
        s = spans[0]
        before = q[:s.pos]
        if _RE["as_of"].search(before):
            as_of = s.end
            plan.reasons.append(f"as-of '{s.text}'")
        elif _RE["before"].search(before):
            as_of = s.start - timedelta(days=1)
            plan.reasons.append(f"before '{s.text}' → as of {as_of}")
        elif _RE["since"].search(before):
            rng_from = s.start
            plan.reasons.append(f"since '{s.text}'")
        elif _RE["after"].search(before):
            rng_from = s.end + timedelta(days=1)
            plan.reasons.append(f"after '{s.text}'")
        else:
            rng_from, rng_to = s.start, s.end
            plan.reasons.append(f"period '{s.text}'")

    # Explicit UI/CLI filters are hard constraints and win over inferred dates
    filters = filters or {}
    if filters.get("date_from"):
        rng_from = date.fromisoformat(str(filters["date_from"]))
    if filters.get("date_to"):
        rng_to = date.fromisoformat(str(filters["date_to"]))
    plan.date_from = _di(rng_from) if rng_from else None
    plan.date_to = _di(rng_to) if rng_to else None
    plan.as_of = _di(as_of) if as_of else None

    # ── mode ──────────────────────────────────────────────────────────────────
    mid = MEETING_ID_RE.search(q)
    rev = _RE["revision"].search(q)
    if mid:
        plan.mode, plan.meeting_id = "meeting", mid.group(1)
        plan.reasons.append("explicit meeting id")
    if rev:
        n = next((int(g) for g in rev.groups() if g), 1)
        plan.mode, plan.source_version = "revision", n
        plan.reasons.append(f"explicit revision request → source_version={n}")
    elif mid:
        pass
    elif _RE["comparison"].search(q):
        plan.mode = "comparison"
        plan.reasons.append("change/comparison wording")
    elif _RE["previous"].search(q):
        plan.mode = "previous"
        plan.reasons.append("previous-call wording")
    elif _RE["history"].search(q):
        plan.mode = "all_history"
        plan.reasons.append("history/timeline wording")
    elif plan.as_of:
        plan.mode, plan.recency_boost = "as_of", True
    elif plan.date_from or plan.date_to:
        plan.mode = "historical"
        plan.reasons.append("explicit period → no recency boost")
    elif _RE["latest"].search(q):
        plan.mode, plan.recency_boost = "latest", True
        plan.reasons.append("latest/current wording → prefer newest relevant meeting")
    elif plan.account_id and _RE["overview"].search(q):
        plan.mode, plan.recency_boost = "account_overview", True
        plan.reasons.append("vague account question → latest few calls")
    elif plan.account_id:
        plan.mode = "account"

    # Timeline modes need an account; without one fall back to plain search (+ dates).
    # ("latest" works without an account: newest relevant meeting across the tenant.)
    if plan.mode in ("previous", "comparison", "all_history", "account_overview") and not plan.account_id:
        plan.reasons.append(f"'{plan.mode}' needs an account but none was resolved → semantic search")
        plan.mode = "historical" if (plan.date_from or plan.date_to) else ("as_of" if plan.as_of else "none")
        plan.recency_boost = plan.mode == "as_of"
    return plan
