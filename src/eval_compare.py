"""
eval_compare.py — Compare actual RAG answers against the golden set and produce a metrics report.

Usage:
  python -m src.eval_compare                              # default paths
  python -m src.eval_compare --results outputs/eval_results.json
  python -m src.eval_compare --out outputs/eval_report.json

Metrics (by pipeline phase):

  RETRIEVAL
    1. Hit@k              — was the gold meeting in the top-k retrieved chunks?
    2. MRR                — mean reciprocal rank of the gold meeting

  GENERATION
    3. Faithfulness       — of the claims in the ACTUAL answer, what fraction are
                            supported by the RETRIEVED CONTEXT?  (true "groundedness";
                            catches hallucination; does not need a golden answer)
    4. Answer correctness — of the key facts in the EXPECTED answer, what fraction
                            appear in the actual answer? (fact recall, partial credit)
       Correctness pass   — % of questions where every expected fact was covered
                            (this is what the old "groundedness_rate" was measuring)
    5. Answer quality     — 6 dimensions scored 0-2 each, max 12
    6. Abstention         — did the model say "I don't know" exactly when it should?

Why faithfulness and correctness are split:
  The old single "groundedness" grader compared the answer to the GOLDEN answer.
  An answer that honestly says "the context doesn't explain X" scored 0 even though
  it was perfectly grounded — the real failure was retrieval, not generation.
  Splitting them tells you which stage to fix:
      low correctness + high faithfulness  → retrieval/context problem
      low faithfulness                      → generation/hallucination problem

Graders use forced tool-use so the output is always a valid JSON object
(no more "Unterminated string" / truncated-JSON grader errors).
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GRADER_MODEL = "claude-haiku-5-5"
GRADER_MAX_TOKENS = 4096
CORRECTNESS_PASS = 1.0   # a question "passes" only if every expected fact is covered

# Single definition shared with the pipeline, so eval and runtime agree on what an abstention is
from src.rag_pipeline import ABSTENTION_PHRASES, is_abstention  # noqa: E402,F401  (re-exported)


# ── Clients ───────────────────────────────────────────────────────────────────

def _get_anthropic_client():
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=key)


# ── Structured grader call ────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Fallback: pull the first {...} object out of free text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _fill(template: str, **values) -> str:
    """Safe substitution — values may contain { } without breaking anything."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", v or "")
    return template


def _grade(client, prompt: str, tool_name: str, schema: dict, force_text: bool = False) -> dict:
    """
    Ask the grader to answer by calling a tool whose input_schema is the
    result shape. The API returns tool input as an already-parsed dict, so
    there is nothing to mis-parse. Falls back to text + JSON extraction if
    the model/config rejects forced tool use.
    """
    import anthropic  # lazy: lets the scoring helpers be imported/tested without the SDK
    if not force_text:
        tool = {"name": tool_name, "description": "Submit the evaluation result.",
                "input_schema": schema}
        try:
            resp = client.messages.create(
                model=GRADER_MODEL,
                max_tokens=GRADER_MAX_TOKENS,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    return dict(block.input)
            raise ValueError(f"no tool_use block (stop_reason={resp.stop_reason})")
        except anthropic.BadRequestError:
            pass  # model/config rejects forced tool use → text path below
    resp = client.messages.create(
        model=GRADER_MODEL,
        max_tokens=GRADER_MAX_TOKENS,
        messages=[{"role": "user", "content":
                   prompt + "\n\nReply with ONLY a single JSON object (no prose) matching this schema:\n"
                   + json.dumps(schema)}],
    )
    if resp.stop_reason == "max_tokens":
        raise ValueError("grader hit max_tokens — output truncated")
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _parse_json(text)


_VERDICT_WORDS = re.compile(r"\b(covered|partial|missing)\b", re.I)


def _items(data: dict, key: str, required: tuple[str, ...]) -> list[dict]:
    """
    Normalise the grader's item list. Shapes seen in practice (Haiku grader, golden-v3):
      [{"fact": ..., "verdict": ...}]            ← expected
      "[{\"fact\": ..., \"verdict\": ...}]"        ← whole list JSON-encoded as ONE string (q028)
      ["{\"fact\": ..., ...}", ...]               ← each item JSON-encoded
      {"facts": [...]} / {"items": [...]}        ← list wrapped in another object
      ["Annual renewal - covered", ...]          ← plain strings (verdict words recovered)
    Raises ValueError (with a preview of what came back) if nothing usable is left.
    """
    raw = data.get(key)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    if isinstance(raw, dict):
        inner = raw.get(key) if isinstance(raw.get(key), list) else next(
            (v for v in raw.values() if isinstance(v, list)), None)
        raw = inner if inner is not None else [raw]

    out = []
    for it in raw or []:
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except ValueError:
                m = _VERDICT_WORDS.search(it)
                if required == ("fact", "verdict") and m:
                    it = {"fact": _VERDICT_WORDS.sub("", it).strip(" -:()"), "verdict": m.group(1).lower()}
        if isinstance(it, dict) and all(k in it for k in required):
            out.append(it)
    if raw and not out:
        raise ValueError(f"grader returned malformed '{key}' items: {type(data.get(key)).__name__} "
                         f"{str(data.get(key))[:160]!r}")
    return out


def _grade_items(client, prompt, tool_name, schema, key, required, attempts=2) -> dict:
    """
    _grade + validation of the item list. Retries once with forced tool use, then falls back
    to a plain-JSON text reply (a different decoding path) before giving up.
    """
    last = None
    for i in range(attempts + 1):
        try:
            data = _grade(client, prompt, tool_name, schema) if i < attempts else \
                _grade(client, prompt, tool_name, schema, force_text=True)
            data[key] = _items(data, key, required)
            return data
        except (ValueError, TypeError) as e:
            last = e
    raise last


def _is_abstention(text: str) -> bool:
    return is_abstention(text)


# ── Metric 1 & 2: Hit@k and MRR ──────────────────────────────────────────────

def eval_retrieval(result: dict) -> dict:
    gold_id = result.get("gold_meeting_id")
    if not gold_id or gold_id == "REPLACE_WITH_REAL_MEETING_ID":
        return {"hit": None, "rank": None, "skipped": True, "reason": "no gold_meeting_id set"}

    rank = None
    for i, mid in enumerate(result.get("retrieved_meeting_ids", []), 1):
        if mid == gold_id:
            rank = i
            break
    return {"hit": rank is not None, "rank": rank, "skipped": False}


# ── Metric 3: Faithfulness (answer vs retrieved context) ─────────────────────

FAITHFULNESS_PROMPT = """You are checking a RAG answer for hallucination.

Retrieved context (the ONLY information the assistant was allowed to use):
<context>
{context}
</context>

Assistant answer:
<answer>
{actual}
</answer>

Steps:
1. Break the answer into atomic factual claims (names, numbers, dates, decisions,
   who-said-what). Ignore citations, hedges, and statements that information is
   missing from the context — those are not factual claims.
2. For each claim, decide if it is directly supported by the context.
   "supported" = the context states it or it follows trivially.
   "unsupported" = not in the context, or contradicts it.
"""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim":     {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "reasoning"],
}


def _context_for(result: dict) -> tuple[str, bool]:
    """Return (context_text, is_full). Old result files only have 200-char previews."""
    if result.get("retrieved_context"):
        return result["retrieved_context"], True
    chunks = result.get("retrieved_chunks", [])
    if chunks and chunks[0].get("text"):
        return "\n\n---\n\n".join(c["text"] for c in chunks), True
    return "\n\n---\n\n".join(c.get("text_preview", "") for c in chunks), False


def eval_faithfulness(client, result: dict) -> dict:
    actual = result.get("actual_answer", "")
    context, full = _context_for(result)
    if not full:
        return {"score": None, "skipped": True,
                "reason": "results file has no full context — re-run eval_run.py"}
    if _is_abstention(actual):
        return {"score": None, "skipped": True, "reason": "answer abstained — no claims to check"}
    try:
        data = _grade_items(client, _fill(FAITHFULNESS_PROMPT, context=context, actual=actual),
                            "submit_faithfulness", FAITHFULNESS_SCHEMA, "claims", ("claim", "supported"))
        claims = data.get("claims", [])
        if not claims:
            return {"score": None, "skipped": True, "reason": "no factual claims extracted"}
        supported = sum(1 for c in claims if c.get("supported"))
        data["score"] = round(supported / len(claims), 3)
        data["unsupported_claims"] = [c["claim"] for c in claims if not c.get("supported")]
        return data
    except Exception as e:
        return {"score": None, "skipped": True, "reason": f"grader error: {e}"}


# ── Metric 4: Answer correctness (answer vs golden answer, per fact) ─────────

CORRECTNESS_PROMPT = """You are grading a RAG answer against a reference answer.

Question: {question}

Reference answer (ground truth):
<reference>
{expected}
</reference>

Assistant answer:
<answer>
{actual}
</answer>

Steps:
1. Break the REFERENCE answer into its atomic key facts (one name / number /
   decision / reason per fact). Usually 2-6 facts.
2. For each fact, mark the assistant answer:
   "covered"  — stated, wording may differ
   "partial"  — mentioned but missing a qualifier (e.g. "MFA" instead of "adaptive MFA beta")
   "missing"  — absent or contradicted
Be lenient on wording, strict on substance. Extra correct detail is not penalised.
"""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact":    {"type": "string"},
                    "verdict": {"type": "string", "enum": ["covered", "partial", "missing"]},
                },
                "required": ["fact", "verdict"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["facts", "reasoning"],
}


def eval_correctness(client, result: dict) -> dict:
    expected = result.get("expected_answer")
    if not expected:
        return {"score": None, "skipped": True, "reason": "no expected_answer in golden set"}
    try:
        data = _grade_items(client, _fill(CORRECTNESS_PROMPT,
                                          question=result.get("question", ""),
                                          expected=expected,
                                          actual=result.get("actual_answer", "")),
                            "submit_correctness", CORRECTNESS_SCHEMA, "facts", ("fact", "verdict"))
        facts = data.get("facts", [])
        if not facts:
            return {"score": None, "skipped": True, "reason": "no facts extracted"}
        pts = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
        data["score"] = round(sum(pts.get(f.get("verdict"), 0) for f in facts) / len(facts), 3)
        data["passed"] = data["score"] >= CORRECTNESS_PASS
        data["missing_facts"] = [f["fact"] for f in facts if f.get("verdict") != "covered"]
        return data
    except Exception as e:
        return {"score": None, "skipped": True, "reason": f"grader error: {e}"}


# ── Metric 5: Answer Quality Scorecard ───────────────────────────────────────

QUALITY_PROMPT = """You are an evaluation judge for a RAG chatbot that answers questions about meeting transcripts.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Score the actual answer on each dimension from 0 to 2 (0 = wrong/missing, 1 = partial, 2 = correct/complete):
1. correctness   — are the facts accurate compared to the expected answer?
2. relevance     — does it directly address the question?
3. completeness  — does it cover all key points from the expected answer?
4. clarity       — is it easy to understand?
5. groundedness  — does it avoid making up facts not supported by context?
6. citations     — does it cite the meeting title and date?
"""

_DIMS = ["correctness", "relevance", "completeness", "clarity", "groundedness", "citations"]
QUALITY_SCHEMA = {
    "type": "object",
    "properties": {**{d: {"type": "integer", "minimum": 0, "maximum": 2} for d in _DIMS},
                   "reasoning": {"type": "string"}},
    "required": _DIMS + ["reasoning"],
}


def eval_answer_quality(client, result: dict) -> dict:
    expected = result.get("expected_answer")
    if not expected:
        return {"total": None, "skipped": True, "reason": "no expected_answer in golden set"}
    try:
        data = _grade(client, _fill(QUALITY_PROMPT,
                                    question=result.get("question", ""),
                                    expected=expected,
                                    actual=result.get("actual_answer", "")),
                      "submit_quality", QUALITY_SCHEMA)
        data["total"] = sum(int(data.get(d, 0)) for d in _DIMS)
        return data
    except Exception as e:
        return {"total": None, "skipped": True, "reason": f"grader error: {e}"}


# ── Metric 6: Abstention ──────────────────────────────────────────────────────

def eval_abstention(result: dict) -> dict:
    answerable = result.get("answerable", True)
    abstained = _is_abstention(result.get("actual_answer", ""))
    if answerable:
        correct = not abstained
        label = "TN" if correct else "FP (abstained on an answerable question)"
    else:
        correct = abstained
        label = "TP" if correct else "FN (answered instead of abstaining)"
    return {"answerable": answerable, "abstained": abstained, "correct": correct, "label": label}


# ── Metric 7: Security (deterministic — tenant / RBAC leakage) ───────────────

def eval_security(result: dict) -> dict:
    """
    Re-check every retrieved chunk against the caller's tenant and role.
    Done here from the raw chunk metadata (not only trusting the trace) so a
    bug in the runtime check can't hide a leak.
    """
    from src.roles import get_allowed_levels
    role = result.get("role")
    allowed = set(get_allowed_levels(role)) if role else None
    tenant = (result.get("_run") or {}).get("tenant_id")
    chunks = result.get("retrieved_chunks", [])
    rbac = [c.get("chunk_id") for c in chunks if allowed is not None and c.get("access_level") not in allowed]
    cross = [c.get("chunk_id") for c in chunks if tenant and c.get("tenant_id") and c.get("tenant_id") != tenant]
    return {"role": role, "rbac_violations": rbac, "tenant_violations": cross,
            "violations": len(rbac) + len(cross)}


# ── Metric 8: Temporal / account correctness (deterministic + one LLM check) ──

STALE_PROMPT = """The question was asked about a customer whose situation changed over time.

Question: {question}

Assistant answer:
<answer>
{actual}
</answer>

Outdated statements (true at some earlier point, but superseded by newer meetings):
<outdated>
{outdated}
</outdated>

For each outdated statement, decide whether the answer presents it as the CURRENT state.
Mentioning it explicitly as past history ("in March they had…, since then…") is NOT presenting it as current.
"""

STALE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"}, "presented_as_current": {"type": "boolean"}},
            "required": ["statement", "presented_as_current"]}},
        "reasoning": {"type": "string"},
    },
    "required": ["items", "reasoning"],
}


def _di(s: str | None) -> int | None:
    return int(s.replace("-", "")) if s else None


def eval_temporal(client, result: dict) -> dict:
    """Only scores the labels a golden item actually has; everything else is None."""
    plan = result.get("plan") or {}
    chunks = result.get("retrieved_chunks", [])
    got_meetings = result.get("retrieved_meeting_ids", [])
    out: dict = {"mode": plan.get("mode"), "account_id": plan.get("account_id")}

    if "expected_account_id" in result:
        out["account_correct"] = plan.get("account_id") == result["expected_account_id"]
    if "expected_temporal_mode" in result:
        out["mode_correct"] = plan.get("mode") == result["expected_temporal_mode"]
    if result.get("latest_meeting_id"):
        allowed = result["latest_meeting_id"]
        allowed = allowed if isinstance(allowed, list) else [allowed]
        out["latest_correct"] = bool(got_meetings) and got_meetings[0] in allowed
    if result.get("expected_meeting_ids"):
        need = set(result["expected_meeting_ids"])
        out["required_meetings_recall"] = round(len(need & set(got_meetings)) / len(need), 3)
    lo, hi = _di(result.get("forbidden_before")), _di(result.get("forbidden_after"))
    if lo or hi:
        out["window_leaks"] = [c.get("meeting_id") for c in chunks if c.get("date_int") and
                               ((hi and c["date_int"] > hi) or (lo and c["date_int"] < lo))]
        out["window_leak"] = bool(out["window_leaks"])
    if result.get("forbidden_account_ids"):
        bad = set(result["forbidden_account_ids"])
        out["account_mixing"] = any(c.get("account_id") in bad for c in chunks)
    if result.get("expected_source_version"):
        v = result["expected_source_version"]
        out["revision_correct"] = bool(chunks) and all(c.get("source_version") == v for c in chunks)
    elif chunks and plan:
        out["revision_correct"] = all(c.get("is_current_revision", True) for c in chunks)
    if result.get("outdated_facts") and not _is_abstention(result.get("actual_answer", "")):
        try:
            data = _grade_items(client, _fill(STALE_PROMPT, question=result.get("question", ""),
                                              actual=result.get("actual_answer", ""),
                                              outdated="\n".join(f"- {f}" for f in result["outdated_facts"])),
                                "submit_staleness", STALE_SCHEMA, "items", ("statement", "presented_as_current"))
            out["stale_answer"] = any(i.get("presented_as_current") for i in data.get("items", []))
            out["stale_items"] = [i["statement"] for i in data.get("items", []) if i.get("presented_as_current")]
        except Exception as e:
            out["stale_error"] = f"grader error: {e}"
    return out


# ── Aggregate and report ──────────────────────────────────────────────────────


def _percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))]

def _pct(xs):
    return round(sum(xs) / len(xs) * 100, 1) if xs else None


def _fmt(v, suffix="%"):
    return f"{v}{suffix}" if v is not None else "N/A"


def compare(results_path: str, out_path: str):
    print(f"\n📂 Loading eval results from: {results_path}")
    raw = json.loads(Path(results_path).read_text())
    # New format: {"run": {...config...}, "results": [...]}; old format: bare list
    run_meta = raw.get("run", {}) if isinstance(raw, dict) else {}
    results = raw["results"] if isinstance(raw, dict) else raw
    for r in results:
        r["_run"] = run_meta
    print(f"   {len(results)} questions to evaluate  "
          f"({run_meta.get('golden_version', 'unversioned')}, "
          f"prompt {run_meta.get('config', {}).get('prompt_version', '?')})\n")

    client = _get_anthropic_client()
    rows = []

    for result in results:
        qid, question = result["id"], result["question"]
        print(f"▶ [{qid}] {question[:60]}...")

        row = {
            "id":           qid,
            "question":     question,
            "answerable":   result.get("answerable", True),
            "gold_title":   result.get("gold_title"),
            "retrieval":    eval_retrieval(result),
            "faithfulness": eval_faithfulness(client, result),
            "correctness":  eval_correctness(client, result),
            "quality":      eval_answer_quality(client, result),
            "abstention":   eval_abstention(result),
            "security":     eval_security(result),
            "temporal":     eval_temporal(client, result),
            "latency_ms":   (result.get("latency_ms") or {}).get("total"),
            "cost_usd":     result.get("cost_usd"),
            "error":        result.get("error"),
        }
        rows.append(row)

        r, f, c, q = row["retrieval"], row["faithfulness"], row["correctness"], row["quality"]
        hit = "✅ HIT" if r.get("hit") else ("❌ MISS" if r.get("hit") is False else "⚪ N/A")
        fs = f"{f['score']:.2f}" if f.get("score") is not None else "—"
        cs = f"{c['score']:.2f}" if c.get("score") is not None else "—"
        print(f"  {hit}  Faithful: {fs}  Correct: {cs}  "
              f"Quality: {q.get('total', 'N/A')}/12  Abstention: {row['abstention']['label']}")
        for err in (f, c, q):
            if str(err.get("reason", "")).startswith("grader error"):
                print(f"    ⚠ {err['reason']}")
        print()

    # ── Aggregate ──────────────────────────────────────────────────────────────
    ret = [r["retrieval"] for r in rows if not r["retrieval"].get("skipped")]
    hit_rate = _pct([r["hit"] for r in ret])
    mrr = round(sum(1 / r["rank"] for r in ret if r["rank"]) / len(ret) * 100, 1) if ret else None

    faith = [r["faithfulness"]["score"] for r in rows if r["faithfulness"].get("score") is not None]
    corr  = [r["correctness"]["score"] for r in rows if r["correctness"].get("score") is not None]
    passd = [r["correctness"]["passed"] for r in rows if r["correctness"].get("score") is not None]
    qual  = [r["quality"]["total"] for r in rows if r["quality"].get("total") is not None]
    abst  = [r["abstention"]["correct"] for r in rows]
    errors = sum(1 for r in rows for k in ("faithfulness", "correctness", "quality")
                 if str(r[k].get("reason", "")).startswith("grader error"))
    sec_violations = sum(r["security"]["violations"] for r in rows)
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    run_errors = sum(1 for r in rows if r.get("error"))
    truncated = sum(1 for r in results if r.get("truncated"))

    def t_rate(key, invert=False):
        vals = [r["temporal"][key] for r in rows if r["temporal"].get(key) is not None]
        if not vals:
            return None
        return _pct([(not v) if invert else v for v in vals])

    recalls = [r["temporal"]["required_meetings_recall"] for r in rows
               if r["temporal"].get("required_meetings_recall") is not None]
    temporal_metrics = {
        "account_resolution_accuracy": t_rate("account_correct"),
        "temporal_intent_accuracy": t_rate("mode_correct"),
        "latest_selection_accuracy": t_rate("latest_correct"),
        "required_meetings_recall": round(sum(recalls) / len(recalls) * 100, 1) if recalls else None,
        "as_of_leakage_rate": t_rate("window_leak"),
        "account_mixing_rate": t_rate("account_mixing"),
        "correct_revision_rate": t_rate("revision_correct"),
        "stale_answer_rate": t_rate("stale_answer"),
    }

    # Numeric metrics (0–100 scale) — what eval_gate.py and dashboards read
    metrics = {
        "total_questions": len(results),
        "hit_at_k": hit_rate,
        "mrr": mrr,
        "faithfulness": _pct(faith),
        "answer_correctness": _pct(corr),
        "correctness_pass_rate": _pct(passd),
        "avg_answer_quality": round(sum(qual) / len(qual), 2) if qual else None,
        "abstention_accuracy": _pct(abst),
        "security_violations": sec_violations,
        "p50_latency_ms": _percentile(lat, 50),
        "p95_latency_ms": _percentile(lat, 95),
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "grader_errors": errors,
        "run_errors": run_errors,
        "truncated_answers": truncated,
        **temporal_metrics,
    }

    summary = {
        "total_questions":          len(results),
        "— retrieval —":            "",
        "hit_at_k":                 _fmt(hit_rate),
        "mrr":                      _fmt(mrr),
        "— generation —":           "",
        "faithfulness":             _fmt(_pct(faith)),
        "answer_correctness":       _fmt(_pct(corr)),
        "correctness_pass_rate":    _fmt(_pct(passd)),
        "avg_answer_quality":       _fmt(round(sum(qual) / len(qual), 2) if qual else None, "/12"),
        "abstention_accuracy":      _fmt(_pct(abst)),
        "— temporal —":             "",
        **{k: _fmt(v) for k, v in temporal_metrics.items()},
        "— security —":             "",
        "security_violations":      sec_violations,
        "— operations —":           "",
        "p50 / p95 latency":        f"{_fmt(metrics['p50_latency_ms'], ' ms')} / {_fmt(metrics['p95_latency_ms'], ' ms')}",
        "avg cost / question":      _fmt(metrics["avg_cost_usd"], " USD"),
        "— grader health —":        "",
        "graded_counts":            f"faith={len(faith)} correct={len(corr)} quality={len(qual)}",
        "grader_errors":            errors,
        "run_errors":               run_errors,
        "truncated_answers":        truncated,
    }

    for r in results:
        r.pop("_run", None)
    report = {"run": run_meta, "summary": summary, "metrics": metrics, "details": rows}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))

    print("=" * 55)
    print("  EVAL REPORT SUMMARY")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print("=" * 55)
    print(f"\n✅ Full report saved to: {out_path}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare RAG results against golden set")
    parser.add_argument("--results", default="outputs/eval_results.json")
    parser.add_argument("--out", default="outputs/eval_report.json")
    args = parser.parse_args()
    compare(args.results, args.out)


if __name__ == "__main__":
    main()
