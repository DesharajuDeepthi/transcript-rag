"""
monitor.py — turn traces + online evals + feedback into metrics and alerts.

  python -m src.monitor                       # last 24h vs trailing 7 days, rules from monitoring/alerts.yml
  python -m src.monitor --json outputs/monitoring/report.json

Exit code: 2 if any PAGE alert, 1 if any WARN alert and --fail-on-warn, else 0
(so a scheduler / CI job can notify on non-zero).

Every rule is evaluated overall and per segment (tenant, role, query category,
model, prompt version, index version) so a regression hidden by the overall
average still fires. Rules need a minimum sample count before they can fire.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.telemetry import read_jsonl

DEFAULT_RULES = "monitoring/alerts.yml"
DEFAULT_TRACES = "outputs/traces/traces.jsonl"
DEFAULT_ONLINE = "outputs/traces/online_eval.jsonl"
DEFAULT_FEEDBACK = "outputs/traces/feedback.jsonl"


def load_rules(path: str) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def _ts(row: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(row.get("timestamp")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _pct(values: list) -> float | None:
    return round(100 * sum(values) / len(values), 2) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


def compute_metrics(traces: list[dict], evals: dict[str, dict], feedback: dict[str, list[dict]]) -> dict:
    """Return {metric: (value, n_samples)}. Rates are 0–100."""
    ev = [evals[t["request_id"]] for t in traces if t["request_id"] in evals]

    def ev_metric(key):
        vals = [e[key] for e in ev if e.get(key) is not None]
        return (_pct(vals) if vals else None, len(vals))

    n = len(traces)
    fb = [f for t in traces for f in feedback.get(t["request_id"], [])]

    def check_rate(key):
        return (_pct([bool(t.get("checks", {}).get(key)) for t in traces]), n)

    temporal = [t.get("temporal") or {} for t in traces]
    latest = [x["latest_is_newest"] for x in temporal if x.get("latest_is_newest") is not None]
    acct_q = [x for x in temporal if x.get("account_resolution")]
    lat = [t["latency_ms"]["total"] for t in traces if t.get("latency_ms")]
    cost = [t["cost_usd"] for t in traces if t.get("cost_usd") is not None]
    return {
        "requests": (n, n),
        "groundedness": ev_metric("groundedness"),
        "citation_precision": ev_metric("citation_precision"),
        "answer_relevance": ev_metric("answer_relevance"),
        "context_relevance": ev_metric("context_relevance"),
        "refusal_correct": ev_metric("refusal_correct"),
        "refusal_rate": (_pct([bool(t.get("abstained")) for t in traces]), n),
        "empty_retrieval_rate": (_pct([bool(t.get("checks", {}).get("empty_retrieval")) for t in traces]), n),
        "stale_answer_rate": ev_metric("stale_answer"),
        "as_of_leakage_rate": check_rate("as_of_leakage"),
        "account_mixing_rate": check_rate("account_mixing"),
        "stale_revision_rate": check_rate("stale_revision"),
        "truncation_rate": check_rate("truncated"),
        "latest_is_newest_rate": (_pct(latest), len(latest)),
        "account_ambiguous_rate": (_pct([x["account_resolution"] == "multiple" for x in acct_q]), len(acct_q)),
        "negative_feedback_rate": (_pct([f.get("rating") in ("down", "incorrect") for f in fb]), len(fb)),
        "p50_latency_ms": (_percentile(lat, 50), len(lat)),
        "p95_latency_ms": (_percentile(lat, 95), len(lat)),
        "avg_cost_usd": (round(sum(cost) / len(cost), 6) if cost else None, len(cost)),
    }


def _check_rule(rule: dict, cur: dict, base: dict) -> str | None:
    metric, kind, thr = rule["metric"], rule["kind"], rule["threshold"]
    value, n = cur.get(metric, (None, 0))
    if value is None or n < rule.get("min_samples", 0):
        return None
    b = base.get(metric, (None, 0))[0]
    if kind == "below" and value < thr:
        return f"{metric}={value} < {thr}"
    if kind == "above" and value > thr:
        return f"{metric}={value} > {thr}"
    if kind == "drop_pp_vs_baseline" and b is not None and b - value > thr:
        return f"{metric} fell {round(b - value, 2)}pp ({b} → {value}), limit {thr}pp"
    if kind == "ratio_vs_baseline" and b not in (None, 0) and value / b > thr:
        return f"{metric} ×{round(value / b, 2)} vs baseline ({b} → {value}), limit ×{thr}"
    return None


def _index_age_hours(chroma_dir: str, now: datetime) -> float | None:
    try:
        built = json.loads((Path(chroma_dir) / "index_manifest.json").read_text())["built_at"]
        return round((now - datetime.fromisoformat(built)).total_seconds() / 3600, 1)
    except (OSError, KeyError, ValueError):
        return None


def run(rules: dict, traces_path: str, online_path: str, feedback_path: str,
        now: datetime | None = None, chroma_dir: str = "data/chroma") -> dict:
    now = now or datetime.now(timezone.utc)
    win = rules.get("windows", {})
    cur_start = now - timedelta(hours=win.get("current_hours", 24))
    base_start = cur_start - timedelta(days=win.get("baseline_days", 7))

    traces = [t for t in read_jsonl(traces_path) if t.get("type") == "request" and t.get("source") != "eval"]
    evals = {e["request_id"]: e for e in read_jsonl(online_path) if not e.get("error")}
    feedback: dict[str, list] = defaultdict(list)
    for f in read_jsonl(feedback_path):
        feedback[f.get("request_id")].append(f)

    current = [t for t in traces if (ts := _ts(t)) and cur_start <= ts <= now]
    baseline = [t for t in traces if (ts := _ts(t)) and base_start <= ts < cur_start]

    alerts: list[dict] = []

    # Page: any single security event in the current window
    page_checks = {p["check"] for p in rules.get("page", [])}
    for t in current:
        c = t.get("checks", {})
        hits = [k for k in ("tenant_violation", "rbac_violation") if k in page_checks and c.get(k)]
        if "severe_pii" in page_checks and {"SSN", "CREDIT_CARD"} & set(c.get("pii_in_answer", [])):
            hits.append("severe_pii")
        for h in hits:
            alerts.append({"severity": "page", "rule": h, "segment": "overall",
                           "detail": f"request {t['request_id']} (tenant={t.get('tenant_id')}, role={t.get('role')})"})

    # Warn: trend rules, overall + per segment
    segments: list[tuple[str, list, list]] = [("overall", current, baseline)]
    for dim in rules.get("segment_by", []):
        for val in sorted({str(t.get(dim)) for t in current}):
            segments.append((f"{dim}={val}",
                             [t for t in current if str(t.get(dim)) == val],
                             [t for t in baseline if str(t.get(dim)) == val]))

    seg_metrics = {}
    for name, cur_rows, base_rows in segments:
        cur_m = compute_metrics(cur_rows, evals, feedback)
        base_m = compute_metrics(base_rows, evals, feedback)
        if name == "overall":
            age = _index_age_hours(chroma_dir, now)
            cur_m["index_age_hours"] = (age, 1 if age is not None else 0)
        seg_metrics[name] = {"current": {k: v[0] for k, v in cur_m.items()},
                             "baseline": {k: v[0] for k, v in base_m.items()},
                             "samples": {k: v[1] for k, v in cur_m.items()}}
        for rule in rules.get("warn", []):
            if rule.get("overall_only") and name != "overall":
                continue
            msg = _check_rule(rule, cur_m, base_m)
            if msg:
                alerts.append({"severity": "warn", "rule": rule["name"], "segment": name, "detail": msg})

    return {"generated_at": now.isoformat(), "window": {"current_from": cur_start.isoformat(),
            "baseline_from": base_start.isoformat()}, "alerts": alerts, "metrics": seg_metrics}


def print_report(report: dict) -> None:
    o = report["metrics"]["overall"]
    print("=" * 64)
    print("  RAG MONITOR — overall, current window vs baseline")
    print("=" * 64)
    for k, v in o["current"].items():
        print(f"  {k:<24} {str(v):>12}   baseline {str(o['baseline'].get(k)):>12}   n={o['samples'].get(k)}")
    print("-" * 64)
    if not report["alerts"]:
        print("  ✅ no alerts")
    for a in report["alerts"]:
        icon = "🚨 PAGE" if a["severity"] == "page" else "⚠️  WARN"
        print(f"  {icon}  {a['rule']}  [{a['segment']}]  {a['detail']}")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser(description="Compute live RAG metrics and evaluate alert rules")
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--traces", default=DEFAULT_TRACES)
    p.add_argument("--online", default=DEFAULT_ONLINE)
    p.add_argument("--feedback", default=DEFAULT_FEEDBACK)
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--json", default=None, help="also write the full report here")
    p.add_argument("--fail-on-warn", action="store_true")
    a = p.parse_args()

    report = run(load_rules(a.rules), a.traces, a.online, a.feedback, chroma_dir=a.chroma_dir)
    print_report(report)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2))
    sev = {x["severity"] for x in report["alerts"]}
    sys.exit(2 if "page" in sev else 1 if ("warn" in sev and a.fail_on_warn) else 0)


if __name__ == "__main__":
    main()
