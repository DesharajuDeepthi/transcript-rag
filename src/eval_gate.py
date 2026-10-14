"""
eval_gate.py — block a release when the offline golden-set eval regresses.

  python -m src.eval_gate                                   # check outputs/eval_report.json
  python -m src.eval_gate --report outputs/eval_report.json --baseline data/evals/baselines/golden-v2.json
  python -m src.eval_gate --update-baseline                 # approve current numbers as the new baseline

Run it after eval_run + eval_compare whenever retrieval, prompts, models,
chunking or the index change (CI, before promoting a new index, nightly).
Thresholds live in monitoring/alerts.yml under `release_gate`.
Exit code 1 = gate failed → do not deploy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPORT = "outputs/eval_report.json"
DEFAULT_RULES = "monitoring/alerts.yml"
BASELINE_DIR = Path("data/evals/baselines")

_SUMMARY_KEYS = {  # older reports only have formatted strings in `summary`
    "hit_at_k": "hit_at_k", "mrr": "mrr", "faithfulness": "faithfulness",
    "answer_correctness": "answer_correctness", "correctness_pass_rate": "correctness_pass_rate",
    "avg_answer_quality": "avg_answer_quality", "abstention_accuracy": "abstention_accuracy",
    "grader_errors": "grader_errors",
}


def report_metrics(report: dict) -> dict:
    """Numeric metrics from a report (new `metrics` block, or parsed from `summary` strings)."""
    if isinstance(report.get("metrics"), dict):
        return dict(report["metrics"])
    out = {}
    for src, dst in _SUMMARY_KEYS.items():
        v = report.get("summary", {}).get(src)
        if isinstance(v, (int, float)):
            out[dst] = float(v)
        elif isinstance(v, str) and (m := re.match(r"\s*(-?[\d.]+)", v)):
            out[dst] = float(m.group(1))
    return out


def check(metrics: dict, baseline: dict | None, gate: dict) -> list[dict]:
    results = []

    def add(name, ok, detail):
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    def val(k):
        return metrics.get(k)

    if (v := val("hit_at_k")) is not None:
        add("hit_at_k", v >= gate["hit_at_k_min"], f"{v} ≥ {gate['hit_at_k_min']}")
    if (v := val("faithfulness")) is not None:
        add("faithfulness", v >= gate["faithfulness_min"], f"{v} ≥ {gate['faithfulness_min']}")
    if (v := val("abstention_accuracy")) is not None:
        add("abstention_accuracy", v >= gate["abstention_accuracy_min"], f"{v} ≥ {gate['abstention_accuracy_min']}")
    if "security_violations" in metrics:
        v = metrics["security_violations"]
        add("security_violations", v <= gate["security_violations_max"], f"{v} ≤ {gate['security_violations_max']}")
    else:
        add("security_violations", False, "report has no security_violations metric — re-run eval_compare")
    if (v := val("grader_errors")) is not None:
        add("grader_errors", v <= gate["grader_errors_max"], f"{v} ≤ {gate['grader_errors_max']}")
    for metric, key in (("run_errors", "run_errors_max"), ("truncated_answers", "truncated_answers_max")):
        if (v := val(metric)) is not None and key in gate:
            add(metric, v <= gate[key], f"{v} ≤ {gate[key]}")

    for metric, key, op in (("temporal_intent_accuracy", "temporal_intent_accuracy_min", ">="),
                            ("account_resolution_accuracy", "account_resolution_accuracy_min", ">="),
                            ("latest_selection_accuracy", "latest_selection_accuracy_min", ">="),
                            ("as_of_leakage_rate", "as_of_leakage_rate_max", "<="),
                            ("account_mixing_rate", "account_mixing_rate_max", "<="),
                            ("stale_answer_rate", "stale_answer_rate_max", "<=")):
        if (v := val(metric)) is not None and key in gate:
            ok = v >= gate[key] if op == ">=" else v <= gate[key]
            add(metric, ok, f"{v} {op} {gate[key]}")

    if baseline:
        b = baseline.get("metrics", baseline)
        for metric, key in (("mrr", "mrr_max_drop_pp"), ("answer_correctness", "answer_correctness_max_drop_pp")):
            if val(metric) is not None and b.get(metric) is not None:
                drop = round(b[metric] - val(metric), 2)
                add(f"{metric}_vs_baseline", drop <= gate[key],
                    f"{b[metric]} → {val(metric)} (drop {drop}pp, limit {gate[key]}pp)")
    else:
        add("baseline", True, "no baseline yet — relative checks skipped (run with --update-baseline to approve)")
    return results


def baseline_path_for(report: dict) -> Path:
    version = (report.get("run") or {}).get("golden_version") or "golden"
    return BASELINE_DIR / f"{version}.json"


def main():
    p = argparse.ArgumentParser(description="Release gate for the offline golden-set eval")
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--baseline", default=None)
    p.add_argument("--update-baseline", action="store_true",
                   help="write the current metrics as the approved baseline (after a human review)")
    a = p.parse_args()

    import yaml
    gate = yaml.safe_load(Path(a.rules).read_text())["release_gate"]
    report = json.loads(Path(a.report).read_text())
    metrics = report_metrics(report)
    bpath = Path(a.baseline) if a.baseline else baseline_path_for(report)
    baseline = json.loads(bpath.read_text()) if bpath.exists() else None

    results = check(metrics, baseline, gate)
    print("=" * 60)
    print(f"  RELEASE GATE  (baseline: {bpath if baseline else 'none'})")
    print("=" * 60)
    for r in results:
        print(f"  {'✅' if r['passed'] else '❌'} {r['check']:<30} {r['detail']}")
    failed = [r for r in results if not r["passed"]]
    print("=" * 60)
    print("  GATE FAILED — do not deploy" if failed else "  GATE PASSED")

    if a.update_baseline:
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps({
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "source_report": a.report,
            "run": report.get("run"),
            "metrics": metrics,
        }, indent=2))
        print(f"  Baseline written to {bpath}")
    sys.exit(1 if failed and not a.update_baseline else 0)


if __name__ == "__main__":
    main()
