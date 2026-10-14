"""
eval_run.py — run a versioned golden set through the SAME pipeline the chatbot uses.

Usage:
  python -m src.eval_run                                         # data/evals/golden-v3.json
  python -m src.eval_run --golden data/evals/golden-v1.json
  python -m src.eval_run --top-k 10 --prompt-version answer-v1   # try a candidate config
  python -m src.eval_run --out outputs/my_results.json

What it does:
  1. Loads the golden set. Items may set "as_user" (an employee email) to run the
     question with that person's role — used for RBAC / unauthorised-access cases.
  2. For each question calls RAGPipeline.answer() — identical retrieval, filters,
     neighbour expansion, prompt and model as src/rag_chat.py.
  3. Saves {"run": <config + versions>, "results": [...]} so every metric in the
     report can be traced back to the exact configuration that produced it.
  4. Run eval_compare.py next, then eval_gate.py to decide whether to ship.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.rag_pipeline import Principal, RAGConfig, RAGPipeline
from src.telemetry import Tracer

load_dotenv()

DEFAULT_GOLDEN = "data/evals/golden-v3.json"

# Optional golden-set labels for account / temporal questions (see data/evals/README.md)
TEMPORAL_LABELS = ("expected_account_id", "expected_temporal_mode", "latest_meeting_id", "expected_meeting_ids",
                   "forbidden_before", "forbidden_after", "forbidden_account_ids", "expected_source_version",
                   "outdated_facts")
EVAL_TRACE_PATH = "outputs/traces/eval_traces.jsonl"   # kept apart from live traffic


def golden_version(path: str, items: list) -> str:
    stem = Path(path).stem
    digest = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:8]
    return stem if stem.startswith("golden-v") else f"{stem}@{digest}"


def principal_for(item: dict, tenant_id: str) -> Principal:
    """
    Golden items with "as_user" run as that employee. This is the eval harness deliberately
    SIMULATING a login (allow=True) — the tenant still comes from the minted token, and it
    must match the tenant being evaluated or the run stops (misconfiguration, not a pass).
    """
    if item.get("as_user"):
        p = Principal.dev_login(item["as_user"], allow=True)
        if p.tenant_id != tenant_id:
            raise SystemExit(f"{item['id']}: {item['as_user']} belongs to tenant {p.tenant_id!r} but the eval "
                             f"runs against {tenant_id!r}. Index and evaluate that tenant (--tenant {p.tenant_id}).")
        return p
    return Principal(tenant_id=tenant_id)     # tenant isolation only, no RBAC


def run(golden_path: str, out_path: str, config: RAGConfig, tenant_id: str):
    print(f"\n📂 Loading golden set from: {golden_path}")
    golden_set = json.loads(Path(golden_path).read_text())
    gv = golden_version(golden_path, golden_set)
    print(f"   {len(golden_set)} questions  ({gv})\n")

    pipeline = RAGPipeline(config, tracer=Tracer(path=EVAL_TRACE_PATH, sample_rate=0.0))
    print(f"✅ Index connected: {pipeline.collection.count()} chunks  (index {config.index_version})")
    print(f"   model={config.chat_model} top_k={config.top_k} prompt={config.prompt_version} "
          f"({config.prompt_sha}) tenant={tenant_id}\n")

    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "golden_path": golden_path,
        "golden_version": gv,
        "tenant_id": tenant_id,
        "config": config.describe(),
    }
    results = []

    for item in golden_set:
        qid, question = item["id"], item["question"]
        filters = item.get("filters", {})
        principal = principal_for(item, tenant_id)
        print(f"▶ [{qid}] {question[:80]}..." + (f"  (as {principal.role})" if principal.role else ""))

        try:
            res = pipeline.answer(question, principal, filters=filters, source="eval",
                                  request_id=f"eval-{gv}-{qid}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")
        except RuntimeError as e:   # generation failed after retries — record it, keep going
            print(f"  ⚠ {e}\n")
            results.append({"id": qid, "question": question, "filters": filters,
                            "answerable": item.get("answerable", True), "error": str(e),
                            "gold_meeting_id": item.get("gold_meeting_id"),
                            "expected_answer": item.get("expected_answer"),
                            "actual_answer": "", "retrieved_meeting_ids": []})
            continue
        print(f"  Answer: {res.answer[:100]}...  ({res.latency_ms['total']:.0f} ms)\n")

        trace = res.trace or {}
        results.append({
            "id": qid,
            "question": question,
            "category": item.get("category"),
            "filters": filters,
            "as_user": item.get("as_user"),
            "role": principal.role,
            "answerable": item.get("answerable", True),
            # Golden reference
            "gold_meeting_id": item.get("gold_meeting_id"),
            "gold_title": item.get("gold_title"),
            "expected_answer": item.get("expected_answer"),
            # Actual outputs
            "actual_answer": res.answer,
            "abstained": res.abstained,
            "truncated": res.truncated,
            "stop_reason": res.usage.get("stop_reason"),
            "citations": res.citations,
            "retrieved_context": res.retrieval.context,          # exactly what the LLM saw
            "retrieved_chunks": [{
                "rank": c.rank, "chunk_id": c.chunk_id, "meeting_id": c.meeting_id,
                "title": c.title, "date": c.date, "relevance_pct": c.relevance_pct,
                "chunk_type": c.chunk_type, "speaker": c.speaker, "time_start": c.time_start,
                "access_level": c.access_level, "tenant_id": c.tenant_id,
                "account_id": c.account_id, "date_int": c.date_int,
                "source_version": c.source_version, "is_current_revision": c.is_current_revision,
                "text_preview": c.text[:200], "text": c.text,
            } for c in res.retrieval.chunks],
            "retrieved_meeting_ids": [c.meeting_id for c in res.retrieval.chunks],
            # Temporal / account understanding
            "plan": res.retrieval.plan,
            "primary_meeting_id": res.retrieval.primary_meeting_id,
            "timeline_meeting_ids": res.retrieval.timeline_meeting_ids,
            **{k: item[k] for k in TEMPORAL_LABELS if k in item},
            # Ops
            "request_id": res.request_id,
            "latency_ms": res.latency_ms,
            "tokens": res.usage,
            "cost_usd": trace.get("cost_usd"),
            "security_checks": trace.get("checks"),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
        })

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"run": run_meta, "results": results}, indent=2))
    print(f"\n✅ Results saved to: {out_path}")
    print("   Next: python -m src.eval_compare && python -m src.eval_gate\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run golden set through the shared RAG pipeline")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default="outputs/eval_results.json")
    parser.add_argument("--tenant", default=os.getenv("TENANT_ID", "aegis"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--model", default=None, help="override generation model")
    args = parser.parse_args()

    cfg = RAGConfig.from_env(top_k=args.top_k, prompt_version=args.prompt_version, chat_model=args.model)
    run(args.golden, args.out, cfg, args.tenant)


if __name__ == "__main__":
    main()
