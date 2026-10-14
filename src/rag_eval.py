"""
rag_eval.py — Evaluation suite for the RAG chatbot.

Covers every dimension from the RAG evaluation framework:
  2.1  Retrieval hit rate (hit@k, MRR)
  2.2  Context relevance & sufficiency  (LLM-graded)
  2.3  Groundedness / faithfulness      (LLM-graded, claim-by-claim)
  2.4  Answer quality scorecard         (correctness, completeness, clarity, citations)
  2.5  Abstention on unanswerable Qs
  2.6  Stale / contradictory sources    (behavior check)
  2.7  Failure diagnosis                (retrieval vs generation)

Usage:
  python -m src.rag_eval                    # run full suite, print report
  python -m src.rag_eval --questions 20     # use N test questions
  python -m src.rag_eval --out results.json # save results
  python -m src.rag_eval --gen-only         # only generate test set, save and exit
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(os.getenv("DATASET_DIR", "/app/data/dataset"))
CHROMA_DIR   = os.getenv("CHROMA_DIR", "/app/data/chroma")
COLLECTION   = "transcripts"
EMBED_MODEL  = "text-embedding-3-small"
EVAL_MODEL   = "claude-haiku-5-5"   # grader — cheap, fast
TOP_K        = 6

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about meeting transcripts and summaries.
Answer ONLY from the retrieved context. If the context does not contain enough information, say:
"I don't have enough information in the retrieved meetings to answer this confidently."
Always cite the meeting title and date for every fact you state."""


# ── Clients ───────────────────────────────────────────────────────────────────

def _embed_fn():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing")
    return OpenAIEmbeddingFunction(api_key=key, model_name=EMBED_MODEL)


def _anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY missing")
    return anthropic.Anthropic(api_key=key)


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION, embedding_function=_embed_fn())


# ── Test-set generation ───────────────────────────────────────────────────────

def _load_meeting_summaries(n: int) -> list[dict]:
    """Pick n random meetings and load their summary + meeting-info."""
    dirs = sorted(DATASET_DIR.iterdir())
    random.shuffle(dirs)
    meetings = []
    for d in dirs:
        info_path    = d / "meeting-info.json"
        summary_path = d / "summary.json"
        if not info_path.exists() or not summary_path.exists():
            continue
        info    = json.loads(info_path.read_text())
        summary = json.loads(summary_path.read_text())
        meetings.append({"meeting_id": d.name, "info": info, "summary": summary})
        if len(meetings) >= n:
            break
    return meetings


def generate_test_set(n_answerable: int = 20, n_unanswerable: int = 5) -> list[dict]:
    """
    Auto-generate a labeled test set from the real dataset.

    Each answerable question has:
      - question        : natural-language question
      - expected_answer : short reference answer
      - gold_meeting_id : which meeting contains the answer
      - gold_title      : meeting title (for display)
      - answerable      : True

    Unanswerable questions have answerable=False and no gold meeting.
    """
    client  = _anthropic()
    meetings = _load_meeting_summaries(n_answerable)
    test_set = []

    print(f"  Generating {n_answerable} answerable questions via LLM...")
    for m in tqdm(meetings, desc="  Generating Qs"):
        title   = m["info"].get("title", "")
        summary = m["summary"].get("summary", "")
        topics  = ", ".join(m["summary"].get("topics", [])[:4])
        actions = "; ".join(m["summary"].get("actionItems", [])[:3])

        prompt = f"""Given this meeting:
Title: {title}
Summary: {summary}
Topics: {topics}
Action items: {actions}

Write ONE specific factual question a user might ask about this meeting,
and a short (1-2 sentence) reference answer grounded in the summary.

Respond in JSON only, no markdown:
{{"question": "...", "expected_answer": "..."}}"""

        resp = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # strip markdown fences if present
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        try:
            qa = json.loads(raw)
            test_set.append({
                "question":        qa["question"],
                "expected_answer": qa["expected_answer"],
                "gold_meeting_id": m["meeting_id"],
                "gold_title":      title,
                "answerable":      True,
            })
        except Exception as e:
            print(f"  [WARN] Could not parse Q for {title}: {e}")

    # Add unanswerable questions
    unanswerable = [
        "What was the Q3 2099 revenue forecast discussed?",
        "Which meeting covered the Mars office expansion plan?",
        "What did the CEO say about the 2099 acquisition target?",
        "What is the policy for quantum computing hardware requests?",
        "What were the conclusions from the interdimensional team sync?",
    ]
    for q in unanswerable[:n_unanswerable]:
        test_set.append({
            "question":        q,
            "expected_answer": None,
            "gold_meeting_id": None,
            "gold_title":      None,
            "answerable":      False,
        })

    return test_set


# ── Retrieval evaluation (2.1) ────────────────────────────────────────────────

def eval_retrieval(collection, test_set: list[dict], top_k: int = TOP_K) -> dict:
    """
    Measure hit@k and MRR on answerable questions.
    hit@k = 1 if gold_meeting_id appears in top-k results, else 0.
    MRR   = 1/rank of first hit (0 if not found).
    """
    hits, reciprocal_ranks = [], []
    details = []

    answerable = [t for t in test_set if t["answerable"]]
    for item in tqdm(answerable, desc="  Eval retrieval"):
        results = collection.query(
            query_texts=[item["question"]],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        retrieved_ids = [m.get("meeting_id", "") for m in results["metadatas"][0]]
        gold = item["gold_meeting_id"]

        hit = int(gold in retrieved_ids)
        rank = retrieved_ids.index(gold) + 1 if gold in retrieved_ids else 0
        rr   = 1 / rank if rank > 0 else 0.0

        hits.append(hit)
        reciprocal_ranks.append(rr)
        details.append({
            "question":   item["question"],
            "gold":       gold,
            "retrieved":  retrieved_ids,
            "hit":        bool(hit),
            "rank":       rank,
            "rr":         rr,
        })

    return {
        "hit_at_k":  round(sum(hits) / len(hits), 3) if hits else 0,
        "mrr":       round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3) if reciprocal_ranks else 0,
        "n":         len(answerable),
        "details":   details,
    }


# ── Retrieve + generate answer ────────────────────────────────────────────────

def retrieve_and_answer(collection, client, question: str, top_k: int = TOP_K) -> dict:
    """Run retrieval and generation for one question. Returns answer + context."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        relevance = round((1 - dist) * 100, 1)
        context_parts.append(
            f"[{meta.get('title','')} | {meta.get('date','')} | {relevance}%]\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    augmented = f"<retrieved_context>\n{context}\n</retrieved_context>\n\nQuestion: {question}"
    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": augmented}],
    )
    answer = resp.content[0].text.strip()
    return {"answer": answer, "context": context, "retrieved_metas": metas}


# ── LLM grader helper ──────────────────────────────────────────────────────────

def _grade(client, prompt: str) -> dict:
    """Call Claude as grader; returns parsed JSON."""
    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw, "parse_error": True}


# ── Groundedness (2.3) ────────────────────────────────────────────────────────

def eval_groundedness(client, answer: str, context: str) -> dict:
    """
    Decompose answer into claims, check each against context.
    Returns: groundedness score (0-1), list of claims with support status.
    """
    prompt = f"""You are an evaluator checking if an AI answer is grounded in the provided context.

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

Task:
1. Split the answer into individual factual claims (ignore filler phrases).
2. For each claim, check if it is supported by the context.
3. Return JSON only:
{{
  "claims": [
    {{"claim": "...", "supported": true/false, "evidence": "brief quote or 'not found'"}},
    ...
  ],
  "groundedness_score": <fraction of supported claims, 0.0-1.0>
}}"""
    return _grade(client, prompt)


# ── Answer quality scorecard (2.4) ────────────────────────────────────────────

def eval_answer_quality(client, question: str, answer: str, context: str,
                        expected_answer: str | None = None) -> dict:
    """
    Score on 6 dimensions (0=fails, 1=partly, 2=fully):
    correctness, relevance, completeness, clarity, groundedness, citations
    """
    ref_line = f"Reference answer: {expected_answer}" if expected_answer else "No reference answer provided."
    prompt = f"""You are an evaluator scoring an AI answer on 6 dimensions (0=fails, 1=partly, 2=fully).

QUESTION: {question}
{ref_line}

CONTEXT (what was retrieved):
{context[:2000]}

AI ANSWER:
{answer}

Score each dimension and return JSON only:
{{
  "correctness":   {{"score": 0-2, "reason": "..."}},
  "relevance":     {{"score": 0-2, "reason": "..."}},
  "completeness":  {{"score": 0-2, "reason": "..."}},
  "clarity":       {{"score": 0-2, "reason": "..."}},
  "groundedness":  {{"score": 0-2, "reason": "..."}},
  "citations":     {{"score": 0-2, "reason": "..."}},
  "total":         <sum of all scores, max 12>
}}"""
    return _grade(client, prompt)


# ── Abstention check (2.5) ────────────────────────────────────────────────────

def eval_abstention(answer: str) -> dict:
    """
    Check if the answer appropriately abstains for unanswerable questions.
    Looks for the known abstention phrase or similar wording.
    """
    abstention_phrases = [
        "don't have enough information",
        "not enough information",
        "cannot find",
        "couldn't find",
        "no information",
        "not in the retrieved",
        "i don't know",
        "not available",
    ]
    lower = answer.lower()
    abstained = any(p in lower for p in abstention_phrases)
    return {"abstained": abstained, "answer_snippet": answer[:200]}


# ── Failure diagnosis (2.7) ───────────────────────────────────────────────────

def diagnose_failure(hit: bool, answer_quality_total: int, max_score: int = 12) -> str:
    """
    Given hit@k result and answer quality:
    - no hit + bad answer  → retrieval failure
    - hit + bad answer     → generation failure
    - hit + good answer    → passing
    - no hit + good answer → possibly good memory/generalization (flag it)
    """
    threshold = max_score * 0.5   # below 50% = bad answer
    bad_answer = answer_quality_total < threshold

    if not hit and bad_answer:
        return "RETRIEVAL_FAILURE"
    elif hit and bad_answer:
        return "GENERATION_FAILURE"
    elif not hit and not bad_answer:
        return "SUSPICIOUS_PASS"   # answered correctly without the right chunk?
    else:
        return "PASS"


# ── Full evaluation run ────────────────────────────────────────────────────────

def run_eval(test_set: list[dict], top_k: int = TOP_K) -> dict:
    """Run the complete evaluation suite and return a structured report."""
    collection = _get_collection()
    client     = _anthropic()

    # ── 2.1  Retrieval hit rate ────────────────────────────────────────────
    print("\n[2.1] Evaluating retrieval (hit@k, MRR)...")
    retrieval_results = eval_retrieval(collection, test_set, top_k)
    hit_lookup = {d["question"]: d["hit"] for d in retrieval_results["details"]}

    # ── 2.3 / 2.4 / 2.5  Generate + grade each question ──────────────────
    print("\n[2.3/2.4/2.5] Generating answers and grading...")
    per_question = []
    abstention_results = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # 2×2 table

    for item in tqdm(test_set, desc="  Grading"):
        q         = item["question"]
        answerable = item["answerable"]

        gen = retrieve_and_answer(collection, client, q, top_k)
        answer  = gen["answer"]
        context = gen["context"]

        # Groundedness (2.3)
        ground = eval_groundedness(client, answer, context)

        # Answer quality (2.4)
        quality = eval_answer_quality(client, q, answer, context,
                                      expected_answer=item.get("expected_answer"))

        # Abstention (2.5)
        abstention = eval_abstention(answer)
        abstained  = abstention["abstained"]

        # Fill 2×2 abstention table
        if answerable and not abstained:
            abstention_results["tn"] += 1   # correctly answered an answerable Q
        elif answerable and abstained:
            abstention_results["fn"] += 1   # over-cautious: refused an answerable Q
        elif not answerable and not abstained:
            abstention_results["fp"] += 1   # hallucinated on unanswerable Q (bad)
        else:
            abstention_results["tp"] += 1   # correctly abstained (good)

        # Failure diagnosis (2.7)
        hit   = hit_lookup.get(q, False) if answerable else False
        total = quality.get("total", 0) if isinstance(quality.get("total"), int) else 0
        diagnosis = diagnose_failure(hit, total) if answerable else "UNANSWERABLE"

        per_question.append({
            "question":    q,
            "answerable":  answerable,
            "gold_title":  item.get("gold_title"),
            "answer":      answer,
            "hit":         hit if answerable else None,
            "groundedness": ground,
            "quality":     quality,
            "abstention":  abstention,
            "diagnosis":   diagnosis,
        })

    # ── Aggregate scores ───────────────────────────────────────────────────
    answerable_qs = [r for r in per_question if r["answerable"]]
    quality_scores = []
    ground_scores  = []
    for r in answerable_qs:
        q_total = r["quality"].get("total")
        if isinstance(q_total, int):
            quality_scores.append(q_total)
        g_score = r["groundedness"].get("groundedness_score")
        if isinstance(g_score, (int, float)):
            ground_scores.append(g_score)

    # Abstention accuracy
    tp = abstention_results["tp"]
    fp = abstention_results["fp"]
    abstention_accuracy = tp / max(tp + fp, 1)

    # Failure breakdown
    diagnoses = [r["diagnosis"] for r in answerable_qs]
    failure_summary = {
        "PASS":               diagnoses.count("PASS"),
        "RETRIEVAL_FAILURE":  diagnoses.count("RETRIEVAL_FAILURE"),
        "GENERATION_FAILURE": diagnoses.count("GENERATION_FAILURE"),
        "SUSPICIOUS_PASS":    diagnoses.count("SUSPICIOUS_PASS"),
    }

    return {
        "summary": {
            "n_total":             len(test_set),
            "n_answerable":        len(answerable_qs),
            "n_unanswerable":      len(test_set) - len(answerable_qs),
            "retrieval_hit_at_k":  retrieval_results["hit_at_k"],
            "retrieval_mrr":       retrieval_results["mrr"],
            "avg_quality_score":   round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
            "avg_groundedness":    round(sum(ground_scores)  / len(ground_scores),  3) if ground_scores  else 0,
            "abstention_accuracy": round(abstention_accuracy, 3),
            "abstention_table":    abstention_results,
            "failure_breakdown":   failure_summary,
        },
        "retrieval": retrieval_results,
        "per_question": per_question,
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(report: dict):
    s = report["summary"]
    print("\n" + "═" * 60)
    print("  RAG EVALUATION REPORT")
    print("═" * 60)
    print(f"  Questions:          {s['n_total']} ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)")
    print(f"\n  ── Retrieval (2.1) ──────────────────────────")
    print(f"  hit@{TOP_K}:             {s['retrieval_hit_at_k']:.1%}")
    print(f"  MRR:               {s['retrieval_mrr']:.3f}")
    print(f"\n  ── Answer quality (2.3/2.4) ─────────────────")
    print(f"  Avg quality score: {s['avg_quality_score']}/12")
    print(f"  Avg groundedness:  {s['avg_groundedness']:.1%}")
    print(f"\n  ── Abstention (2.5) ──────────────────────────")
    t = s["abstention_table"]
    print(f"  Correctly abstained (unanswerable → refused): {t['tp']}")
    print(f"  Hallucinated       (unanswerable → answered): {t['fp']}  ← bad")
    print(f"  Over-cautious      (answerable → refused):    {t['fn']}  ← bad")
    print(f"  Correctly answered (answerable → answered):   {t['tn']}")
    print(f"  Abstention accuracy: {s['abstention_accuracy']:.1%}")
    print(f"\n  ── Failure diagnosis (2.7) ───────────────────")
    fb = s["failure_breakdown"]
    print(f"  PASS:               {fb['PASS']}")
    print(f"  RETRIEVAL_FAILURE:  {fb['RETRIEVAL_FAILURE']}")
    print(f"  GENERATION_FAILURE: {fb['GENERATION_FAILURE']}")
    print(f"  SUSPICIOUS_PASS:    {fb['SUSPICIOUS_PASS']}")
    print("═" * 60)

    # Show worst 3 failures
    failures = [r for r in report["per_question"]
                if r["diagnosis"] in ("RETRIEVAL_FAILURE", "GENERATION_FAILURE")]
    if failures:
        print("\n  Worst failures to fix:")
        for f in failures[:3]:
            print(f"  [{f['diagnosis']}] {f['question'][:80]}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",  type=int, default=20, help="Number of answerable test Qs")
    parser.add_argument("--unanswerable", type=int, default=5, help="Number of unanswerable Qs")
    parser.add_argument("--top-k",     type=int, default=TOP_K)
    parser.add_argument("--out",       default=None, help="Save JSON report to this path")
    parser.add_argument("--gen-only",  action="store_true", help="Only generate test set, print and exit")
    parser.add_argument("--test-file", default=None, help="Load existing test set JSON instead of generating")
    args = parser.parse_args()

    # Load or generate test set
    if args.test_file:
        print(f"  Loading test set from {args.test_file}")
        test_set = json.loads(Path(args.test_file).read_text())
    else:
        print(f"  Generating test set ({args.questions} answerable + {args.unanswerable} unanswerable)...")
        test_set = generate_test_set(args.questions, args.unanswerable)
        # Always save test set so you can reuse it
        ts_path = Path(args.out).parent / "test_set.json" if args.out else Path("/app/outputs/test_set.json")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        ts_path.write_text(json.dumps(test_set, indent=2))
        print(f"  Test set saved → {ts_path}")

    if args.gen_only:
        for item in test_set:
            mark = "✓" if item["answerable"] else "✗"
            print(f"  [{mark}] {item['question']}")
        return

    # Run evaluation
    report = run_eval(test_set, top_k=args.top_k)
    print_report(report)

    # Save report
    out_path = args.out or "/app/outputs/eval_report.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Full report saved → {out_path}")


if __name__ == "__main__":
    main()
