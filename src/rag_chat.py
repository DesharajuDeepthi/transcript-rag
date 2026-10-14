"""
RAG Chatbot with metadata filtering + multi-tenant isolation + RBAC.
Run: python -m src.rag_chat [--top-k N] [--call-type support|external|internal]
     [--sentiment positive|negative|neutral] [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD]
     [--speaker NAME] [--token JWT | --email USER@DOMAIN (AUTH_MODE=dev)] [--tenant ID (dev only)]

All retrieval, prompting and generation happens in src/rag_pipeline.RAGPipeline —
the same code path the offline evaluator (src/eval_run.py) measures. This file
is only the terminal UI around it.

Multi-tenant:
  Every query is scoped to the tenant in the caller's verified token. Other tenants' data is
  never returned, and the tenant cannot be chosen by the caller.

RBAC:
  With --token the JWT is verified and its role is AND'd into every query, including
  neighbour-turn fetches. --email is a dev-only login simulation (AUTH_MODE=dev). Without either
  the chatbot runs in dev mode (tenant filter only).

Telemetry:
  Each question writes a privacy-safe trace to outputs/traces/traces.jsonl.
  Use /feedback up|down [comment] to rate the last answer.
"""

import argparse
import os
import sys

from src.rag_pipeline import ALLOWED_FILTERS, Principal, RAGConfig, RAGPipeline
from src.roles import get_allowed_levels
from src.telemetry import Tracer


def print_active_filters(tenant: str, filters: dict):
    active = [f"tenant={tenant}"] + [f"{k}={v}" for k, v in filters.items() if v]
    print(f"  Active filters: {', '.join(active)}")


def show_help():
    print("""
Commands:
  /clear              — clear conversation history
  /history            — show current conversation history
  /filters            — show active metadata filters
  /filter <key>=<val> — add a filter on the fly
                        keys: call_type, sentiment, date_from, date_to, speaker
  /unfilter <key>     — remove a filter
  /unfilter all       — remove all filters (tenant_id cannot be removed)
  /sources            — show the sources behind the last answer
  /plan               — show how the last question was understood (account, time window, mode)
  /feedback up|down [comment] — rate the last answer
  /help               — show this message
  quit / exit         — exit

Note: tenant_id and your role are always applied and cannot be changed mid-session.
""")


def resolve_principal(token: str | None, email: str | None, tenant_flag: str | None) -> Principal:
    """
    Who is asking. The tenant is taken ONLY from a verified token:
      --token / RAG_TOKEN  → verified JWT (normal path)
      --email              → dev-only simulation, refused unless AUTH_MODE=dev
      neither              → unauthenticated dev mode (tenant flag, no RBAC), AUTH_MODE=dev only
    A --tenant that disagrees with the token is rejected rather than silently ignored.
    """
    from src.auth import auth_mode
    if token or email:
        principal = Principal.from_token(token) if token else Principal.dev_login(email)
        if tenant_flag and tenant_flag != principal.tenant_id:
            raise PermissionError(f"--tenant {tenant_flag!r} does not match the tenant in your token "
                                  f"({principal.tenant_id!r}); the tenant cannot be overridden.")
        return principal
    if auth_mode() != "dev":
        raise PermissionError("Authentication required: pass --token (or set RAG_TOKEN). "
                              "Unauthenticated access is only allowed with AUTH_MODE=dev.")
    return Principal(tenant_id=tenant_flag or os.getenv("TENANT_ID", "aegis"))


def chat_loop(args):
    cfg = RAGConfig.from_env(top_k=args.top_k)
    tracer = Tracer()

    try:
        principal = resolve_principal(args.token, args.email, args.tenant)
    except PermissionError as e:
        print(f"❌ Auth error: {e}")
        sys.exit(1)

    print("\n🔍 Connecting to ChromaDB...")
    pipeline = RAGPipeline(cfg, tracer=tracer)
    print(f"✅ Index loaded: {pipeline.collection.count()} chunks  (index {cfg.index_version})")
    print(f"🏢 Tenant: {principal.tenant_id}")
    if principal.role:
        print(f"👤 User: {principal.user}  |  Role: {principal.role}")
        print(f"🔐 Access levels: {', '.join(get_allowed_levels(principal.role))}")
    else:
        print("⚠️  Unauthenticated dev mode (AUTH_MODE=dev): tenant filter only, no RBAC")

    filters = {k: getattr(args, k) for k in ALLOWED_FILTERS}
    print(f"\n💬 RAG Chatbot  (model: {cfg.chat_model}, top-k: {cfg.top_k}, prompt: {cfg.prompt_version})")
    print_active_filters(principal.tenant_id, filters)
    print("  Type /help for commands\n")

    history: list[dict] = []
    last = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input == "/help":
            show_help()
            continue
        if user_input == "/clear":
            history = []
            print("  [History cleared]\n")
            continue
        if user_input == "/history":
            for m in history or []:
                who = "You" if m["role"] == "user" else "Bot"
                print(f"  {who}: {m['content'][:120].replace(chr(10), ' ')}...")
            print("  [No history yet]\n" if not history else "")
            continue
        if user_input == "/filters":
            print_active_filters(principal.tenant_id, filters)
            print()
            continue
        if user_input.startswith("/filter "):
            try:
                key, val = user_input[len("/filter "):].split("=", 1)
                key, val = key.strip().lower(), val.strip()
            except ValueError:
                print("  Usage: /filter key=value\n")
                continue
            if key not in filters:
                print(f"  Unknown filter key '{key}'. Use: {list(filters)}  (tenant and role are fixed)\n")
            else:
                filters[key] = val
                print(f"  Filter set: {key} = {val}\n")
            continue
        if user_input.startswith("/unfilter"):
            target = user_input[len("/unfilter"):].strip()
            if target == "all":
                filters = {k: None for k in filters}
                print("  All filters cleared (tenant and role unchanged)\n")
            elif target in filters:
                filters[target] = None
                print(f"  Filter '{target}' removed\n")
            else:
                print(f"  Unknown key '{target}'. Use: {list(filters)} or 'all'\n")
            continue
        if user_input == "/sources":
            if not last:
                print("  [No answer yet]\n")
                continue
            for c in last.retrieval.chunks:
                t = f" t={c.time_start}s" if c.time_start else ""
                print(f"  [{c.rank}] {c.title} ({c.date}){t}  v{c.source_version}  {c.chunk_id}  rel={c.relevance_pct}%")
            print()
            continue
        if user_input == "/plan":
            if not last:
                print("  [No answer yet]\n")
                continue
            pl = last.retrieval.plan
            print(f"  mode={pl.get('mode')}  account={pl.get('account_id')} ({pl.get('account_resolution')})"
                  f"  from={pl.get('date_from')} to={pl.get('date_to')} as_of={pl.get('as_of')}")
            for reason in pl.get("reasons", []):
                print(f"   · {reason}")
            print(f"  {last.retrieval.plan_note}\n")
            continue
        if user_input.startswith("/feedback"):
            parts = user_input.split(None, 2)
            if not last or len(parts) < 2 or parts[1] not in ("up", "down", "incorrect"):
                print("  Usage: /feedback up|down|incorrect [comment]  (rates the last answer)\n")
                continue
            tracer.record_feedback(last.request_id, parts[1], parts[2] if len(parts) > 2 else "", principal)
            print("  Thanks — feedback recorded.\n")
            continue

        # ── Retrieve & Generate (shared pipeline) ──────────────────────────
        print("Assistant: ", end="", flush=True)
        try:
            last = pipeline.answer(user_input, principal, filters=filters, history=history,
                                   on_token=lambda t: print(t, end="", flush=True), source="chat")
        except ValueError as e:          # bad filter value
            print(f"\n[{e}]\n")
            continue
        except RuntimeError as e:        # LLM call failed after retries
            print(f"\n[Error: {e}]\n")
            continue
        lat = last.latency_ms
        cost = (last.trace or {}).get("cost_usd")
        print(f"\n  ⏱ {lat['total']:.0f} ms (retrieval {lat['retrieval']:.0f})"
              f"  ·  {last.usage['input_tokens']}+{last.usage['output_tokens']} tok"
              + (f"  ·  ${cost:.5f}" if cost is not None else "") + "\n")

        history += [{"role": "user", "content": user_input},
                    {"role": "assistant", "content": last.answer}]
        history = history[-(cfg.max_history_turns * 2):]


def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description="RAG chatbot with metadata filtering")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve (default from RAGConfig)")
    parser.add_argument("--token", default=os.getenv("RAG_TOKEN"), help="Signed JWT (python -m src.auth issue <email>)")
    parser.add_argument("--email", default=None,
                        help="DEV ONLY (AUTH_MODE=dev): simulate login as this employee, e.g. vanessa.cruz@aegis.com")
    parser.add_argument("--tenant", default=None,
                        help="Unauthenticated dev mode only; with a token it must match the token's tenant")
    parser.add_argument("--call-type", dest="call_type", choices=["support", "external", "internal"], default=None)
    parser.add_argument("--sentiment", choices=["positive", "negative", "neutral", "mixed"], default=None)
    parser.add_argument("--date-from", dest="date_from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", dest="date_to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--speaker", default=None, help="Filter by speaker name")
    args = parser.parse_args()
    chat_loop(args)          # indexing is an admin job: python -m src.rag_index


if __name__ == "__main__":
    main()
