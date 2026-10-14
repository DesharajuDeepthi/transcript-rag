# Accounts, time and transcript versions

Semantic similarity alone can't answer "What is Bridgeport's **current** renewal status?". An older
call can be the closest match even though a newer one has changed the answer. So retrieval decides
**which account** and **which time window** a question is about *before* anything is searched, and
enforces that with filters rather than only asking the model to prefer newer sources.

## Data model

| Store | Role |
|---|---|
| `data/accounts.json` | Reviewed account registry. `account_id` is a stable slug keyed by the customer's **email domain**, not parsed from titles. |
| `data/chroma/manifest.sqlite` | **Source of truth** for timelines and versions: `accounts`, `meetings`, `meeting_versions`, `index_runs`. (PostgreSQL in production.) |
| Chroma | Searchable index only. Every chunk carries the metadata below. |

```json
{
  "tenant_id": "default", "account_id": "bridgeport-health", "meeting_id": "01KQ4D71…",
  "occurred_at": "2026-04-23T13:30:00+00:00", "occurred_ts": 1776951000, "date_int": 20260423,
  "conversation_sequence": 2, "source_version": 1, "is_current_revision": true, "supersedes_version": 0,
  "source_updated_at": "…", "content_hash": "sha256:…",
  "index_version": "2026-10-13.1", "chunking_version": "speaker-turn-v2"
}
```

Chunk ids include the revision (`<tenant>::<meeting>::v<N>::tx::<i>`), so revisions never collide
and neighbour-turn expansion can't mix them.

## Index updates (`python -m src.rag_index --tenant <t>`)

| Situation | Action |
|---|---|
| New meeting id | v1 written **inactive**, then switched on, then committed as current |
| Same content, same indexing fingerprint, all vectors present | Nothing |
| Same content, vectors missing or incomplete | `repaired`: the same revision is rewritten in place |
| Same content, **indexing settings changed** (chunking version, speaker anonymisation, PII rules, RBAC/call-type tagging rules, account or access level) | `reembedded`: the same revision is rewritten in place with a new fingerprint |
| Changed content hash | vN+1 written inactive, switched on (older revisions off, kept for audit), then committed |
| Meeting removed from the source | Vectors deactivated first, then the meeting is tombstoned in the manifest |
| Earlier call arrives late | `conversation_sequence` renumbered on the affected chunks |
| **Embedding model changed** | Refused. Vectors from two models can't share a collection, so build a new one (`RAG_COLLECTION=transcripts_v2`), evaluate it, then switch over. The pipeline also refuses to query a collection built with a different model. |

**Failure safety.** Each revision is recorded as `pending` *before* its vectors are written, and
`committed` only *after* they are written and switched on. If the run crashes anywhere in between:
- The previous revision keeps serving.
- A brand-new meeting stays invisible: timelines only list committed meetings.
- The next run reuses the pending version number and finishes the job.

Failed runs are recorded in `index_runs` with `status='failed'` and the error.

**Resets.**
- `--reset` deletes and rebuilds **one tenant**; other tenants' vectors and manifest rows are untouched.
- `--reset-all` deletes the whole collection and manifest, for every tenant.
- Vectors for a tenant that the manifest has no record of (a legacy index) are refused until that tenant is reset.

Each run gets an `index_version` of `YYYY-MM-DD.N`. N is never reused, even after resets.

## How a question is understood (`src/query_understanding.py`)

These are deterministic rules, with no LLM call. Each decision is recorded in `plan.reasons`
(shown by `/plan` in the chatbot) and in the trace.

| Question | Mode | Retrieval behaviour |
|---|---|---|
| "What is Bridgeport's current renewal status?" | `latest` | Account filter and current revisions only. Pull a semantic pool, group it by meeting, and take the **newest meeting within 15 relevance points of the best** as authoritative, with sources ordered newest first. |
| "What did Bridgeport say in February?" | `historical` | Account plus that date range, with **no recency boost**. |
| "What did we know as of March 1?" | `as_of` | `date_int <= 20260301` is enforced, then the newest relevant meeting is used within that window. |
| "What changed since the previous call?" | `comparison` | The newest two eligible calls from the manifest timeline, oldest first. |
| "…the call before their most recent one?" | `previous` | The second-newest eligible call. |
| "Summarize the account history" | `all_history` | Every eligible call (up to 8), summary plus best chunk each, chronological. |
| "What happened in meeting 01KQ…?" | `meeting` | Exact `meeting_id`. |
| "Use the original transcript version" | `revision` | `source_version = 1`, superseded revisions allowed. |
| "What is happening with Bridgeport?" | `account_overview` | The newest 3 calls. |
| "Why did Coastal pick Protect?" | `account` | Semantic search inside the account plus internal meetings about it. |

Guards:
- "the March outage" names an event, not a meeting date.
- "Q2 roadmap" and "fiscal year 2027" are topics; years and quarters only filter by date when the question is about conversations.
- Two accounts named in one question means no account filter.
- Timelines are **RBAC-filtered**: for a sales user, "the previous call" skips support cases they can't see, rather than leaking or counting them.

The prompt (`answer-v3`) explains the plan to the model: the newest source is authoritative for
"current" questions, older statuses must not be presented as current, and history goes in
chronological order.

## Metrics

| Metric | Offline (golden-v3 labels) | Live (every request / sampled) |
|---|---|---|
| Account-resolution accuracy | `expected_account_id` | ambiguous-account rate |
| Temporal-intent accuracy | `expected_temporal_mode` | mode distribution (`temporal_mode` segment) |
| Latest-meeting selection | `latest_meeting_id` is the first source | `latest_is_newest` rate |
| As-of leakage | `forbidden_before/after` | `as_of_leakage` check (warn) |
| Account mixing | `forbidden_account_ids` | `account_mixing` check (warn) |
| Correct revision | `expected_source_version` / all current | `stale_revision` check (warn) |
| Stale answer | LLM check against `outdated_facts` | online judge `presents_outdated_as_current` |

The release gate fails on any as-of leakage, account mixing or stale answer, and on intent or
account accuracy below threshold (`monitoring/alerts.yml`).

## Limits

- The dataset contains no corrected or deleted transcripts. Those paths are covered by
  `tests/test_temporal.py`, which runs the real indexer on a synthetic dataset.
- Internal meetings that discuss a customer (for example an outage impact review) have no
  `account_id`, so they are not on that customer's timeline. Plain `account` mode still searches them.
- Account and date parsing is rule-based. An LLM fallback for unresolved or ambiguous questions is
  a natural next step, and the live ambiguous-rate metric shows when it's needed.
