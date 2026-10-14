# Versioned golden sets

| Version | Questions | What changed |
|---|---|---|
| `golden-v1.json` | 15 | Original hand-written set (action items, who-said-what, date filters, abstention). |
| `golden-v2.json` | 19 | v1 + 4 RBAC cases: the same question asked by a role that may see the meeting (must answer) and one that may not (must abstain, zero leaked chunks). |
| `golden-v3.json` | 31 | v2 + 12 account/temporal cases: latest status, as-of (no future calls), exact period, previous call, what changed, account history, look-alike customers (Ironclad vs Ironworks), old commitments not presented as current, exact meeting id, vague account overview. Existing items labelled with expected account / mode where unambiguous. |

Rules:
- Never edit a published version — copy it to `golden-v(N+1).json` and change that.
- `eval_run.py` stamps the version on every results file; `eval_gate.py` compares against
  `baselines/<version>.json`, so baselines are per version.
- Confirmed failures from live traffic (online-eval low scores, thumbs-down) get added in the next version.
- Optional per-item fields: `as_user` (employee email → role), `category`, `filters`.
- Temporal labels (all optional, see `docs/temporal.md`): `expected_account_id`, `expected_temporal_mode`,
  `latest_meeting_id` (id or list of acceptable ids — must be the first source), `expected_meeting_ids`
  (all must be retrieved), `forbidden_before` / `forbidden_after` (YYYY-MM-DD), `forbidden_account_ids`,
  `expected_source_version`, `outdated_facts` (statements that must not be presented as current).
