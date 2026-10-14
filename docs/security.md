# Authentication, tenancy and access control

## Who is asking

| Mode | How | Use |
|---|---|---|
| `AUTH_MODE=token` (default) | Signed JWT via `--token` / `RAG_TOKEN`. Signature, expiry, issuer, audience and required claims (`sub, role, tenant_id, iat, exp, iss, aud`) are verified, and the role must exist. | Anything real |
| `AUTH_MODE=dev` | `--email` mints a token on the spot with no credential check. This simulates authorization; it is not authentication. Running with no login at all is also allowed (tenant filter only, no RBAC). | Local demos only |
| Eval harness | Golden items with `as_user` simulate that employee. The token is signed with a key that exists only inside the running process, so it can't be replayed elsewhere. | `src/eval_run.py`, tests |

- **Tenant and role come only from the verified token.** A `--tenant` that disagrees with the
  token is rejected, not silently ignored. The eval harness stops if a simulated user's tenant
  differs from the tenant being evaluated.
- **Secret handling.** Outside dev mode, `JWT_SECRET` must be a random value of at least 32
  bytes; the placeholder is refused. In production it comes from a secrets manager.
- `python -m src.auth issue <email>` stands in for an identity provider such as Okta, Google
  Workspace or Azure AD. `src/auth.py:EMPLOYEE_REGISTRY` stands in for its user directory.

## What they can see

Every read, including neighbour-turn fetches, is filtered by `tenant_id` and by the role's
allowed `access_level`s (`src/roles.py`). Users can narrow results with filters but can never
remove these two conditions. Account timelines (the SQLite manifest) are filtered the same way,
so "the previous call" never resolves to a meeting the user isn't allowed to see. Every response
is checked inline for cross-tenant or out-of-role chunks; any hit pages someone and fails the
release gate.

## Indexing

`--reset` is tenant-scoped. Index writes are two-phase, and an embedding-model change needs a
new collection. See `docs/temporal.md` for both.
