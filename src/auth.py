"""
auth.py — Authentication and token management for the RAG pipeline.

Flow (AUTH_MODE=token, the default):
  1. The identity provider (here: `python -m src.auth issue <email>`, standing in for
     Okta / Google Workspace / Azure AD) issues a signed JWT.
  2. The client presents it (`rag_chat --token …` or RAG_TOKEN).
  3. verify_token() checks signature, expiry, issuer, audience and required claims.
     tenant_id AND role come ONLY from the verified token — never from CLI flags or
     request parameters — so a user cannot select another tenant.

AUTH_MODE=dev enables "email login": an email is looked up and a token minted on the
spot. That is authorization SIMULATION, not authentication (anyone can type an email),
so it is refused unless AUTH_MODE=dev is set explicitly. The offline eval harness uses
it deliberately (allow=True) to run golden questions as specific employees.

Outside dev mode JWT_SECRET must be a random value of at least 32 bytes; the built-in
placeholder is refused. In production it comes from a secrets manager.

Employee roles reflect the dataset categories:
  executive         — All Hands, competitive, win/loss, all customer & incident calls
  sales             — Customer calls + competitive landscape
  customer_success  — Customer calls, support cases, incident calls
  support_engineer  — Support cases, engineering meetings, incidents
  engineering       — Eng standups/sprints, incidents, internal compliance
  compliance        — Internal SOC 2 / Comply v2 meetings + customer compliance calls
"""

from __future__ import annotations

import os
import secrets
import time
from typing import TypedDict

import jwt  # PyJWT — add to requirements.txt: PyJWT>=2.8.0
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
# Read at call time (not import time) so deployments and tests can change them.

DEV_PLACEHOLDER_SECRET = "change-me-in-production"
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "transcript-intelligence"
JWT_AUDIENCE = "transcript-rag"
TOKEN_TTL_SEC = 8 * 3600  # 8 hours
REQUIRED_CLAIMS = ["sub", "role", "tenant_id", "iat", "exp", "iss", "aud"]


# Simulated logins outside AUTH_MODE=dev (eval harness) are signed with a key that exists only
# inside this process, so such a token can never be replayed against a real deployment.
_EPHEMERAL_SECRET = secrets.token_urlsafe(48)


def auth_mode() -> str:
    return os.getenv("AUTH_MODE", "token").strip().lower()


def _secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if auth_mode() == "dev":
        return secret or DEV_PLACEHOLDER_SECRET
    if not secret or secret == DEV_PLACEHOLDER_SECRET or len(secret.encode()) < 32:
        raise RuntimeError("JWT_SECRET must be set to a random value of at least 32 bytes "
                           "(e.g. `python -c 'import secrets; print(secrets.token_urlsafe(48))'`). "
                           "The placeholder secret is only accepted with AUTH_MODE=dev.")
    return secret


# ── Employee registry ─────────────────────────────────────────────────────────
# Replace with a real database / identity provider lookup in production.
# Format: "email": {"role": <role>, "tenant_id": <tenant>, "name": <display name>}

EMPLOYEE_REGISTRY: dict[str, dict] = {
    # ── Executives ────────────────────────────────────────────────────────────
    "ceo@aegis.com":      {"role": "executive",        "tenant_id": "aegis", "name": "Alex Morgan"},
    "cto@aegis.com":      {"role": "executive",        "tenant_id": "aegis", "name": "Jordan Lee"},
    "cro@aegis.com":      {"role": "executive",        "tenant_id": "aegis", "name": "Sam Rivera"},

    # ── Sales ─────────────────────────────────────────────────────────────────
    "vanessa.cruz@aegis.com":   {"role": "sales", "tenant_id": "aegis", "name": "Vanessa Cruz"},
    "marcus.bell@aegis.com":    {"role": "sales", "tenant_id": "aegis", "name": "Marcus Bell"},
    "sofia.petrov@aegis.com":   {"role": "sales", "tenant_id": "aegis", "name": "Sofia Petrov"},

    # ── Customer Success ──────────────────────────────────────────────────────
    "catherine.page@aegis.com": {"role": "customer_success", "tenant_id": "aegis", "name": "Catherine Page"},
    "derek.walsh@aegis.com":    {"role": "customer_success", "tenant_id": "aegis", "name": "Derek Walsh"},

    # ── Support Engineers ─────────────────────────────────────────────────────
    "kendra.wallace@aegis.com": {"role": "support_engineer", "tenant_id": "aegis", "name": "Kendra Wallace"},
    "ravi.shankar@aegis.com":   {"role": "support_engineer", "tenant_id": "aegis", "name": "Ravi Shankar"},

    # ── Engineering ───────────────────────────────────────────────────────────
    "lena.kowalski@aegis.com":  {"role": "engineering", "tenant_id": "aegis", "name": "Lena Kowalski"},
    "tom.chen@aegis.com":       {"role": "engineering", "tenant_id": "aegis", "name": "Tom Chen"},

    # ── Compliance ────────────────────────────────────────────────────────────
    "priya.nair@aegis.com":     {"role": "compliance", "tenant_id": "aegis", "name": "Priya Nair"},
    "james.ford@aegis.com":     {"role": "compliance", "tenant_id": "aegis", "name": "James Ford"},
}


# ── Token types ───────────────────────────────────────────────────────────────

class TokenClaims(TypedDict):
    sub:       str   # employee email
    name:      str   # display name
    role:      str   # one of roles.ROLES keys
    tenant_id: str   # ChromaDB tenant isolation key
    iat:       int   # issued-at (Unix timestamp)
    exp:       int   # expires-at (Unix timestamp)
    iss:       str
    aud:       str


# ── Public API ────────────────────────────────────────────────────────────────

def issue_token(email: str, ttl_sec: int = TOKEN_TTL_SEC, *, _key: str | None = None) -> str:
    """
    Identity-provider side: look up the employee and sign a JWT.
    Raises PermissionError if the email is not in the registry.
    """
    employee = EMPLOYEE_REGISTRY.get(email.lower().strip())
    if not employee:
        raise PermissionError(
            f"No account found for '{email}'. "
            "Contact your administrator to be added to the system."
        )
    now = int(time.time())
    payload: TokenClaims = {
        "sub":       email.lower().strip(),
        "name":      employee["name"],
        "role":      employee["role"],
        "tenant_id": employee["tenant_id"],
        "iat":       now,
        "exp":       now + ttl_sec,
        "iss":       JWT_ISSUER,
        "aud":       JWT_AUDIENCE,
    }
    return jwt.encode(payload, _key or _secret(), algorithm=JWT_ALGORITHM)


def verify_token(token: str, *, _key: str | None = None) -> TokenClaims:
    """
    Verify signature, expiry, issuer, audience and required claims; check the role exists.
    Raises PermissionError on any failure (expired, invalid, tampered, incomplete).
    """
    from src.roles import ROLES
    try:
        claims = jwt.decode(token, _key or _secret(), algorithms=[JWT_ALGORITHM], audience=JWT_AUDIENCE,
                            issuer=JWT_ISSUER, options={"require": REQUIRED_CLAIMS})
    except jwt.ExpiredSignatureError:
        raise PermissionError("Session expired — please log in again.")
    except jwt.InvalidTokenError as exc:
        raise PermissionError(f"Invalid or tampered token: {exc}")
    if claims["role"] not in ROLES:
        raise PermissionError(f"Token carries unknown role {claims['role']!r}.")
    if not isinstance(claims["tenant_id"], str) or not claims["tenant_id"].strip():
        raise PermissionError("Token carries no tenant.")
    return claims  # type: ignore[return-value]


def dev_login(email: str, *, allow: bool = False) -> TokenClaims:
    """
    Email → verified claims WITHOUT any credential check. Development / eval simulation only.
    Refused unless AUTH_MODE=dev or the caller passes allow=True (eval harness, tests).
    With allow=True outside dev mode the token is signed and verified with a process-local
    ephemeral key, so it is never a usable credential.
    """
    if auth_mode() == "dev":
        return verify_token(issue_token(email))
    if allow:
        return verify_token(issue_token(email, _key=_EPHEMERAL_SECRET), _key=_EPHEMERAL_SECRET)
    raise PermissionError("Email login is a development simulation and is disabled. "
                          "Present a signed token (--token / RAG_TOKEN), or set AUTH_MODE=dev locally.")


def whoami(token: str) -> dict:
    """Return a safe, human-readable summary of the token's identity."""
    c = verify_token(token)
    return {"email": c["sub"], "name": c.get("name"), "role": c["role"], "tenant_id": c["tenant_id"]}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Issue / inspect tokens (stand-in for an identity provider)")
    ap.add_argument("cmd", choices=["issue", "whoami"])
    ap.add_argument("value", help="email (issue) or token (whoami)")
    a = ap.parse_args()
    print(issue_token(a.value) if a.cmd == "issue" else json.dumps(whoami(a.value), indent=2))
