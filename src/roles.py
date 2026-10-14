"""
roles.py — Role definitions and access-level policies for the RAG pipeline.

Roles are derived from the actual dataset structure (100 meetings across 6 categories):

  customer_calls   (38)  — "Aegis / <Customer> - <Topic>" meetings
  support_cases    (27)  — "Support Case #XXXX - <Customer> <Issue>" meetings
  incidents        (12)  — INCIDENT / URGENT / ESCALATION / Outage meetings
  engineering      (10)  — Standups, sprint planning, retros, roadmap syncs
  compliance_audit  (7)  — SOC 2, Comply v2 internal launch/design/GA meetings
  executive         (6)  — All Hands, Win/Loss, Competitive analysis/strategy

Access levels assigned to each chunk at index time:
  "public"          — All Hands, general announcements (everyone can read)
  "customer"        — Customer-facing calls (sales, CS, support, executives)
  "support"         — Support cases (support team, CS, executives)
  "engineering"     — Internal eng meetings (engineering, executives)
  "incident"        — Outage/escalation war rooms (eng, executives, CS)
  "compliance"      — Internal compliance/audit meetings (compliance, executives)
  "executive"       — Strategic/competitive/financial (executives only)

Role → access levels they may query:
  executive         — everything
  sales             — customer + executive (competitive landscape for selling)
  customer_success  — customer + support + incident (full account context)
  support_engineer  — support + engineering + incident (technical triage)
  engineering       — engineering + incident + compliance (build + on-call)
  compliance        — compliance + customer (audit prep + customer compliance calls)
"""

from __future__ import annotations

# ── Access-level taxonomy ─────────────────────────────────────────────────────

ALL_ACCESS_LEVELS = [
    "public",
    "customer",
    "support",
    "engineering",
    "incident",
    "compliance",
    "executive",
]

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES: dict[str, dict] = {
    "executive": {
        "description": "Full visibility — all meeting categories",
        "access_levels": ALL_ACCESS_LEVELS,
    },
    "sales": {
        "description": "Customer calls and strategic/competitive context",
        "access_levels": ["public", "customer", "executive"],
    },
    "customer_success": {
        "description": "Full account context — customer calls, support cases, incidents",
        "access_levels": ["public", "customer", "support", "incident"],
    },
    "support_engineer": {
        "description": "Technical triage — support cases, engineering, incidents",
        "access_levels": ["public", "support", "engineering", "incident"],
    },
    "engineering": {
        "description": "Internal build and on-call — engineering, incidents, compliance",
        "access_levels": ["public", "engineering", "incident", "compliance"],
    },
    "compliance": {
        "description": "Audit and regulatory — compliance meetings + customer compliance calls",
        "access_levels": ["public", "compliance", "customer"],
    },
}


def get_allowed_levels(role: str) -> list[str]:
    """Return the list of access levels this role may query."""
    if role not in ROLES:
        raise PermissionError(
            f"Unknown role '{role}'. Valid roles: {list(ROLES.keys())}"
        )
    return ROLES[role]["access_levels"]


def build_role_filter(role: str) -> dict:
    """
    Return a ChromaDB where-clause that restricts results to chunks this role
    may see.  Always AND this into every retrieval query — never let callers
    skip it.

    Example output for 'sales':
        {"access_level": {"$in": ["public", "customer", "executive"]}}
    """
    levels = get_allowed_levels(role)
    return {"access_level": {"$in": levels}}


def list_roles() -> list[dict]:
    """Return all roles with their descriptions (safe to show to admins)."""
    return [
        {"role": name, "description": info["description"],
         "access_levels": info["access_levels"]}
        for name, info in ROLES.items()
    ]
