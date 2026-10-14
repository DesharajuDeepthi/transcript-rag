"""
Governance layer for the Transcript Intelligence MCP server.

Provides:
  check_access(role, tool_name) -> (allowed: bool, reason: str)
  audit_log(...)                -> writes one JSON line to mcp_server/audit.log
  timer                         -> context manager; sets self.elapsed_ms on exit
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

# ---------------------------------------------------------------------------
# Role → allowed tools
# ---------------------------------------------------------------------------

_ROLE_PERMISSIONS: dict[str, set[str]] = {
    # Full access — data analysts, engineers
    "analyst": {
        "search_meetings",
        "get_sentiment_trends",
        "score_churn_risk",
        "find_recurring_topics",
        "get_action_items",
    },
    # Read-only subset — external viewers, limited stakeholders
    "viewer": {
        "get_sentiment_trends",
        "find_recurring_topics",
    },
}

_ALL_TOOLS: set[str] = _ROLE_PERMISSIONS["analyst"]   # superset


def check_access(role: str, tool_name: str) -> tuple[bool, str]:
    """
    Return (True, "ok") if *role* may call *tool_name*, else (False, reason).

    Unknown roles are denied by default.
    Unknown tool names are denied (fail-closed).
    """
    if tool_name not in _ALL_TOOLS:
        return False, f"Unknown tool '{tool_name}'"

    allowed = _ROLE_PERMISSIONS.get(role, set())
    if tool_name in allowed:
        return True, "ok"

    if role not in _ROLE_PERMISSIONS:
        return False, f"Unknown role '{role}'"

    return False, (
        f"Role '{role}' does not have access to '{tool_name}'. "
        f"Permitted tools: {sorted(allowed) or '(none)'}"
    )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

_LOG_PATH: Path = Path(__file__).parent / "audit.log"
_log_lock: threading.Lock = threading.Lock()


def audit_log(
    tool_name: str,
    args: dict[str, Any],
    result_summary: str,
    duration_ms: float,
    role: str,
    status: str,          # "ok" | "denied" | "error"
) -> None:
    """
    Append one JSON line to mcp_server/audit.log.

    *result_summary* must be a short string — do NOT pass the full result dict.
    This keeps the log readable and avoids leaking large payloads.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "role": role,
        "args": args,
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "result": result_summary,
    }
    line = json.dumps(entry, default=str)
    with _log_lock:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class timer:
    """
    Measure wall-clock time for a block.

    Usage::

        with timer() as t:
            do_work()
        print(t.elapsed_ms)   # float, milliseconds
    """

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000
