"""
One-off, idempotent patch for docker-compose.yml (production-readiness review, 2026-10-13).

  python3 scripts/patch_compose_blockers.py            # edits docker-compose.yml in place (backup: .bak)

  1. Default tenant "aegis" (the tenant the employees in src/auth.py belong to).
  2. rag service: no --tenant flag (the token decides); AUTH_MODE defaults to dev for local demos.
  3. test service: mount data/ and monitoring/ so accounts.json and golden sets are available.
"""
import re
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "docker-compose.yml")
s = path.read_text()
orig = s

# 1. default tenant
s = s.replace("${TENANT_ID:-default}", "${TENANT_ID:-aegis}")


def block(name: str) -> tuple[int, int]:
    m = re.search(rf"^  {re.escape(name)}:\n", s, re.M)
    if not m:
        raise SystemExit(f"service '{name}' not found in {path}")
    nxt = re.search(r"^  (?:#|[A-Za-z0-9_-]+:)", s[m.end():], re.M)
    return m.start(), m.end() + (nxt.start() if nxt else len(s) - m.end())


# 2. rag service
a, b = block("rag")
rag = s[a:b]
rag = re.sub(r"(command:\s*)python -m src\.rag_chat --tenant \$\{TENANT_ID:-aegis\}", r"\1python -m src.rag_chat", rag)
rag = re.sub(r"command: >\n\s*sh -c \"python -m src\.rag_chat --tenant \$\{TENANT_ID:-aegis\}\n\s*\$\$\{EMAIL:\+--email \$\$EMAIL\}\"",
             "command: python -m src.rag_chat", rag)
if "AUTH_MODE=" not in rag:
    rag = re.sub(r"(\n(\s+)- TENANT_ID=[^\n]*)",
                 r"\1\n\2- AUTH_MODE=${AUTH_MODE:-dev}   # local demo: allows --email login; use token mode + JWT_SECRET for real use",
                 rag, count=1)
s = s[:a] + rag + s[b:]

# 3. test service
a, b = block("test")
test = s[a:b]
if "./data:/app/data" not in test:
    test = test.replace("      - ./data/dataset:/app/data/dataset:ro\n",
                        "      - ./data:/app/data:ro              # accounts.json, golden sets, dataset\n"
                        "      - ./monitoring:/app/monitoring:ro\n")
s = s[:a] + test + s[b:]

if s == orig:
    print("docker-compose.yml already patched — nothing to do")
else:
    shutil.copy(path, path.with_suffix(".yml.bak"))
    path.write_text(s)
    print(f"patched {path} (backup: {path.with_suffix('.yml.bak')})")
