"""
accounts.py — stable customer account identifiers.

An account_id is NOT a name parsed out of a meeting title. Titles are
inconsistent ("Crestline Wealth Group Policy Sync Delay" is Crestline Wealth +
"Group Policy", not a company called "Crestline Wealth Group"). Instead each
account is keyed by its customer email domain, which every customer meeting
carries in meeting-info.json → allEmails, and given a stable slug id.

data/accounts.json is the reviewed registry (commit it). Build / refresh it:

    python -m src.accounts build          # derive from data/dataset, keep existing ids
    python -m src.accounts show

At index time rag_index.py maps each meeting's external domain → account_id
and copies the registry (with aliases) into the SQLite manifest, which is what
query-time account resolution reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

INTERNAL_DOMAINS = {d.strip().lower() for d in os.getenv("INTERNAL_EMAIL_DOMAINS", "aegiscloud.com,aegis.com").split(",")}
REGISTRY_PATH = Path(os.getenv("ACCOUNTS_PATH", "data/accounts.json"))

# Trailing words that may be dropped to form a shorter alias ("Coastal Living Co" → "Coastal Living")
_GENERIC_SUFFIX = {"co", "inc", "llc", "ltd", "corp", "group", "systems", "industries", "investments",
                   "software", "platform", "brands", "manufacturing", "logistics", "marketplace",
                   "commerce", "capital", "financial", "banking", "insurance", "goods", "labs", "data",
                   "cloud", "medical", "clinical", "health", "pharma", "wealth", "trust", "precision", "ai"}
# First words too generic to be an alias on their own
_COMMON_FIRST = {"global", "first", "united", "american", "national", "the", "new", "data", "cloud", "health"}

_TITLE_NAME_PATTERNS = [
    re.compile(r"^aegis\s*/\s*(?P<name>.+?)\s+-\s+", re.I),
    re.compile(r"^(?:urgent|escalation)\s*:\s*(?P<name>.+?)\s+-\s+", re.I),
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("’", "'").replace("'s", ""))).strip()


@dataclass
class Account:
    account_id: str
    name: str
    domains: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


def external_domains(meeting_info: dict) -> list[str]:
    emails = meeting_info.get("allEmails") or meeting_info.get("invitees") or []
    return sorted({e.split("@", 1)[1].lower() for e in emails
                   if "@" in e and e.split("@", 1)[1].lower() not in INTERNAL_DOMAINS})


class AccountRegistry:
    def __init__(self, accounts: list[Account] | None = None):
        self.accounts: dict[str, Account] = {a.account_id: a for a in (accounts or [])}
        self._by_domain = {d: a.account_id for a in self.accounts.values() for d in a.domains}

    # ── persistence ───────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path = REGISTRY_PATH) -> "AccountRegistry":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls([Account(**a) for a in json.loads(p.read_text())["accounts"]])

    def save(self, path: str | Path = REGISTRY_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted((asdict(a) for a in self.accounts.values()), key=lambda a: a["account_id"])
        p.write_text(json.dumps({"version": 1, "accounts": rows}, indent=2) + "\n")

    # ── lookup ────────────────────────────────────────────────────────────────
    def for_meeting(self, meeting_info: dict) -> Account | None:
        """Account of a meeting via its external email domain. None = internal meeting."""
        ids = {self._by_domain[d] for d in external_domains(meeting_info) if d in self._by_domain}
        return self.accounts[ids.pop()] if len(ids) == 1 else None

    def add_domain(self, domain: str, name_hint: str | None = None) -> Account:
        """Auto-register an unseen domain (logged by the indexer; review data/accounts.json)."""
        name = name_hint or domain.split(".")[0].replace("-", " ").title()
        aid = slugify(name)
        while aid in self.accounts and domain not in self.accounts[aid].domains:
            aid += "-2"
        acct = self.accounts.setdefault(aid, Account(aid, name, [], []))
        acct.domains = sorted(set(acct.domains) | {domain})
        acct.aliases = sorted(set(acct.aliases) | set(_aliases_for(name, domain)))
        self._by_domain[domain] = aid
        return acct


def _aliases_for(name: str, domain: str) -> list[str]:
    words = name.split()
    out = {normalize(name)}
    trimmed = list(words)
    while len(trimmed) > 1 and trimmed[-1].lower() in _GENERIC_SUFFIX:
        trimmed.pop()
        out.add(normalize(" ".join(trimmed)))
    if len(words[0]) >= 4 and words[0].lower() not in _COMMON_FIRST:
        out.add(normalize(words[0]))
    out.add(normalize(domain.split(".")[0]))
    return sorted(a for a in out if a)


def _name_from_title(title: str) -> str | None:
    for pat in _TITLE_NAME_PATTERNS:
        m = pat.match(title.strip())
        if m:
            return m.group("name").strip()
    return None


def build_registry(dataset_dir: str | Path, existing: AccountRegistry | None = None) -> AccountRegistry:
    """Derive accounts from the dataset. Existing account_ids are kept stable."""
    names_by_domain: dict[str, Counter] = defaultdict(Counter)
    for d in sorted(Path(dataset_dir).iterdir()):
        p = d / "meeting-info.json"
        if not p.exists():
            continue
        info = json.loads(p.read_text())
        doms = external_domains(info)
        if len(doms) != 1:
            continue
        nm = _name_from_title(info.get("title", ""))
        names_by_domain[doms[0]][nm or ""] += 1

    reg = existing or AccountRegistry()
    for domain, names in sorted(names_by_domain.items()):
        if domain in reg._by_domain:
            continue
        best = next((n for n, _ in names.most_common() if n), None)
        reg.add_domain(domain, best)

    # A short alias shared by two accounts is ambiguous: drop it from both.
    counts = Counter(a for acct in reg.accounts.values() for a in acct.aliases)
    for acct in reg.accounts.values():
        acct.aliases = sorted(a for a in acct.aliases if counts[a] == 1 or a == normalize(acct.name))
    return reg


def main():
    p = argparse.ArgumentParser(description="Build / inspect the account registry")
    p.add_argument("cmd", choices=["build", "show"])
    p.add_argument("--dataset", default=os.getenv("DATASET_DIR", "data/dataset"))
    p.add_argument("--path", default=str(REGISTRY_PATH))
    a = p.parse_args()
    if a.cmd == "build":
        reg = build_registry(a.dataset, AccountRegistry.load(a.path))
        reg.save(a.path)
        print(f"✅ {len(reg.accounts)} accounts → {a.path}")
    reg = AccountRegistry.load(a.path)
    for acct in sorted(reg.accounts.values(), key=lambda x: x.account_id):
        print(f"  {acct.account_id:<26} {acct.name:<26} {','.join(acct.domains):<26} {acct.aliases}")


if __name__ == "__main__":
    main()
