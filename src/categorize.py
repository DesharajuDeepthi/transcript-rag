from __future__ import annotations
"""
Categorization module for Transcript Intelligence.

Hybrid approach:
1. Rule-based first: Pattern matching on meeting titles (free, transparent)
2. LLM fallback: gpt-4o-mini for ambiguous cases (~$0.30 total)
3. Confidence scoring: Flag low-confidence cases for review

Usage:
    from src.categorize import categorize_meetings
    df = categorize_meetings(df)
"""

import os
import re
import json
from typing import Literal, TypedDict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


CallType = Literal["support", "external", "internal"]
SubTheme = Literal[
    "incident_response",
    "customer_renewal",
    "customer_onboarding",
    "customer_support_issue",
    "compliance_security",
    "product_planning",
    "engineering_sync",
    "cross_team_escalation",
    "other",
]


class CategorizationResult(TypedDict):
    call_type: CallType
    sub_theme: SubTheme
    confidence: float
    method: Literal["rules", "llm"]
    reasoning: str


# ============================================================================
# Compiled patterns
# ============================================================================

_SUPPORT_PATTERNS = [
    re.compile(r"support case", re.I),
    re.compile(r"ticket\s*#", re.I),
    re.compile(r"\bcase\s*#", re.I),
]

_EXTERNAL_PATTERNS = [
    re.compile(r"^[A-Za-z]\w+\s*/\s*[A-Za-z]", re.I),
    re.compile(r"\bqbr\b", re.I),
    re.compile(r"quarterly business review", re.I),
    re.compile(r"\brenewal\b", re.I),
]

_INTERNAL_PATTERNS = [
    re.compile(r"standup", re.I),
    re.compile(r"outage", re.I),
    re.compile(r"postmortem", re.I),
    re.compile(r"\bsync\b", re.I),
    re.compile(r"weekly\s+(eng|engineering|team)", re.I),
    re.compile(r"\bplanning\b", re.I),
    re.compile(r"retrospective|retro\b", re.I),
    re.compile(r"remediation", re.I),
    re.compile(r"\breview\b", re.I),
]

_SUB_THEME_PATTERNS: dict[str, list[re.Pattern]] = {
    "incident_response": [
        re.compile(r"outage|postmortem|incident|remediation", re.I),
    ],
    "customer_renewal": [
        re.compile(r"renewal|qbr|business review", re.I),
    ],
    "customer_onboarding": [
        re.compile(r"onboarding|kickoff|kick-off", re.I),
    ],
    "customer_support_issue": [
        re.compile(r"support case|ticket\s*#|case\s*#", re.I),
    ],
    "compliance_security": [
        re.compile(r"iso\s*27001|soc\s*2|compliance|audit|security review", re.I),
    ],
    "product_planning": [
        re.compile(r"q\d\s+planning|roadmap|feature", re.I),
    ],
    "engineering_sync": [
        re.compile(r"engineering\s+(standup|sync)|weekly\s+eng|standup", re.I),
    ],
    "cross_team_escalation": [
        re.compile(r"escalation|cross-team", re.I),
    ],
}


def categorize_by_rules(
    title: str,
    summary: str = "",
    participants: list[str] | None = None,
) -> CategorizationResult | None:
    """
    Categorize a meeting using rule-based pattern matching.

    Returns CategorizationResult if confident (>=0.85), else None (fall back to LLM).
    """
    title_str = title or ""
    participants = participants or []
    domains = {email.split("@")[-1] for email in participants if "@" in email}
    has_external_domain = len(domains) > 1

    if any(p.search(title_str) for p in _SUPPORT_PATTERNS):
        call_type: CallType = "support"
        confidence = 0.95
    elif any(p.search(title_str) for p in _EXTERNAL_PATTERNS) or has_external_domain:
        call_type = "external"
        confidence = 0.90 if has_external_domain else 0.85
    elif any(p.search(title_str) for p in _INTERNAL_PATTERNS):
        call_type = "internal"
        confidence = 0.90
    else:
        return None

    sub_theme: SubTheme = "other"
    for theme, patterns in _SUB_THEME_PATTERNS.items():
        if any(p.search(title_str) or p.search(summary or "") for p in patterns):
            sub_theme = theme  # type: ignore
            break

    return {
        "call_type": call_type,
        "sub_theme": sub_theme,
        "confidence": confidence,
        "method": "rules",
        "reasoning": f"Matched title pattern (domains: {len(domains)})",
    }


# ============================================================================
# LLM fallback
# ============================================================================

_LLM_SYSTEM_PROMPT = """You categorize B2B SaaS meeting transcripts.

Three CALL TYPES:
- support: Customer support cases (customer reporting an issue, vendor helping resolve)
- external: Account manager + customer calls (renewals, planning, QBRs, adoption, feedback)
- internal: Internal company calls (engineering syncs, outages, planning, escalations)

Nine SUB-THEMES:
- incident_response: Outages, postmortems, incident reviews
- customer_renewal: Renewal discussions, QBRs, account planning
- customer_onboarding: New customer setup, training
- customer_support_issue: Specific ticket/case discussions
- compliance_security: ISO, SOC2, audits, security reviews
- product_planning: Feature requests, roadmap, quarterly planning
- engineering_sync: Standups, technical syncs
- cross_team_escalation: Issues spanning multiple teams
- other: Doesn't fit cleanly

Respond ONLY with valid JSON:
{
  "call_type": "support|external|internal",
  "sub_theme": "<one of the 9>",
  "confidence": 0.0-1.0,
  "reasoning": "<one sentence>"
}"""


def categorize_by_llm(
    title: str,
    summary: str,
    topics: list[str],
    client,
) -> CategorizationResult:
    """Categorize a meeting using gpt-4o-mini. Used for ambiguous cases."""
    user_message = (
        f"TITLE: {title}\n\n"
        f"SUMMARY: {summary[:500]}\n\n"
        f"PRE-COMPUTED TOPICS: {', '.join(topics)}\n\n"
        f"Categorize this meeting."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        return {
            "call_type": parsed["call_type"],
            "sub_theme": parsed.get("sub_theme", "other"),
            "confidence": float(parsed.get("confidence", 0.7)),
            "method": "llm",
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "call_type": "internal",
            "sub_theme": "other",
            "confidence": 0.3,
            "method": "llm",
            "reasoning": f"LLM call failed: {e}",
        }


# ============================================================================
# Main entry point
# ============================================================================

def categorize_meetings(
    df: pd.DataFrame,
    use_llm: bool = True,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Categorize all meetings. Tries rules first; LLM fallback for ambiguous cases.

    Adds columns: call_type, sub_theme, category_confidence, category_method, category_reasoning.
    """
    df = df.copy()

    client = None
    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except Exception as e:
            print(f"⚠️  LLM client init failed: {e}. Falling back to rules-only.")
            use_llm = False

    results = []
    rule_count = 0
    llm_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing"):
        result = categorize_by_rules(
            title=row["title"],
            summary=row.get("summary", ""),
            participants=row.get("participants", []),
        )

        if result is None:
            if use_llm and client:
                result = categorize_by_llm(
                    title=row["title"],
                    summary=row.get("summary", ""),
                    topics=row.get("topics", []),
                    client=client,
                )
                llm_count += 1
            else:
                result = {
                    "call_type": "internal",
                    "sub_theme": "other",
                    "confidence": 0.3,
                    "method": "rules",
                    "reasoning": "No rule matched; LLM disabled",
                }
        else:
            rule_count += 1

        results.append(result)

    df["call_type"] = [r["call_type"] for r in results]
    df["sub_theme"] = [r["sub_theme"] for r in results]
    df["category_confidence"] = [r["confidence"] for r in results]
    df["category_method"] = [r["method"] for r in results]
    df["category_reasoning"] = [r["reasoning"] for r in results]

    total = len(df)
    print(f"\n✅ Categorization complete:")
    print(f"   Rule-based: {rule_count} ({100*rule_count/total:.0f}%)")
    print(f"   LLM-based:  {llm_count} ({100*llm_count/total:.0f}%)")
    print(f"\n   Call type distribution:")
    print(df["call_type"].value_counts().to_string())
    print(f"\n   Sub-theme distribution:")
    print(df["sub_theme"].value_counts().to_string())

    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)
    df_save = df.drop(columns=["transcript_text"], errors="ignore").copy()
    for col in ["participants", "action_items", "topics", "key_moments",
                "key_moment_types", "speakers"]:
        if col in df_save.columns:
            df_save[col] = df_save[col].apply(json.dumps)
    df_save.to_csv("outputs/categorized.csv", index=False)
    print("\n💾 Saved to outputs/categorized.csv")

    return df


def review_low_confidence(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Filter meetings with low categorization confidence for manual review."""
    low = df[df["category_confidence"] < threshold].sort_values("category_confidence")
    print(f"\n🔍 {len(low)} meetings below confidence {threshold}:")
    for _, row in low[["meeting_id", "title", "call_type", "sub_theme",
                        "category_confidence", "category_reasoning"]].iterrows():
        print(f"  [{row['category_confidence']:.2f}] {row['title'][:60]}")
        print(f"       → {row['call_type']} / {row['sub_theme']}")
        print(f"       Reason: {row['category_reasoning']}")
    return low


if __name__ == "__main__":
    from src.loader import load_meetings
    df = load_meetings()
    df = categorize_meetings(df)
    review_low_confidence(df)
