"""
pii_scrub.py — Detect and redact PII from text before indexing.

What it redacts (regex-based, no external model required):
  - Email addresses          → [EMAIL]
  - US phone numbers         → [PHONE]
  - US Social Security Nos   → [SSN]
  - Credit card numbers      → [CREDIT_CARD]
  - IP addresses             → [IP_ADDRESS]

Optional (set PII_SCRUB_NAMES=true in .env + install spaCy en_core_web_sm):
  - PERSON names             → [PERSON]
  - ORG names                → [ORG]

Speaker names in transcript metadata are intentionally NOT redacted —
they are stored as metadata for filtering, not embedded in the text body.
If your policy requires full anonymisation, set ANONYMIZE_SPEAKERS=true
and speaker names will be replaced with Speaker_1, Speaker_2 etc. per meeting.

Usage (standalone test):
  python -m src.pii_scrub "Call John at john@acme.com or 555-123-4567"
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # SSN before phone so ###-##-#### matches before ###-####
    ("SSN",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card (Visa/MC/Amex/Discover — 13-16 digits with optional separators)
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,15}\d\b")),
    # Email
    ("EMAIL",       re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    # US phone: (555) 123-4567 | 555-123-4567 | +1 555 123 4567 | 5551234567
    ("PHONE",       re.compile(
        r"(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})\b"
    )),
    # IPv4
    ("IP_ADDRESS",  re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )),
]

# ── Optional spaCy NER ────────────────────────────────────────────────────────

_nlp = None  # lazy-loaded

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    if os.getenv("PII_SCRUB_NAMES", "").lower() not in ("1", "true", "yes"):
        return None
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy NER loaded for name redaction.")
    except (ImportError, OSError):
        logger.warning(
            "PII_SCRUB_NAMES=true but spaCy / en_core_web_sm not installed. "
            "Name redaction disabled. Run: pip install spacy && python -m spacy download en_core_web_sm"
        )
        _nlp = False  # don't retry
    return _nlp if _nlp else None


# ── Speaker anonymisation ─────────────────────────────────────────────────────

def build_speaker_map(speakers: list[str]) -> dict[str, str]:
    """
    Return {real_name: "Speaker_N"} mapping for a meeting.
    Call once per meeting and pass the result to scrub_text().
    Only active when ANONYMIZE_SPEAKERS=true in env.
    """
    if os.getenv("ANONYMIZE_SPEAKERS", "").lower() not in ("1", "true", "yes"):
        return {}
    mapping: dict[str, str] = {}
    counter = 1
    for name in speakers:
        if name and name not in mapping:
            mapping[name] = f"Speaker_{counter}"
            counter += 1
    return mapping


# ── Core scrub function ───────────────────────────────────────────────────────

def scrub_text(text: str, speaker_map: dict[str, str] | None = None) -> str:
    """
    Return a copy of text with PII replaced by placeholder tokens.

    Args:
        text:        Raw text (transcript turn or summary body).
        speaker_map: Optional {real_name: anonymised_name} from build_speaker_map().
                     Applied before other scrubbing so name regex doesn't double-replace.
    """
    if not text:
        return text

    # 1. Speaker anonymisation (exact-match, word-boundary aware)
    if speaker_map:
        for real, anon in speaker_map.items():
            text = re.sub(r"\b" + re.escape(real) + r"\b", anon, text)

    # 2. Regex PII
    for label, pattern in _PATTERNS:
        text = pattern.sub(f"[{label}]", text)

    # 3. Optional spaCy NER for remaining PERSON / ORG names
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        replacements: list[tuple[int, int, str]] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                replacements.append((ent.start_char, ent.end_char, "[PERSON]"))
            elif ent.label_ == "ORG":
                replacements.append((ent.start_char, ent.end_char, "[ORG]"))
        # Apply in reverse order so offsets stay valid
        for start, end, repl in sorted(replacements, reverse=True):
            text = text[:start] + repl + text[end:]

    return text


def scrub_summary(summary_text: str) -> str:
    """Convenience wrapper for summary chunks (no speaker map needed)."""
    return scrub_text(summary_text)


# ── Audit helper ──────────────────────────────────────────────────────────────

def detect_pii(text: str) -> list[dict]:
    """
    Return a list of detected PII spans (for audit logging, not redaction).
    Does NOT include spaCy results — regex only for speed.

    Example return:
      [{"type": "EMAIL", "value": "john@acme.com", "start": 14, "end": 27}]
    """
    found = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            found.append({
                "type":  label,
                "value": m.group(),
                "start": m.start(),
                "end":   m.end(),
            })
    return sorted(found, key=lambda x: x["start"])


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sample = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Email john.doe@acme.com or call (415) 555-1234. "
        "SSN: 123-45-6789. IP: 192.168.1.100. "
        "Card: 4111 1111 1111 1111."
    )
    print("Original :", sample)
    print("Scrubbed :", scrub_text(sample))
    print("Detected :", detect_pii(sample))
