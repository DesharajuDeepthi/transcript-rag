"""Lightweight validation tests for the categorization module."""

import pytest
from src.categorize import categorize_by_rules


# ---------------------------------------------------------------------------
# Helpers (import shim)
# ---------------------------------------------------------------------------

def _rules(title, participants=None):
    return categorize_by_rules(title=title, participants=participants or [])


# ---------------------------------------------------------------------------
# Call type classification
# ---------------------------------------------------------------------------

class TestCallTypeRules:
    def test_support_case_keyword(self):
        r = _rules("Support Case #9279 - Summit Trust Billing Inquiry")
        assert r is not None
        assert r["call_type"] == "support"
        assert r["confidence"] >= 0.90

    def test_case_hash_keyword(self):
        r = _rules("Case #1234 - Password Reset Issue")
        assert r is not None
        assert r["call_type"] == "support"

    def test_external_slash_pattern(self):
        r = _rules("Aegis / Redwood Clinical - ISO 27001 Preparation")
        assert r is not None
        assert r["call_type"] == "external"

    def test_external_qbr(self):
        r = _rules("Q2 QBR - Cobalt Software")
        assert r is not None
        assert r["call_type"] == "external"

    def test_internal_standup(self):
        r = _rules("Weekly Engineering Standup")
        assert r is not None
        assert r["call_type"] == "internal"

    def test_internal_outage(self):
        r = _rules("Detect Outage - Remediation Plan Review")
        assert r is not None
        assert r["call_type"] == "internal"

    def test_internal_planning(self):
        r = _rules("Q3 Planning Session")
        assert r is not None
        assert r["call_type"] == "internal"

    def test_ambiguous_returns_none(self):
        r = _rules("Catch-up call")
        assert r is None  # falls through to LLM

    def test_external_domain_detection(self):
        r = _rules(
            "Team sync",
            participants=["alice@aegiscloud.com", "bob@customer.com"],
        )
        # Multiple domains → external
        assert r is not None
        assert r["call_type"] == "external"


# ---------------------------------------------------------------------------
# Sub-theme classification
# ---------------------------------------------------------------------------

class TestSubThemeRules:
    def test_incident_response(self):
        r = _rules("Detect Outage - Remediation Plan Review")
        assert r is not None
        assert r["sub_theme"] == "incident_response"

    def test_compliance_security(self):
        r = _rules("Aegis / Acme - ISO 27001 Preparation")
        assert r is not None
        assert r["sub_theme"] == "compliance_security"

    def test_customer_support_issue(self):
        r = _rules("Support Case #1234 - Login Issue")
        assert r is not None
        assert r["sub_theme"] == "customer_support_issue"

    def test_engineering_sync(self):
        r = _rules("Weekly Engineering Standup")
        assert r is not None
        assert r["sub_theme"] == "engineering_sync"

    def test_customer_renewal(self):
        r = _rules("Aegis / BigCo - Contract Renewal Discussion")
        assert r is not None
        assert r["sub_theme"] == "customer_renewal"


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_support_high_confidence(self):
        r = _rules("Support Case #9999 - Data Export Issue")
        assert r is not None
        assert r["confidence"] >= 0.90

    def test_method_is_rules(self):
        r = _rules("Weekly Engineering Standup")
        assert r is not None
        assert r["method"] == "rules"


# ---------------------------------------------------------------------------
# Account name extraction (imported via shim in categorize module)
# ---------------------------------------------------------------------------

def test_extract_account_external():
    from src.churn_scorer import extract_account_name
    name = extract_account_name("Aegis / Redwood Clinical - ISO 27001 Preparation")
    assert name == "Redwood Clinical"


def test_extract_account_support():
    from src.churn_scorer import extract_account_name
    name = extract_account_name("Support Case #9279 - Summit Trust Billing Inquiry")
    assert name == "Summit Trust"


def test_extract_account_none():
    from src.churn_scorer import extract_account_name
    name = extract_account_name("Weekly Engineering Standup")
    assert name is None


# ---------------------------------------------------------------------------
# Account name normalisation — suffix stripping
# ---------------------------------------------------------------------------

class TestAccountNormalization:
    """Verify that product/legal suffixes are stripped so variant titles
    roll up to the same canonical account name."""

    def _name(self, title):
        from src.churn_scorer import extract_account_name
        return extract_account_name(title)

    def test_api_suffix_stripped(self):
        """'Vanta Health Systems API' should roll up to 'Vanta Health Systems'."""
        assert self._name("Aegis / Vanta Health Systems API - Q2 Planning") == "Vanta Health Systems"

    def test_platform_suffix_stripped(self):
        assert self._name("Aegis / Cobalt Platform - Renewal Discussion") == "Cobalt"

    def test_inc_suffix_stripped(self):
        assert self._name("Aegis / Summit Trust, Inc. - Onboarding") == "Summit Trust"

    def test_no_suffix_unchanged(self):
        """Names without suffixes must pass through intact."""
        assert self._name("Aegis / Redwood Clinical - Compliance Review") == "Redwood Clinical"

    def test_stacked_suffixes_stripped(self):
        """Multiple stacked suffixes (e.g. 'API Platform') should both be removed."""
        assert self._name("Aegis / Northstar Pharma API Platform - Incident") == "Northstar Pharma"

    def test_rollup_consistency(self):
        """Two title variants for same account must produce identical names."""
        from src.churn_scorer import extract_account_name
        base    = extract_account_name("Aegis / Vanta Health Systems - Contract Renewal")
        variant = extract_account_name("Aegis / Vanta Health Systems API - Integration Review")
        assert base == variant == "Vanta Health Systems"
