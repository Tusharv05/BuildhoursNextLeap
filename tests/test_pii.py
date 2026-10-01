"""L1 PII tests.

The security property under test is not "does the regex match" but "does the value leave
the process". NFR-4 forbids logging it, so the tests assert that the matched substring is
absent from every object the guard hands back.
"""

from __future__ import annotations

import pytest

from mf_rag.guardrails import PII_PATTERNS, detect_pii, guard_input


# Real-looking but entirely fictional values.
PAN = "ABCDE1234F"
AADHAAR = "234567890124"
PHONE = "9876543210"
EMAIL = "ravi.kumar@example.com"
ACCOUNT = "7788990011"
OTP = "448291"


@pytest.mark.parametrize(
    "text,expected",
    [
        (f"My PAN is {PAN}", "pan"),
        (f"PAN: {PAN} please help", "pan"),
        (f"aadhaar number {AADHAAR}", "aadhaar"),
        (f"Aadhaar is {AADHAAR}", "aadhaar"),
        (f"call me on {PHONE}", "phone"),
        (f"+91 {PHONE}", "phone"),
        (f"email me at {EMAIL}", "email"),
        (f"account number {ACCOUNT}", "account_number"),
        (f"Account no. {ACCOUNT}", "account_number"),
        (f"my OTP is {OTP}", "otp"),
        (f"OTP {OTP}", "otp"),
        (f"one time password {OTP}", "otp"),
        ("verification code: 552310", "otp"),
        ("IFSC SBIN0001234", "ifsc"),
    ],
)
def test_detect_pii_finds_each_pattern(text, expected):
    assert expected in detect_pii(text)


def test_detect_pii_reports_every_matching_pattern():
    text = f"My PAN is {PAN}, aadhaar {AADHAAR}, phone {PHONE}, email {EMAIL}"
    assert set(detect_pii(text)) >= {"pan", "aadhaar", "phone", "email"}


@pytest.mark.parametrize(
    "text",
    [
        "What is the exit load of HDFC Small Cap Fund?",
        "What is the ELSS lock-in period?",
        "Minimum SIP amount is Rs. 500",
        "The expense ratio is 1.03%",
        "Lock-in period is 3 years",
        "NAV as on 2026-09-27",
    ],
)
def test_factual_questions_are_not_flagged(text):
    """False positives here would block the whole demo, so numeric facts must pass."""
    assert detect_pii(text) == []


def test_lowercase_pan_is_not_a_pan():
    """A PAN is uppercase by definition; matching case-insensitively would hit normal words."""
    assert detect_pii("my pan abcde1234f") == []


def test_detect_pii_never_returns_the_value():
    """The return value is pattern names, so it is safe to log (NFR-4)."""
    hits = detect_pii(f"My PAN is {PAN} and my email is {EMAIL}")
    assert hits
    for secret in (PAN, EMAIL):
        assert secret not in " ".join(hits)
    assert PAN not in repr(hits)


def test_guard_input_blocks_pii():
    result = guard_input(f"My PAN is {PAN}")
    assert result.allowed is False
    assert result.matched_pattern_names == ["pan"]
    assert "pan" in result.reason


def test_guard_result_never_contains_the_matched_text():
    """NFR-4: the reason string must not become an accidental log of the PAN."""
    result = guard_input(f"My PAN is {PAN}, email {EMAIL}")
    assert PAN not in result.reason
    assert EMAIL not in result.reason
    assert PAN not in repr(result)


def test_guard_input_allows_clean_text():
    result = guard_input("What is the exit load of HDFC Small Cap Fund?")
    assert result.allowed is True
    assert result.reason is None
    assert result.matched_pattern_names == []


def test_guard_input_can_be_disabled():
    assert guard_input(f"My PAN is {PAN}", enabled=False).allowed is True


@pytest.mark.parametrize("text", ["", None, "   "])
def test_detect_pii_tolerates_empty(text):
    assert detect_pii(text or "") == []


def test_every_pattern_is_named_and_reachable():
    names = {name for name, _ in PII_PATTERNS}
    assert names == {
        "pan",
        "aadhaar",
        "phone",
        "email",
        "account_number",
        "otp",
        "ifsc",
        "demat",
    }


def test_pii_answer_never_echoes_the_value():
    from mf_rag.guardrails import pii_answer

    answer = pii_answer()
    for secret in (PAN, AADHAAR, PHONE, EMAIL, ACCOUNT, OTP):
        assert secret not in answer.text
    assert answer.is_refusal is True
    assert answer.trace.pii_blocked is True
