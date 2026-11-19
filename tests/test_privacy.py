from __future__ import annotations

import pytest

from mblab.privacy import scan_text

# Fake credentials are assembled at runtime so the source never contains a
# literal that secret scanners would flag.
FAKE_OPENAI = "sk-" + "proj-" + "A" * 24
FAKE_GITHUB = "gh" + "p_" + "a1" * 18
FAKE_AWS = "AK" + "IA" + "B" * 16


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail me: a.b@example.co.kr", "e-mail address"),
        ("call 010-1234-5678", "phone number"),
        ("call +82 10 1234 5678", "phone number"),
        ("id 900101-1234567", "resident registration number"),
        (FAKE_OPENAI, "API key or token"),
        (FAKE_GITHUB, "API key or token"),
        (FAKE_AWS, "API key or token"),
        ("card 4111 1111 1111 1111", "payment card number"),
    ],
)
def test_detects_common_patterns(text, kind):
    assert kind in {f.kind for f in scan_text(text, "notes")}


@pytest.mark.parametrize(
    "text",
    [
        "ANSWER: 2027-02-27",
        "C(23,3) - 4*C(13,3) + 6 = 1771 - 1144 + 6 = 633\nANSWER: 633",
        "2027 -> 57 -> 74 -> 65 -> 61 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4 -> 16",
        "Thought for 1m 12s",
        "1234567890123456",  # fails the Luhn check
        "",
    ],
)
def test_no_false_positives_on_task_content(text):
    assert scan_text(text, "response_text") == []
