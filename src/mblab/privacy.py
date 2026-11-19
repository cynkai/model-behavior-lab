"""Best-effort scan for personal data or secrets in recorded text.

This is a safety net before sharing a study, not a guarantee: it finds common
patterns (e-mail addresses, phone numbers, card numbers, API keys) and never
prints the matched value itself. Read your data before you publish it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("e-mail address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone number", re.compile(r"(?<![\d-])01[016789][ -]?\d{3,4}[ -]?\d{4}(?![\d-])")),
    ("phone number", re.compile(r"(?<!\w)\+\d{1,3}[ -]?\d{1,4}[ -]?\d{3,4}[ -]?\d{4}(?!\d)")),
    ("resident registration number", re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)")),
    ("API key or token", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("API key or token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}")),
    ("API key or token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("API key or token", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API key or token", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("API key or token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
]
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


@dataclass(frozen=True)
class PrivacyFinding:
    kind: str
    field: str


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def scan_text(text: str | None, field: str) -> list[PrivacyFinding]:
    if not text:
        return []
    findings: list[PrivacyFinding] = []
    kinds: set[str] = set()
    for kind, pattern in _PATTERNS:
        if kind not in kinds and pattern.search(text):
            kinds.add(kind)
            findings.append(PrivacyFinding(kind, field))
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            findings.append(PrivacyFinding("payment card number", field))
            break
    return findings
