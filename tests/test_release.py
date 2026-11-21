"""The version is the same everywhere, and the current release is on top of the changelog."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import mblab

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == mblab.__version__
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert f'name = "model-behavior-lab"\nversion = "{mblab.__version__}"\n' in lock


def test_changelog_has_the_current_release_on_top():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    releases = [h for h in headings if h != "Unreleased"]
    assert headings.count("Unreleased") <= 1 and headings[0] in {"Unreleased", releases[0]}
    assert re.fullmatch(rf"{re.escape(mblab.__version__)} — \d{{4}}-\d{{2}}-\d{{2}}", releases[0])
    assert len(set(releases)) == len(releases)


def test_readmes_name_the_current_release_tag():
    for readme in ("README.md", "docs/README.ko.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        assert f"github.com/cynkai/model-behavior-lab@v{mblab.__version__}" in text, readme
