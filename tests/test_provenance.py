"""Which code made a report: the version and the git commit, never guessed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import mblab
from mblab import provenance
from mblab.provenance import CodeVersion, from_checkout, from_pip_record

ROOT = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _checkout(tmp_path: Path, package: str = "src/mblab") -> tuple[Path, Path]:
    root = tmp_path / "repo"
    package_dir = root / package
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("hi\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first")
    return root, package_dir


def test_labels_say_how_far_the_code_is_known():
    assert CodeVersion("0.2.0", SHA, False).label() == "0.2.0 (commit 0123456789ab)"
    assert CodeVersion("0.2.0", SHA, None).label() == "0.2.0 (commit 0123456789ab)"
    assert (
        CodeVersion("0.2.0", SHA, True).label()
        == "0.2.0 (commit 0123456789ab + uncommitted changes)"
    )
    assert CodeVersion("0.2.0").label() == "0.2.0 (commit unknown)"


def test_install_command_pins_the_commit_or_the_release_tag():
    repo = mblab.REPOSITORY
    assert CodeVersion("0.2.0", SHA).install_command() == f'pip install "git+{repo}@{SHA}"'
    assert CodeVersion("0.2.0").install_command() == f'pip install "git+{repo}@v0.2.0"'


@needs_git
def test_checkout_commit_and_changes_are_read(tmp_path):
    root, package_dir = _checkout(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    assert from_checkout(package_dir) == CodeVersion(mblab.__version__, head, False)
    (root / "README.md").write_text("changed\n", encoding="utf-8")  # outside src/mblab
    assert from_checkout(package_dir).modified is False
    (package_dir / "__init__.py").write_text("x = 2\n", encoding="utf-8")
    assert from_checkout(package_dir) == CodeVersion(mblab.__version__, head, True)
    _git(root, "checkout", "--", "src/mblab/__init__.py")
    (package_dir / "new_module.py").write_text("y = 1\n", encoding="utf-8")  # untracked
    assert from_checkout(package_dir).modified is True


@needs_git
def test_another_repository_is_never_reported(tmp_path, monkeypatch):
    # A virtual environment inside some project's repository: the package is not at
    # src/mblab of that repository, so its commit says nothing about this code.
    _, package_dir = _checkout(tmp_path, package=".venv/lib/site-packages/mblab")
    assert from_checkout(package_dir) is None
    assert from_checkout(tmp_path / "not-a-repo") is None


@needs_git
def test_git_environment_variables_do_not_redirect_the_lookup(tmp_path, monkeypatch):
    root, package_dir = _checkout(tmp_path)
    other, _ = _checkout(tmp_path / "other")
    (other / "README.md").write_text("another project\n", encoding="utf-8")
    _git(other, "commit", "-q", "-am", "second")
    head = _git(root, "rev-parse", "HEAD")
    assert head != _git(other, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert from_checkout(package_dir).commit == head


@needs_git
def test_this_checkout_is_identified():
    try:
        head = _git(ROOT, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("the tests do not run from a git checkout")
    found = from_checkout()
    if found is None:
        pytest.skip("mblab is not imported from this checkout")
    assert found.commit == head and found.version == mblab.__version__


class FakeDist:
    def __init__(self, direct_url: str | None, version: str, files_at: Path):
        self.direct_url = direct_url
        self.version = version
        self.files_at = files_at

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self.direct_url

    def locate_file(self, path: str) -> Path:
        return self.files_at / path


def _vcs(commit: str = SHA, vcs: str = "git") -> str:
    return json.dumps(
        {
            "url": mblab.REPOSITORY,
            "vcs_info": {"vcs": vcs, "commit_id": commit, "requested_revision": "v0.2.0"},
        }
    )


@pytest.mark.parametrize(
    ("direct_url", "version", "here", "expected"),
    [
        (_vcs(), None, True, SHA),
        (json.dumps({"url": "file:///src", "dir_info": {"editable": True}}), None, True, None),
        (json.dumps({"url": "file:///src", "dir_info": {}}), None, True, None),
        (_vcs(vcs="hg"), None, True, None),
        (_vcs(commit="v0.2.0"), None, True, None),  # a ref, not a commit
        ("{not json", None, True, None),
        (None, None, True, None),
        (_vcs(), "9.9.9", True, None),  # metadata of another installed version
        (_vcs(), None, False, None),  # the record belongs to another copy of the package
    ],
)
def test_pip_record_names_the_installed_commit_only(
    monkeypatch, tmp_path, direct_url, version, here, expected
):
    package_dir = tmp_path / "site-packages" / "mblab"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    files_at = package_dir.parent if here else tmp_path / "elsewhere"
    dist = FakeDist(direct_url, version or mblab.__version__, files_at)
    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: dist)
    found = from_pip_record(package_dir)
    assert (found.commit if found else None) == expected


def test_missing_distribution_means_unknown(monkeypatch):
    def missing(name):
        raise provenance.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "distribution", missing)
    assert from_pip_record() is None
