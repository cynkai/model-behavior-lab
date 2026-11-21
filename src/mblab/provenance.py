"""Which code is running: the package version and, when it can be found, the git commit.

Between releases every commit carries the same version number, so the version alone
does not identify the code that made a report. The commit does. It is read from one of
two places and never guessed:

* a git checkout of this repository that the package runs from (``pip install -e .``,
  ``uv sync``): the checked-out commit, and whether ``src/mblab`` has changes that are
  not committed;
* the note pip keeps of a ``pip install "git+https://…@<ref>"`` install (PEP 610
  ``direct_url.json``), which names the commit that was installed.

A copy installed any other way (a wheel, a local directory without git) reports the
commit as unknown.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from mblab import REPOSITORY, __version__

DISTRIBUTION = "model-behavior-lab"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CodeVersion:
    version: str
    commit: str | None = None
    #: True when ``src/mblab`` differs from the commit; None when that was not checked.
    modified: bool | None = None

    def label(self) -> str:
        """``0.2.0 (commit 1a2b3c4d5e6f)``: the version and how far the commit is known."""
        if self.commit is None:
            return f"{self.version} (commit unknown)"
        if self.modified:
            return f"{self.version} (commit {self.commit[:12]} + uncommitted changes)"
        return f"{self.version} (commit {self.commit[:12]})"

    def install_ref(self) -> str:
        """The git ref that installs this code: the commit, else the version's release tag."""
        return self.commit or f"v{self.version}"

    def install_command(self) -> str:
        return f'pip install "git+{REPOSITORY}@{self.install_ref()}"'


def _git(directory: Path, *args: str) -> str | None:
    # GIT_DIR and friends would point git at another repository than this file's.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        done = subprocess.run(
            ["git", "-C", str(directory), "-c", "core.fsmonitor=false", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def from_checkout(package_dir: Path = _PACKAGE_DIR) -> CodeVersion | None:
    """The commit of the git checkout this package runs from, if it runs from one."""
    top = _git(package_dir, "rev-parse", "--show-toplevel")
    if not top:
        return None
    root = Path(top)
    # Only a checkout of this repository counts. A virtual environment inside some other
    # repository would otherwise report that repository's commit.
    if not _same_file(root / "src" / "mblab" / "__init__.py", package_dir / "__init__.py"):
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None or not _SHA.match(commit):
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src/mblab")
    return CodeVersion(__version__, commit, None if status is None else bool(status))


def from_pip_record(package_dir: Path = _PACKAGE_DIR) -> CodeVersion | None:
    """The commit pip noted when it installed this package from a git URL."""
    try:
        dist = metadata.distribution(DISTRIBUTION)
        text = dist.read_text("direct_url.json")
        installed_here = _same_file(
            Path(str(dist.locate_file("mblab/__init__.py"))), package_dir / "__init__.py"
        )
    except Exception:  # not installed, or unreadable metadata: the commit stays unknown
        return None
    if not text or not installed_here or dist.version != __version__:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    vcs = data.get("vcs_info") if isinstance(data, dict) else None
    if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
        return None
    commit = vcs.get("commit_id")
    if not isinstance(commit, str) or not _SHA.match(commit):
        return None
    return CodeVersion(__version__, commit, None)


@functools.cache
def code_version() -> CodeVersion:
    """The running code, as exactly as it can be identified (looked up once per process)."""
    return from_checkout() or from_pip_record() or CodeVersion(__version__)
