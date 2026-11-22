"""Report destinations remain distinct under the host filesystem's path rules."""

import errno
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab.benchmark import BENCHMARK_FILE, Benchmark, write_reports
from mblab.cli import app
from mblab.study import StudyError

from .test_benchmark import _copy


def _reports(root: Path, paths: list[str]) -> Benchmark:
    benchmark = Benchmark.load(root)
    original = benchmark.definition.reports[0]
    benchmark.definition.reports = [
        original.model_copy(update={"id": f"report-{index}", "path": path})
        for index, path in enumerate(paths)
    ]
    return benchmark


@pytest.mark.parametrize("alias", ["reports/./preliminary.md", "reports//preliminary.md"])
@pytest.mark.parametrize("check", [False, True])
def test_cli_rejects_equivalent_report_paths_before_writing(
    tmp_path: Path, alias: str, check: bool
) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, ["reports/preliminary.md", alias])
    (root / BENCHMARK_FILE).write_text(benchmark.definition.model_dump_json(), encoding="utf-8")
    archive = root / "reports/preliminary.md"
    archive.write_text("preserve this archive\n", encoding="utf-8")

    command = ["benchmark", "report", str(root)]
    result = CliRunner().invoke(app, [*command, "--check"] if check else command)

    assert result.exit_code == 2
    assert "duplicate report path" in result.output
    assert archive.read_text(encoding="utf-8") == "preserve this archive\n"


@pytest.mark.parametrize("directory", ["reports", "new/nested"])
def test_case_aliases_follow_the_destination_filesystem(tmp_path: Path, directory: str) -> None:
    root = _copy(tmp_path)
    probe = root / "CaseProbe"
    probe.touch()
    case_sensitive = not (root / "caseprobe").exists()
    benchmark = _reports(root, [f"{directory}/Result.md", f"{directory}/result.md"])

    if case_sensitive:
        paths = write_reports(benchmark)
        assert len(paths) == 2
        assert not paths[0].samefile(paths[1])
    else:
        with pytest.raises(StudyError, match="duplicate report path"):
            write_reports(benchmark)
        assert not (root / directory / "Result.md").exists()


def test_directory_symlink_alias_is_rejected_before_writing(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    (root / "alias").symlink_to(root / "reports", target_is_directory=True)
    benchmark = _reports(root, ["reports/preliminary.md", "alias/preliminary.md"])
    archive = root / "reports/preliminary.md"
    original = archive.read_bytes()

    with pytest.raises(StudyError, match="duplicate report path"):
        write_reports(benchmark)
    assert archive.read_bytes() == original


def test_selected_report_still_checks_other_destinations(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, ["reports/preliminary.md", "reports/./preliminary.md"])
    archive = root / "reports/preliminary.md"
    original = archive.read_bytes()

    with pytest.raises(StudyError, match="duplicate report path"):
        write_reports(benchmark, "report-1")
    assert archive.read_bytes() == original


def test_distinct_nested_paths_remain_writable(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, ["first/nested/result.md", "second/nested/result.md"])

    paths = write_reports(benchmark)

    assert all(path.is_file() for path in paths)
    assert not paths[0].samefile(paths[1])


@pytest.mark.skipif(Path("a\\b") == Path("a/b"), reason="requires POSIX filename rules")
def test_posix_backslash_filename_remains_distinct(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, ["new\\result.md", "new/result.md"])

    paths = write_reports(benchmark)

    assert all(path.is_file() for path in paths)
    assert not paths[0].samefile(paths[1])


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("link_kind", ["directory", "file"])
def test_cli_rejects_symlink_escape_without_writing(
    tmp_path: Path, check: bool, link_kind: str
) -> None:
    root = _copy(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    archive = outside / "external.md"
    archive.write_text("outside archive\n", encoding="utf-8")
    link = root / "escape.md"
    if link_kind == "directory":
        link.symlink_to(outside, target_is_directory=True)
        output = "escape.md/external.md"
    else:
        link.symlink_to(archive)
        output = "escape.md"
    benchmark = _reports(root, ["new.md", output])
    (root / BENCHMARK_FILE).write_text(benchmark.definition.model_dump_json(), encoding="utf-8")

    command = ["benchmark", "report", str(root)]
    result = CliRunner().invoke(app, [*command, "--check"] if check else command)

    assert result.exit_code == 2
    assert "inside the benchmark directory" in result.output
    assert archive.read_text(encoding="utf-8") == "outside archive\n"
    assert not (root / "new.md").exists()


def test_regular_file_ancestor_is_rejected_before_any_write(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    (root / "blocked").write_text("regular file\n", encoding="utf-8")
    benchmark = _reports(root, ["new.md", "blocked/result.md"])

    with pytest.raises(StudyError, match="not a directory"):
        write_reports(benchmark)
    assert not (root / "new.md").exists()


def test_directory_symlink_inside_root_remains_writable(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    (root / "alias").symlink_to(root / "reports", target_is_directory=True)
    benchmark = _reports(root, ["alias/new.md"])

    paths = write_reports(benchmark)

    assert paths[0].samefile(root / "reports/new.md")


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin filesystem capability query")
def test_unsupported_case_sensitivity_query_fails_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, ["new.md", "New.md"])

    def unsupported(path: Path, name: int) -> int:
        raise OSError(errno.EINVAL, "unsupported filesystem query")

    monkeypatch.setattr("mblab.benchmark.os.pathconf", unsupported)

    with pytest.raises(StudyError, match="cannot determine filesystem case sensitivity"):
        write_reports(benchmark)
    assert not (root / "new.md").exists()


@pytest.mark.parametrize("paths", [["new.md"], ["first.md", "second.md"]])
def test_distinct_paths_do_not_require_case_sensitivity_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paths: list[str]
) -> None:
    root = _copy(tmp_path)
    benchmark = _reports(root, paths)

    def unsupported(path: Path, name: int) -> int:
        raise OSError(errno.EINVAL, "unsupported filesystem query")

    monkeypatch.setattr("mblab.benchmark.os.pathconf", unsupported, raising=False)

    written = write_reports(benchmark)

    assert all(path.is_file() for path in written)


def test_final_symlink_to_another_report_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    archive = root / "reports/preliminary.md"
    (root / "linked.md").symlink_to(archive)
    benchmark = _reports(root, ["reports/preliminary.md", "linked.md"])
    original = archive.read_bytes()

    with pytest.raises(StudyError, match="duplicate report path"):
        write_reports(benchmark, "report-1")
    assert archive.read_bytes() == original
    assert (root / "linked.md").is_symlink()


@pytest.mark.parametrize(
    ("name", "alias"),
    [("한글.md", "한글.md"), ("Épreuve.md", "épreuve.md"), ("straße.md", "STRASSE.md")],
    ids=["korean-normalization", "accented-case", "sharp-s-case"],
)
def test_unicode_aliases_follow_observed_filesystem_semantics(
    tmp_path: Path, name: str, alias: str
) -> None:
    root = _copy(tmp_path)
    probes = root / "probes"
    probes.mkdir()
    (probes / name).touch()
    aliases = (probes / alias).exists()
    benchmark = _reports(root, [f"new/{name}", f"new/{alias}"])

    if aliases:
        with pytest.raises(StudyError, match="duplicate report path"):
            write_reports(benchmark)
        assert not (root / "new").exists()
    else:
        paths = write_reports(benchmark)
        assert not paths[0].samefile(paths[1])


@pytest.mark.parametrize("reverse", [False, True])
def test_absent_output_cannot_be_another_outputs_ancestor(tmp_path: Path, reverse: bool) -> None:
    root = _copy(tmp_path)
    paths = ["summary.md", "summary.md/detail.md"]
    benchmark = _reports(root, list(reversed(paths)) if reverse else paths)

    with pytest.raises(StudyError, match="report path"):
        write_reports(benchmark)
    assert not (root / "summary.md").exists()
