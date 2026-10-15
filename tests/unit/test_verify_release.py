"""Unit tests for the Python version atom in scripts/verify_release.py.

The script is not a package, so it is loaded from its path. Each test writes
the files `check_python_atom` reads into a temporary root and points the
module's ROOT there; the last test runs the check on the repository itself.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_release.py"

MINIMUM = "Python minimum agrees"
RANGE = "Python classifiers run from the minimum"
MATRIX = "CI test matrix equals the classifiers"


@pytest.fixture(scope="module")
def verify_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def root(verify_release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verify_release, "ROOT", tmp_path)
    return tmp_path


def _write_tree(
    root: Path,
    *,
    requires: str = ">=3.14",
    classifiers: tuple[str, ...] = ("3.14", "3.15"),
    ruff: str = "py314",
    mypy: str = "3.14",
    docker: str = "3.14",
    setup: str = "3.14",
    matrix: str = '["3.14", "3.15"]',
    lock: str = ">=3.14",
) -> None:
    """Write every file the check reads, declaring the given versions."""
    listed = "".join(f'    "Programming Language :: Python :: {v}",\n' for v in classifiers)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nrequires-python = "{requires}"\n'
        f'classifiers = [\n    "Programming Language :: Python :: 3",\n{listed}]\n\n'
        f'[tool.ruff]\ntarget-version = "{ruff}"\n\n'
        f'[tool.mypy]\npython_version = "{mypy}"\n',
        encoding="utf-8",
    )
    (root / "pylock.toml").write_text(
        f'lock-version = "1.0"\nrequires-python = "{lock}"\n', encoding="utf-8"
    )
    (root / "Dockerfile").write_text(
        f"FROM python:{docker}-slim AS builder\nFROM python:{docker}-slim AS runtime\n",
        encoding="utf-8",
    )
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n"
        "  lint:\n"
        "    steps:\n"
        "      - uses: actions/setup-python@v7.0.0\n"
        "        with:\n"
        f'          python-version: "{setup}"\n'
        "  test:\n"
        "    strategy:\n"
        "      matrix:\n"
        f"        python-version: {matrix}\n"
        "    steps:\n"
        "      - uses: actions/setup-python@v7.0.0\n"
        "        with:\n"
        "          python-version: ${{ matrix.python-version }}\n",
        encoding="utf-8",
    )


def _run_check(verify_release: ModuleType, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Run the check; return its FAIL lines, asserting they match its result."""
    passed = verify_release.check_python_atom()
    failed = [line for line in capsys.readouterr().out.splitlines() if ": FAIL" in line]
    assert passed is (not failed)
    return failed


def test_minimum_with_a_higher_maximum_passes(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root)
    assert _run_check(verify_release, capsys) == []


def test_a_single_supported_version_passes(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=("3.14",), matrix='["3.14"]')
    assert _run_check(verify_release, capsys) == []


def test_versions_compare_as_numbers(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(
        root,
        requires=">=3.9",
        classifiers=("3.10", "3.9"),
        ruff="py39",
        mypy="3.9",
        docker="3.9",
        setup="3.9",
        matrix='["3.10", "3.9"]',
        lock=">=3.9",
    )
    assert _run_check(verify_release, capsys) == []


def test_minimum_is_the_lower_bound_of_requires_python(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, requires="<3.16,>=3.14", lock="<3.16,>=3.14")
    assert _run_check(verify_release, capsys) == []


@pytest.mark.parametrize(
    "site",
    [
        {"ruff": "py315"},
        {"mypy": "3.15"},
        {"docker": "3.15"},
        {"setup": "3.15"},
        {"lock": ">=3.15"},
    ],
    ids=["ruff", "mypy", "dockerfile", "workflow", "pylock"],
)
def test_a_minimum_site_at_the_maximum_fails(
    verify_release: ModuleType,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    site: dict[str, str],
) -> None:
    _write_tree(root, **site)
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MINIMUM in failed[0]


def test_lowest_classifier_above_the_minimum_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=("3.15",), matrix='["3.15"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and RANGE in failed[0]


def test_lowest_classifier_below_the_minimum_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=("3.13", "3.14", "3.15"), matrix='["3.13", "3.14", "3.15"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and RANGE in failed[0]


def test_gap_in_the_classifiers_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=("3.14", "3.16"), matrix='["3.14", "3.16"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and RANGE in failed[0]


def test_no_versioned_classifier_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=(), matrix="[]")
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and RANGE in failed[0]


def test_matrix_without_the_maximum_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, matrix='["3.14"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MATRIX in failed[0]


def test_matrix_beyond_the_classifiers_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, classifiers=("3.14",), matrix='["3.14", "3.15"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MATRIX in failed[0]


@pytest.mark.parametrize("entry", ["3.15t", "pypy3.15", "3.15-dev"])
def test_matrix_entry_other_than_plain_x_y_fails(
    verify_release: ModuleType,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    entry: str,
) -> None:
    _write_tree(root, matrix=f'["3.14", "{entry}"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MATRIX in failed[0]


def test_matrix_list_left_open_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, matrix='["3.14",\n          "3.15"]')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MATRIX in failed[0] and "not closed" in failed[0]


def test_matrix_not_written_as_a_flow_list_fails(
    verify_release: ModuleType, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(root, matrix='\n          - "3.14"\n          - "3.15"')
    failed = _run_check(verify_release, capsys)
    assert len(failed) == 1 and MATRIX in failed[0] and "no python-version list" in failed[0]


def test_repository_declares_a_consistent_range(
    verify_release: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_check(verify_release, capsys) == []
