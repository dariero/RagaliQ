#!/usr/bin/env python3
"""Consistency checks for facts RagaliQ stores in more than one place.

  1. Version consistency across pyproject.toml and __init__.py
  2. Python minimum, classifiers and CI test matrix agree
  3. ruff floor, pre-commit rev and lock pin agree on major.minor
  4. Every declared dependency appears in pylock.toml (reported, not enforced)

Usage:
    python scripts/verify_release.py

CI runs it on every PR. The release workflow builds the distributions and runs
twine check itself (.github/workflows/release.yml).
"""

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ragaliq"


def header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    marker = "+" if passed else "x"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{marker}] {label}: {status}{suffix}")
    return passed


def check_version_consistency() -> bool:
    """Verify version is the same in pyproject.toml and __init__.py."""
    header("Version Consistency")

    # Read pyproject.toml version
    pyproject = ROOT / "pyproject.toml"
    pyproject_version = None
    for line in pyproject.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m:
            pyproject_version = m.group(1)
            break

    # Read __init__.py version
    init_file = SRC / "__init__.py"
    init_version = None
    for line in init_file.read_text().splitlines():
        m = re.match(r'^__version__\s*=\s*"([^"]+)"', line)
        if m:
            init_version = m.group(1)
            break

    ok = True
    ok &= check(
        "pyproject.toml version found", pyproject_version is not None, str(pyproject_version)
    )
    ok &= check("__init__.py version found", init_version is not None, str(init_version))
    ok &= check(
        "Versions match",
        pyproject_version == init_version,
        f"pyproject={pyproject_version} init={init_version}",
    )
    return ok


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pylock() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pylock.toml").read_text(encoding="utf-8"))


def _two_part(text: str) -> str | None:
    """Pull a bare `major.minor` out of any of the forms these files use."""
    m = re.search(r"(\d+\.\d+)", text)
    return m.group(1) if m else None


def _dist_name(spec: str) -> str:
    return re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")


def _floor(spec: str) -> str | None:
    """Pull the `major.minor` of a `>=` lower bound out of a version specifier."""
    m = re.search(r">=\s*(\d+\.\d+)", spec)
    return m.group(1) if m else None


def _version_key(version: str) -> tuple[int, int]:
    """Order `major.minor` strings numerically, so 3.9 sorts before 3.10."""
    major, minor = version.split(".")
    return int(major), int(minor)


def check_python_atom() -> bool:
    """The supported Python range is declared the same way at every site.

    The minimum is the `>=` floor of requires-python. ruff's target-version,
    mypy's python_version, the Dockerfile (x2), every single-version
    `python-version:` in the workflows and pylock.toml's requires-python
    equal it. The maximum is the highest `Programming Language :: Python ::
    X.Y` classifier. The classifiers run from the minimum to the maximum with
    no gap, and the CI test matrix (a `python-version: [...]` list) holds
    exactly the classified versions, each written as a plain X.Y. ci.yml's
    `${{ matrix.python-version }}` is derived from the matrix list, so it is
    not an independent site.
    """
    header("Python version atom")
    data = _pyproject()
    proj = data["project"]
    tool = data.get("tool", {})
    sites: list[tuple[str, str | None]] = []
    matrices: list[tuple[str, list[str]]] = []

    minimum = _floor(proj.get("requires-python", ""))
    sites.append(("pyproject requires-python", minimum))

    classified: list[str] = []
    for c in proj.get("classifiers", []):
        if cm := re.fullmatch(r"Programming Language :: Python :: (\d+\.\d+)", c):
            classified.append(cm.group(1))
    classified.sort(key=_version_key)

    target = str(tool.get("ruff", {}).get("target-version", ""))
    m = re.fullmatch(r"py(\d)(\d+)", target)
    sites.append(("ruff target-version", f"{m.group(1)}.{m.group(2)}" if m else None))

    sites.append(
        ("mypy python_version", _two_part(str(tool.get("mypy", {}).get("python_version", ""))))
    )

    for i, line in enumerate((ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("FROM python:"):
            sites.append((f"Dockerfile:{i}", _two_part(line)))

    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for i, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            if "python-version" not in line or "${{" in line:
                continue
            # A list is a test matrix. Read it whole: _two_part would return
            # only its first version and pass a matrix that lacks the maximum.
            # A list left open continues on lines this scan does not read, so
            # it counts as empty rather than as its first line.
            if mx := re.search(r"python-version:\s*\[([^\]]*)(\]?)", line):
                items = [s.strip().strip("\"'") for s in mx.group(1).split(",") if s.strip()]
                if mx.group(2):
                    matrices.append((f"{wf.name}:{i}", items))
                else:
                    matrices.append((f"{wf.name}:{i} (list not closed on its line)", []))
            elif (ver := _two_part(line)) is not None:
                sites.append((f"{wf.name}:{i}", ver))

    sites.append(("pylock requires-python", _floor(str(_pylock().get("requires-python", "")))))

    for label, value in sites:
        print(f"      {label}: {value if value is not None else 'UNPARSEABLE'}")
    print(f"      classifiers: {', '.join(classified) or 'none'}")
    for label, versions in matrices:
        print(f"      {label} matrix: {', '.join(versions) or 'none'}")

    values = {v for _, v in sites}
    ok = check(
        f"Python minimum agrees across {len(sites)} sites",
        len(values) == 1 and None not in values,
        f"{minimum if len(values) == 1 else sorted(str(v) for v in values)}",
    )

    expected: list[str] = []
    if minimum is not None and classified:
        major, low = _version_key(minimum)
        top_major, high = _version_key(classified[-1])
        if top_major == major:
            expected = [f"{major}.{n}" for n in range(low, high + 1)]
    in_range = bool(classified) and classified == expected
    ok &= check(
        "Python classifiers run from the minimum with no gap",
        in_range,
        f"{classified[0]} to {classified[-1]}"
        if in_range
        else f"classifiers={classified} expected={expected}",
    )

    mismatched = [label for label, versions in matrices if sorted(versions) != sorted(classified)]
    ok &= check(
        "CI test matrix equals the classifiers",
        bool(matrices) and not mismatched,
        f"mismatched: {mismatched}"
        if mismatched
        else ", ".join(label for label, _ in matrices) or "no python-version list found",
    )
    return ok


def check_ruff_atom() -> bool:
    """The ruff floor, the pre-commit rev and the lock pin agree on major.minor.

    A split leaves the pre-commit hook and `.venv/bin/ruff` formatting the tree
    differently.
    """
    header("ruff version atom")
    data = _pyproject()
    sites: list[tuple[str, str | None]] = []

    for spec in data["project"]["optional-dependencies"]["dev"]:
        if _dist_name(spec) == "ruff":
            sites.append(("dev extra floor", _two_part(spec)))

    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    if m := re.search(r"ruff-pre-commit\s*\n\s*rev:\s*v?([\d.]+)", text):
        sites.append(("pre-commit rev", _two_part(m.group(1))))

    for pkg in _pylock().get("packages", []):
        if pkg["name"].lower() == "ruff":
            sites.append(("pylock pin", _two_part(pkg.get("version", ""))))

    values = {v for _, v in sites}
    ok = len(sites) >= 3 and len(values) == 1 and None not in values
    for label, value in sites:
        print(f"      {label}: {value}")
    return check(
        f"ruff atom agrees across {len(sites)} sites",
        ok,
        f"{values.pop() if ok else sorted(str(v) for v in values)}",
    )


def check_floors_against_lock() -> bool:
    """Report any declared dependency that pylock.toml does not carry.

    CI installs with `uv pip sync pylock.toml` plus `--no-deps`, so anything
    absent from the lock is never exercised. Reported, not failed: the docs
    extra is installed only by the Docs workflow, not from the lock.
    """
    header("Declared dependencies vs lock")
    proj = _pyproject()["project"]
    locked = {p["name"].lower().replace("_", "-") for p in _pylock().get("packages", [])}

    declared: list[tuple[str, str]] = [("runtime", s) for s in proj.get("dependencies", [])]
    for group, specs in proj.get("optional-dependencies", {}).items():
        declared += [(f"extra:{group}", s) for s in specs]

    absent = [(g, s) for g, s in declared if _dist_name(s) not in locked]
    for group, spec in absent:
        print(f"      NOT IN LOCK  {group:<14} {spec}  (never exercised by ci.yml)")

    return check(
        "Declared dependencies present in pylock.toml",
        True,
        f"{len(declared) - len(absent)}/{len(declared)} locked"
        + (f", {len(absent)} absent (reported, not enforced)" if absent else ""),
    )


def main() -> None:
    print("RagaliQ Consistency Checks")
    print(f"Root: {ROOT}")

    results = [
        check_version_consistency(),
        check_python_atom(),
        check_ruff_atom(),
        check_floors_against_lock(),
    ]

    header("RESULT")
    total = len(results)
    passed = sum(results)
    if passed < total:
        print(f"  {passed}/{total} passed, {total - passed} failed. Fix failures before releasing.")
        sys.exit(1)
    print(f"  All {total} consistency checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
