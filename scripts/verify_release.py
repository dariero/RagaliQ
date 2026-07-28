#!/usr/bin/env python3
"""Release verification script for RagaliQ.

Checks all acceptance criteria before publishing a release:
  1. Version consistency across pyproject.toml and __init__.py
  2. Python version agrees across every site that declares it
  3. ruff floor, pre-commit rev and lock pin agree on major.minor
  4. The hatch default env references the dev extra rather than copying it
  5. Every declared dependency appears in pylock.toml (reported, not enforced)
  6. Required files exist (LICENSE, README.md, CHANGELOG.md)
  7. No TODOs/FIXMEs in source code
  8. Package builds successfully
  9. twine check passes on built artifacts
 10. Top-level imports work

Usage:
    python scripts/verify_release.py                     # all 10, before a release
    python scripts/verify_release.py --consistency-only  # checks 1-5, in CI

`--consistency-only` runs the atom checks and nothing else: no `python -m build`
on every push, and no TODO scan turning main red on unrelated work. Those five
are the ones that guard facts stored in more than one place, so they are worth
running on every PR.
"""

import importlib
import re
import subprocess
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


def check_required_files() -> bool:
    """Verify required project files exist."""
    header("Required Files")

    required = ["LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml"]
    ok = True
    for name in required:
        exists = (ROOT / name).exists()
        ok &= check(name, exists)
    return ok


def check_no_todos() -> bool:
    """Verify no TODO/FIXME/HACK/XXX markers in source code."""
    header("No TODOs in Source")

    patterns = ["TODO", "FIXME", "HACK", "XXX"]
    found: list[str] = []

    for py_file in SRC.rglob("*.py"):
        rel = py_file.relative_to(ROOT)
        for i, line in enumerate(py_file.read_text().splitlines(), 1):
            for pattern in patterns:
                if pattern in line and "noqa" not in line:
                    found.append(f"  {rel}:{i}: {line.strip()}")

    ok = check("No TODO/FIXME/HACK/XXX in src/", len(found) == 0, f"{len(found)} found")
    for hit in found[:10]:
        print(f"    {hit}")
    return ok


def check_build() -> bool | None:
    """Verify the package builds via the PEP 517 frontend (`python -m build`).

    Uses the standard build frontend rather than `hatch build` so the check runs
    on any environment with `build` installed, without requiring `hatch` on PATH.
    """
    header("Package Build")

    dist = ROOT / "dist"
    if dist.exists():
        for f in dist.iterdir():
            f.unlink()

    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "No module named build" in result.stderr:
            print("  [~] build not installed — skipping (install with: pip install build)")
            return None
        check("python -m build", False)
        print(f"    stderr: {result.stderr[:500]}")
        return False

    ok = check("python -m build", True)
    artifacts = list(dist.glob("*"))
    ok &= check("Build artifacts created", len(artifacts) >= 2, f"{len(artifacts)} files")
    return ok


def check_twine() -> bool | None:
    """Verify twine check passes on built artifacts."""
    header("Twine Check")

    dist = ROOT / "dist"
    artifacts = list(dist.glob("*"))
    if not artifacts:
        check("twine check", False, "no dist/ artifacts found")
        return False

    result = subprocess.run(
        ["python", "-m", "twine", "check", *[str(a) for a in artifacts]],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # twine may not be installed — that's acceptable in dev
        if "No module named" in result.stderr:
            print("  [~] twine not installed — skipping (install with: pip install twine)")
            return None
        check("twine check", False, result.stdout.strip())
        return False

    return check("twine check", True)


def check_imports() -> bool:
    """Verify top-level imports work."""
    header("Import Check")

    try:
        mod = importlib.import_module("ragaliq")
        expected = [
            "RagaliQ",
            "RAGTestCase",
            "RAGTestResult",
            "EvalStatus",
            "Evaluator",
            "EvaluationResult",
            "ClaudeJudge",
            "LLMJudge",
            "JudgeConfig",
            "DatasetLoader",
            "TestCaseGenerator",
            "ConsoleReporter",
            "HTMLReporter",
            "JSONReporter",
            "__version__",
        ]
        missing = [name for name in expected if not hasattr(mod, name)]
        ok = check("Top-level imports", len(missing) == 0, f"missing: {missing}" if missing else "")
        return ok
    except Exception as e:
        check("Top-level imports", False, str(e))
        return False


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


def check_python_atom() -> bool:
    """Every declaration of the Python version agrees.

    The same fact is stored in pyproject (x4), the Dockerfile (x2), the
    workflows and pylock.toml. ci.yml's `${{ matrix.python-version }}` is
    derived from the matrix list, so it is not an independent site.
    """
    header("Python version atom")
    data = _pyproject()
    proj = data["project"]
    tool = data.get("tool", {})
    sites: list[tuple[str, str | None]] = []

    sites.append(("pyproject requires-python", _two_part(proj.get("requires-python", ""))))

    versioned = [
        c
        for c in proj.get("classifiers", [])
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", c)
    ]
    sites.append(("pyproject classifier", _two_part(versioned[0]) if len(versioned) == 1 else None))

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
            if (ver := _two_part(line)) is not None:
                sites.append((f"{wf.name}:{i}", ver))

    sites.append(("pylock requires-python", _two_part(str(_pylock().get("requires-python", "")))))

    values = {v for _, v in sites}
    ok = len(values) == 1 and None not in values
    for label, value in sites:
        print(f"      {label}: {value if value is not None else 'UNPARSEABLE'}")
    return check(
        f"Python atom agrees across {len(sites)} sites",
        ok,
        f"{values.pop() if ok else sorted(str(v) for v in values)}",
    )


def check_ruff_atom() -> bool:
    """The ruff floor, the pre-commit rev and the lock pin agree on major.minor.

    A split leaves the pre-commit hook and `hatch run lint` formatting the tree
    differently. The hatch env no longer duplicates the floor (it uses
    `features = ["dev"]`), so this is three sites, not four.
    """
    header("ruff version atom")
    data = _pyproject()
    sites: list[tuple[str, str | None]] = []

    for spec in data["project"]["optional-dependencies"]["dev"]:
        if _dist_name(spec) == "ruff":
            sites.append(("dev extra floor", _two_part(spec)))

    for spec in data["tool"]["hatch"]["envs"]["default"].get("dependencies", []):
        if _dist_name(spec) == "ruff":
            sites.append(("hatch env floor (should not exist)", _two_part(spec)))

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


def check_dev_lists() -> bool:
    """The hatch default env references the dev extra rather than copying it.

    Two copies of the tool list means the local gates and the CI gates can
    install different tools. `build` is the one allowed extra: it is needed by
    `hatch run build` but is not a dependency of the published package.
    """
    header("Dev dependency list")
    env = _pyproject()["tool"]["hatch"]["envs"]["default"]
    features = env.get("features", [])
    extras = {_dist_name(s) for s in env.get("dependencies", [])}
    allowed = {"build"}

    ok = check("hatch env uses features = ['dev']", "dev" in features, str(features))
    ok &= check(
        "hatch env adds nothing beyond the allowed extras",
        extras <= allowed,
        f"extras={sorted(extras) or 'none'} allowed={sorted(allowed)}",
    )
    return ok


def check_floors_against_lock() -> bool:
    """Report any declared dependency that pylock.toml does not carry.

    CI installs with `uv pip sync pylock.toml` plus `--no-deps`, so anything
    absent from the lock is never exercised. Reported, not failed: the openai
    extra is a known dead declaration pending its own decision.
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
    # --consistency-only runs the atom checks and nothing else, so CI can gate
    # every PR on them without paying for `python -m build` on each push and
    # without check_no_todos turning main red on unrelated work.
    consistency_only = "--consistency-only" in sys.argv

    mode = "Consistency Checks" if consistency_only else "Release Verification"
    print(f"RagaliQ {mode}")
    print(f"Root: {ROOT}")

    consistency: list[bool | None] = [
        check_version_consistency(),
        check_python_atom(),
        check_ruff_atom(),
        check_dev_lists(),
        check_floors_against_lock(),
    ]

    results: list[bool | None] = consistency
    if not consistency_only:
        results = [
            *consistency,
            check_required_files(),
            check_no_todos(),
            check_imports(),
            check_build(),
            check_twine(),
        ]

    header("RESULT")
    total = len(results)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)

    if failed:
        print(
            f"  {passed}/{total} passed, {failed} failed, {skipped} skipped. "
            "Fix failures before releasing."
        )
        sys.exit(1)
    if skipped:
        # A skipped check (missing build/twine) is NOT verified. Surface it loudly
        # so a skip can never masquerade as a green "ready to release" gate.
        print(
            f"  {passed}/{total} passed, {skipped} skipped (NOT verified). "
            "Install the missing tools to fully verify before releasing."
        )
        sys.exit(0)
    if consistency_only:
        # Never say "ready to release" after running half the checks — that is
        # the same skip-as-green failure this script guards against elsewhere.
        print(f"  All {total} consistency checks passed. Release checks NOT run.")
    else:
        print(f"  All {total} checks passed. Ready to release!")
    sys.exit(0)


if __name__ == "__main__":
    main()
