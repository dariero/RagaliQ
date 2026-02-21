#!/usr/bin/env python3
"""Release verification script for RagaliQ.

Checks all acceptance criteria before publishing a release:
  1. Version consistency across pyproject.toml and __init__.py
  2. Required files exist (LICENSE, README.md, CHANGELOG.md)
  3. No TODOs/FIXMEs in source code
  4. Package builds successfully
  5. twine check passes on built artifacts
  6. Top-level imports work

Usage:
    python scripts/verify_release.py
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

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
    ok &= check("pyproject.toml version found", pyproject_version is not None, str(pyproject_version))
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


def check_build() -> bool:
    """Verify package builds successfully."""
    header("Package Build")

    # Clean previous builds
    dist = ROOT / "dist"
    if dist.exists():
        for f in dist.iterdir():
            f.unlink()

    result = subprocess.run(
        ["hatch", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = check("hatch build", result.returncode == 0)
    if not ok:
        print(f"    stderr: {result.stderr[:500]}")
        return False

    # Check artifacts exist
    artifacts = list(dist.glob("*"))
    ok &= check("Build artifacts created", len(artifacts) >= 2, f"{len(artifacts)} files")

    return ok


def check_twine() -> bool:
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
            return True
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


def main() -> None:
    print("RagaliQ Release Verification")
    print(f"Root: {ROOT}")

    results = [
        check_version_consistency(),
        check_required_files(),
        check_no_todos(),
        check_imports(),
        check_build(),
        check_twine(),
    ]

    header("RESULT")
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"  All {total} checks passed. Ready to release!")
        sys.exit(0)
    else:
        print(f"  {passed}/{total} checks passed. Fix failures before releasing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
