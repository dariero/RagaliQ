"""GitHub Actions integration helpers for RagaliQ.

Provides CI environment detection, workflow command output (annotations,
step summaries, outputs), and a Markdown summary formatter for evaluation
results.  All helpers are safe to call outside GitHub Actions — they
degrade gracefully (return ``False`` or silently skip writes).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult


def is_ci() -> bool:
    """Detect whether the process is running inside any CI system.

    Returns:
        ``True`` when the ``CI`` environment variable is set to a truthy value.
    """
    return os.environ.get("CI", "").lower() in {"true", "1", "yes"}


def is_github_actions() -> bool:
    """Detect whether the process is running inside GitHub Actions.

    Returns:
        ``True`` when the ``GITHUB_ACTIONS`` environment variable equals ``"true"``.
    """
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def set_output(name: str, value: str) -> None:
    """Write a key-value pair to the ``$GITHUB_OUTPUT`` file.

    In GitHub Actions, subsequent steps can read this value with
    ``${{ steps.<id>.outputs.<name> }}``.  Outside Actions this is a no-op.

    Args:
        name: Output parameter name.
        value: Output parameter value (multi-line values are supported via
            the heredoc protocol, but this helper uses the single-line form).
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def write_step_summary(markdown: str) -> None:
    """Append Markdown content to the GitHub Actions step summary.

    The rendered Markdown appears in the workflow run UI below the step log.
    Outside Actions this is a no-op.

    Args:
        markdown: Markdown-formatted string to append.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown)
        if not markdown.endswith("\n"):
            fh.write("\n")


def create_annotations(
    results: list[RAGTestResult],
    threshold: float = 0.7,
) -> None:
    """Emit GitHub Actions annotations for failed or errored test cases.

    Each failing result produces an ``::error::`` workflow command written
    to *stdout*.  GitHub Actions parses these and surfaces them in the
    workflow run UI and on pull-request diffs.

    Args:
        results: Evaluation results to inspect.
        threshold: Score threshold used for pass/fail classification.
    """
    for result in results:
        if result.passed:
            continue

        tc = result.test_case
        failing = [
            f"{metric}={score:.2f}" for metric, score in result.scores.items() if score < threshold
        ]
        detail = ", ".join(failing) if failing else f"status={result.status}"
        message = f"RagaliQ: '{tc.name}' failed — {detail}"

        # Escape characters that break the workflow command protocol
        safe_msg = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{safe_msg}", flush=True)  # noqa: T201


def format_summary_markdown(
    results: list[RAGTestResult],
    threshold: float = 0.7,
) -> str:
    """Build a Markdown summary table suitable for ``write_step_summary()``.

    Args:
        results: Evaluation results to summarize.
        threshold: Score threshold for pass/fail classification.

    Returns:
        A Markdown string with a header, summary stats, and a results table.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    lines: list[str] = [
        "## RagaliQ Evaluation Results\n",
        f"**{passed}/{total}** test cases passed ({failed} failed) — threshold: {threshold}\n",
    ]

    if not results:
        return "\n".join(lines)

    # Collect evaluator names across all results (handles error envelopes with empty scores)
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.scores.keys())
    evaluator_names = sorted(all_keys)

    # --- Results table ---
    header_cols = ["Test Case", "Status"] + evaluator_names
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cols) + " |")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        score_cells = [f"{r.scores.get(ev, 0.0):.2f}" for ev in evaluator_names]
        row = [r.test_case.name, status, *score_cells]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def emit_ci_summary(
    results: list[RAGTestResult],
    threshold: float = 0.7,
) -> None:
    """High-level helper: write step summary + annotations in one call.

    Intended to be called from the CLI ``run`` command when GitHub Actions
    is detected.  Safe to call outside Actions (both sub-calls are no-ops).

    Args:
        results: Evaluation results.
        threshold: Score threshold for pass/fail classification.
    """
    write_step_summary(format_summary_markdown(results, threshold=threshold))
    create_annotations(results, threshold=threshold)

    # Set outputs for downstream steps
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    set_output("total", str(total))
    set_output("passed", str(passed))
    set_output("failed", str(total - passed))
    set_output("pass_rate", f"{passed / total:.4f}" if total else "0.0000")


