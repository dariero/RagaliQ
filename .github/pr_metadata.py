#!/usr/bin/env python3
"""
Auto-generate GitHub PR metadata for RagaliQ project.
Infers labels, milestone, and other fields from branch name and commits.
"""

import re
import subprocess
import sys
from typing import Dict, List


def run_git_command(cmd: List[str]) -> str:
    """Run a git command and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def get_current_branch() -> str:
    """Get the current git branch name."""
    return run_git_command(["git", "branch", "--show-current"])


def get_commits_since_main() -> List[str]:
    """Get commit messages since branching from main."""
    commits = run_git_command(["git", "log", "main..HEAD", "--oneline"])
    return commits.split("\n") if commits else []


def get_changed_files() -> List[str]:
    """Get list of changed files since main."""
    files = run_git_command(["git", "diff", "main...HEAD", "--name-only"])
    return files.split("\n") if files else []


def infer_labels(branch: str, commits: List[str], files: List[str]) -> List[str]:
    """
    Infer 2-3 labels from branch name, commits, and changed files.

    Type labels (pick 1): feat, bug, refactor, chore, docs, research
    Scope labels (pick 0-2): judge, evaluator, core, cli, dataset, report, pytest, async, infra, testing
    """
    labels = set()

    # Combine all text for analysis
    text = f"{branch} {' '.join(commits)} {' '.join(files)}".lower()

    # Label inference rules
    type_rules = {
        "feat": ["feat", "feature", "add", "new", "implement"],
        "bug": ["bug", "fix", "error", "issue", "broken"],
        "refactor": ["refactor", "restructure", "reorganize", "streamline", "improve"],
        "chore": ["chore", "tooling", "upgrade", "dependencies", "setup", "config"],
        "docs": ["docs", "readme", "documentation", "comment", "docstring"],
        "research": ["research", "explore", "experiment", "investigate", "study"],
    }

    scope_rules = {
        "judge": ["judge", "llm", "claude", "openai", "gpt", "anthropic"],
        "evaluator": ["evaluator", "metric", "faithfulness", "relevance", "hallucination"],
        "core": ["core", "base", "runner", "testcase", "pipeline", "abstract"],
        "cli": ["cli", "command", "typer", "click", "argparse"],
        "dataset": ["dataset", "data", "synthetic", "generation", "test-set"],
        "report": ["report", "output", "format", "html", "json", "terminal"],
        "pytest": ["pytest", "plugin", "fixture", "conftest"],
        "async": ["async", "await", "asyncio", "concurrent"],
        "infra": ["infra", "ci", "cd", "docker", "pypi", "github-actions"],
        "testing": ["testing", "test", "unittest", "coverage"],
    }

    # Check type labels
    type_labels = set()
    for label, keywords in type_rules.items():
        if any(keyword in text for keyword in keywords):
            type_labels.add(label)

    # Check scope labels
    scope_labels = set()
    for label, keywords in scope_rules.items():
        if any(keyword in text for keyword in keywords):
            scope_labels.add(label)

    # Combine: 1 type + up to 2 scope labels
    result = list(type_labels)[:1] + list(scope_labels)[:2]

    return result or ["feat"]  # Default to feat if nothing matches


def infer_milestone(branch: str, commits: List[str]) -> str:
    """
    Infer milestone from task scope.

    Milestones: phase-1-foundation, phase-2-advanced, phase-3-production
    """
    text = f"{branch} {' '.join(commits)}".lower()

    # Phase 1: Core features (judge, evaluator, runner)
    if any(keyword in text for keyword in ["judge", "evaluator", "core", "runner", "base"]):
        return "phase-1-foundation"

    # Phase 2: Advanced features (async, plugins, integrations)
    if any(keyword in text for keyword in ["async", "plugin", "integration", "pytest"]):
        return "phase-2-advanced"

    # Phase 3: Production features (cli, reports, deployment)
    if any(keyword in text for keyword in ["cli", "deploy", "report", "production"]):
        return "phase-3-production"

    return "phase-1-foundation"  # Default to phase 1


def generate_pr_metadata() -> Dict[str, any]:
    """Generate complete PR metadata."""
    branch = get_current_branch()
    commits = get_commits_since_main()
    files = get_changed_files()

    # Extract issue number from branch (e.g., feat/12-add-judge -> 12)
    issue_match = re.search(r'\d+', branch)
    issue_number = int(issue_match.group()) if issue_match else None

    metadata = {
        "assignees": ["dariero"],
        "labels": infer_labels(branch, commits, files),
        "project": "RagaliQ",
        "milestone": infer_milestone(branch, commits),
        "issue_number": issue_number,
        "branch": branch,
    }

    return metadata


def format_gh_pr_flags(metadata: Dict[str, any]) -> List[str]:
    """Convert metadata to gh pr create flags."""
    flags = []

    # Assignees
    for assignee in metadata["assignees"]:
        flags.extend(["--assignee", assignee])

    # Labels
    for label in metadata["labels"]:
        flags.extend(["--label", label])

    # Milestone
    if metadata["milestone"]:
        flags.extend(["--milestone", metadata["milestone"]])

    return flags


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # Output JSON for programmatic use
        import json
        metadata = generate_pr_metadata()
        print(json.dumps(metadata, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "--gh-flags":
        # Output gh pr create flags
        metadata = generate_pr_metadata()
        flags = format_gh_pr_flags(metadata)
        print(" ".join(flags))
    else:
        # Human-readable output
        metadata = generate_pr_metadata()
        print("\n🏷️  PR Metadata (auto-generated)\n")
        print(f"Branch:    {metadata['branch']}")
        print(f"Issue:     #{metadata['issue_number']}" if metadata['issue_number'] else "Issue:     N/A")
        print(f"Assignees: {', '.join(metadata['assignees'])}")
        print(f"Labels:    {', '.join(metadata['labels'])}")
        print(f"Project:   {metadata['project']}")
        print(f"Milestone: {metadata['milestone']}")
        print("\n" + "─" * 50)
        print("\nTo use with gh pr create:")
        print(f"  gh pr create $(python .github/pr_metadata.py --gh-flags)")


if __name__ == "__main__":
    main()
