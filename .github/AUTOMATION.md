# GitHub Automation

## PR Metadata Auto-Filler

Automatically infers and fills PR metadata (labels, assignees, milestone) based on branch name and commit history.

### Usage

#### 1. Standalone (view metadata)
```bash
python .github/pr_metadata.py
```

Output:
```
🏷️  PR Metadata (auto-generated)

Branch:    feat/12-add-claude-judge
Issue:     #12
Assignees: dariero
Labels:    feat, judge, core
Project:   RagaliQ
Milestone: phase-1-foundation
```

#### 2. JSON output (for scripts)
```bash
python .github/pr_metadata.py --json
```

```json
{
  "assignees": ["dariero"],
  "labels": ["feat", "judge", "core"],
  "project": "RagaliQ",
  "milestone": "phase-1-foundation",
  "issue_number": 12,
  "branch": "feat/12-add-claude-judge"
}
```

#### 3. GitHub CLI flags (integrated into `/ship`)
```bash
gh pr create \
  --title "Add Claude Judge" \
  --body "..." \
  $(python .github/pr_metadata.py --gh-flags)
```

This automatically adds:
- `--assignee dariero`
- `--label feat --label judge --label core`
- `--milestone phase-1-foundation`

### Label Inference Rules

The script analyzes your branch name, commits, and changed files to infer 2-3 labels:

| Label | Keywords |
|-------|----------|
| `judge` | judge, llm, claude, openai, gpt |
| `evaluator` | evaluator, metric, faithfulness, relevance, hallucination |
| `async` | async, await, asyncio, concurrent |
| `core` | core, base, runner, testcase, pipeline |
| `bug` | bug, fix, error, issue, broken |
| `feat` | feat, feature, add, new, implement |
| `docs` | docs, readme, documentation, comment, docstring |

**Prioritization**: `feat` and `bug` labels are prioritized first (max 1), followed by others (max 2 more).

### Milestone Inference Rules

| Milestone | Scope |
|-----------|-------|
| `phase-1-foundation` | Core features: judge, evaluator, runner, base classes |
| `phase-2-advanced` | Advanced: async, plugins, integrations, pytest |
| `phase-3-production` | Production: CLI, reports, deployment |

**Default**: Falls back to `phase-1-foundation` if no match.

### Integration with `/ship`

The `/ship` command automatically uses this script when creating PRs:

```bash
/ship "Add Claude judge implementation"
```

This will:
1. Run quality gates (lint, typecheck, tests)
2. Auto-generate PR metadata based on your branch/commits
3. Create PR with proper labels, assignees, and milestone
4. Self-review and merge if passing

### Manual Override

If you need to override auto-generated metadata:

```bash
# Create PR manually with custom metadata
gh pr create \
  --assignee dariero \
  --label feat --label judge \
  --milestone phase-2-advanced \
  --title "My Custom Title" \
  --body "My custom description"
```

### Examples

**Example 1: Judge implementation**
```bash
# Branch: feat/3-add-claude-judge
# Commits: "Add Claude API integration", "Implement LLM judge"
# Result: labels=[feat, judge, core], milestone=phase-1-foundation
```

**Example 2: Bug fix**
```bash
# Branch: fix/15-async-timeout
# Commits: "Fix timeout in async executor"
# Result: labels=[bug, async], milestone=phase-2-advanced
```

**Example 3: Documentation**
```bash
# Branch: docs/20-api-reference
# Commits: "Add API docs for evaluators"
# Result: labels=[docs, evaluator], milestone=phase-1-foundation
```
