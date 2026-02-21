# CI/CD Example

Example GitHub Actions workflow for running RagaliQ evaluations in CI.

## Setup

1. Copy `ragaliq-ci.yml` to `.github/workflows/` in your repository
2. Add your `ANTHROPIC_API_KEY` as a repository secret
3. Place your dataset at `datasets/test_cases.json` (or update the path in the workflow)

## Features

When running in GitHub Actions, RagaliQ automatically:

- **Disables Rich output** — no animated spinners cluttering CI logs
- **Writes a step summary** — Markdown results table visible in the Actions run UI
- **Creates annotations** — failing test cases appear as error annotations on PR diffs
- **Sets step outputs** — `total`, `passed`, `failed`, `pass_rate` available for downstream steps
- **Exits with code 1** on any failure — blocks merging when used as a required check
