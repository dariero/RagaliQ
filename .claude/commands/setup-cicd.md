# setup-cicd

## Purpose
Configure CI/CD pipelines for automated LLM/RAG quality testing. Set up GitHub Actions workflows, test matrices, and quality gates for AI systems.

## Usage
Invoke when:
- Setting up GitHub Actions for RAG testing
- Configuring test matrices for multiple evaluators
- Implementing quality gates with thresholds
- Adding coverage and reporting to CI

## Automated Steps

1. **Analyze project requirements**
   - Check existing CI configuration
   - Review test structure and markers
   - Identify required secrets (API keys)

2. **Create/update workflow files**
   ```
   .github/workflows/
   ├── tests.yml           # Main test workflow
   ├── rag-quality.yml     # RAG-specific quality checks
   └── release.yml         # Release automation
   ```

3. **Configure test matrix**
   - Python versions (3.14+)
   - Evaluator combinations
   - Fast vs. full test suites

4. **Add quality gates**
   - Minimum pass rate thresholds
   - Coverage requirements
   - Type checking enforcement

5. **Set up reporting**
   - Test result summaries
   - HTML report artifacts
   - PR comments with results

6. **Document secrets setup**
   - Add to README
   - Create setup instructions

## Domain Expertise Applied

### GitHub Actions Workflows

**1. Main Test Workflow**
```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.14"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run linting
        run: make lint

      - name: Run type checking
        run: make typecheck

      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=ragaliq --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    # Only run on main or when explicitly requested
    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'run-integration')

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run integration tests
        run: |
          pytest tests/integration -v --tb=short
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**2. RAG Quality Workflow**
```yaml
# .github/workflows/rag-quality.yml
name: RAG Quality Check

on:
  pull_request:
    paths:
      - 'src/ragaliq/evaluators/**'
      - 'src/ragaliq/judges/**'
      - 'tests/fixtures/**'

jobs:
  quality-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install ragaliq
        run: pip install -e .

      - name: Run RAG quality tests
        id: rag-tests
        run: |
          ragaliq run tests/fixtures/sample_dataset.json \
            --output json --output-file results.json \
            --threshold 0.7
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Generate HTML report
        if: always()
        run: |
          ragaliq run tests/fixtures/sample_dataset.json \
            --output html --output-file report.html
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Upload report artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rag-quality-report
          path: report.html

      - name: Post results to PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('results.json', 'utf8'));

            const summary = `## RAG Quality Results

            | Metric | Value |
            |--------|-------|
            | Total Tests | ${results.summary.total} |
            | Passed | ${results.summary.passed} |
            | Failed | ${results.summary.failed} |
            | Pass Rate | ${(results.summary.pass_rate * 100).toFixed(1)}% |

            [View full report](${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID})`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });
```

**3. Release Workflow**
```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
```

### CI/CD Best Practices for AI Testing

**Quality Gates**
```yaml
- name: Enforce quality threshold
  run: |
    PASS_RATE=$(jq '.summary.pass_rate' results.json)
    if (( $(echo "$PASS_RATE < 0.8" | bc -l) )); then
      echo "::error::Pass rate ${PASS_RATE} is below threshold 0.8"
      exit 1
    fi
```

**Caching for Speed**
```yaml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}

- name: Cache model downloads
  uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: ${{ runner.os }}-hf-models
```

**Secret Management**
```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  # Never log secrets!
```

### Pitfalls to Avoid
- Don't run expensive API tests on every commit - use labels/paths
- Don't hardcode thresholds - use environment variables
- Don't skip integration tests entirely - run on main branch
- Don't expose API keys in logs - use masked secrets

## Interactive Prompts

**Ask for:**
- CI platform (GitHub Actions, GitLab CI, etc.)?
- Test categories to run (unit, integration, RAG)?
- Quality thresholds (pass rate, coverage)?
- Secrets needed (API keys)?
- Notification preferences (PR comments, Slack)?

**Suggest:**
- Appropriate workflow triggers
- Test matrix configuration
- Caching strategy

**Validate:**
- Workflows are syntactically correct
- Secrets are properly referenced
- Quality gates are reasonable

## Success Criteria
- [ ] Workflows created and valid YAML
- [ ] Unit tests run on every PR
- [ ] Integration tests run on main/labeled PRs
- [ ] Quality gates enforce thresholds
- [ ] Reports uploaded as artifacts
- [ ] PR comments with results (optional)
- [ ] Secrets documented in README
- [ ] `act` dry-run passes (if available)
