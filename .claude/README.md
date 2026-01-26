# RagaliQ Claude Commands

Reusable automation commands for developing RagaliQ - the pytest-inspired testing framework for LLM/RAG systems.

## Quick Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/new-evaluator` | Create LLM response evaluator | Adding faithfulness/relevance/custom metrics |
| `/new-judge` | Create LLM judge backend | Adding OpenAI/Cohere/local model support |
| `/add-test-pattern` | Add pytest patterns | Creating fixtures, assertions, markers |
| `/new-metric` | Implement NLP metric | Adding BLEU/ROUGE/custom calculations |
| `/refactor-async` | Convert to async/await | Optimizing I/O-bound operations |
| `/add-cli-command` | Add CLI command | Extending `ragaliq` tool |
| `/integrate-langchain` | LangChain integration | Testing LangChain chains/agents |
| `/setup-cicd` | Configure CI/CD | GitHub Actions for RAG testing |
| `/optimize-prompts` | Improve judge prompts | Fixing inconsistent/slow evaluations |
| `/add-docs-example` | Create documentation | Writing tutorials and examples |

## Usage

Invoke any command by typing its name:

```
/new-evaluator
```

Each command will:
1. Ask clarifying questions about your requirements
2. Analyze existing codebase patterns
3. Generate implementation with tests
4. Run verification (`make test && make typecheck`)

## Command Categories

### Core Development
- **`/new-evaluator`** - Build evaluators that assess RAG response quality
- **`/new-judge`** - Integrate new LLM providers (Claude, OpenAI, local)
- **`/new-metric`** - Add traditional NLP metrics (BLEU, ROUGE, etc.)

### Testing & Quality
- **`/add-test-pattern`** - Create pytest fixtures and assertion helpers
- **`/optimize-prompts`** - Improve prompt accuracy and reduce latency
- **`/setup-cicd`** - Configure automated quality gates

### Integrations
- **`/integrate-langchain`** - Connect with LangChain ecosystem
- **`/add-cli-command`** - Extend command-line interface
- **`/refactor-async`** - Optimize for parallel execution

### Documentation
- **`/add-docs-example`** - Write tutorials and API documentation

## Project Structure Reference

```
src/ragaliq/
├── core/           # RAGTestCase, Evaluator base, Runner
├── evaluators/     # Faithfulness, Relevance, Hallucination, etc.
├── judges/         # LLM judge implementations
│   └── prompts/    # YAML prompt templates
├── datasets/       # Test data loading and generation
├── reports/        # Console, HTML, JSON reporters
├── integrations/   # Pytest plugin, LangChain, CI helpers
└── cli/            # Typer CLI commands
```

## Workflow Documents

- **`PROJECT_WORKFLOW.md`** - Full implementation plan with task breakdown
- **`docs/PROJECT_PLAN.md`** - Detailed architecture and milestones
- **`docs/ARCHITECTURE.md`** - Component specifications

## Standards Applied

All commands follow:
- **Python 3.14+** syntax and features
- **Pydantic v2** for data validation
- **Async-first** patterns for LLM operations
- **Pytest** for testing with >80% coverage target
- **Ruff** for linting and formatting
- **Type hints** required on all public APIs
- **Google-style** docstrings

## Example Workflow

```
# 1. Add new evaluator
/new-evaluator
> Name: toxicity
> Measures: Response safety and appropriateness
> Algorithm: Use judge to score toxicity 0-1

# 2. Create tests
/add-test-pattern
> Type: Fixture for toxicity testing scenarios

# 3. Add CLI support
/add-cli-command
> Command: ragaliq check-safety

# 4. Document
/add-docs-example
> Type: Tutorial for safety evaluation
```

## Related Files

- **[WORKFLOW.md](WORKFLOW.md)** - Claude Code development workflow and tips
- **[../PROJECT_WORKFLOW.md](../PROJECT_WORKFLOW.md)** - Full task breakdown with prompts
- **[../docs/PROJECT_PLAN.md](../docs/PROJECT_PLAN.md)** - Architecture and implementation phases

## Contributing

When adding new commands:
1. Follow the template structure in existing commands
2. Include Domain Expertise section with best practices
3. Add Interactive Prompts for gathering requirements
4. Define clear Success Criteria
5. Test the command workflow end-to-end
