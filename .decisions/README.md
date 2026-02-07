# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the RagaliQ project. ADRs document significant design decisions, trade-offs, and the reasoning behind architectural choices.

## Active ADRs

| ADR | Title | Status | Date | Issue |
|-----|-------|--------|------|-------|
| [ADR-006](./ADR-006-faithfulness-evaluator.md) | FaithfulnessEvaluator with Claim-Level Decomposition | Implemented | 2026-02-05 | [#6](https://github.com/dariero/RagaliQ/issues/6) |
| [ADR-007](./ADR-007-relevance-evaluator.md) | RelevanceEvaluator as Thin Adapter over Judge | Implemented | 2026-02-06 | [#7](https://github.com/dariero/RagaliQ/issues/7) |
| [ADR-008](./ADR-008-hallucination-evaluator.md) | HallucinationEvaluator Implementation | Implemented | 2026-02-07 | [#8](https://github.com/dariero/RagaliQ/issues/8) |

## ADR Format

All ADRs follow this structure:

```markdown
# ADR-NNN: [Title]

**Status:** [Proposed | Accepted | Implemented | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Issue:** #N — [Issue Title]

## Context

What is the issue we're trying to solve? What constraints exist?

## Proposed Solution

The approach chosen, with architecture diagrams/pseudocode if helpful.

## Principles Applied

Which design principles (SOLID, patterns, etc.) guided this decision?

## Alternatives Considered

What other approaches were evaluated, and why were they rejected or deferred?

## Implementation Details (optional)

Files changed, test coverage, edge cases handled.

## Future Considerations (optional)

Potential enhancements or related work for later.
```

## ADR Numbering

ADR numbers are assigned sequentially and **correspond to GitHub issue numbers** when applicable:
- ADR-006 = Issue #6 (Task 5: FaithfulnessEvaluator)
- ADR-007 = Issue #7 (Task 7: RelevanceEvaluator)
- ADR-008 = Issue #8 (Task 7: HallucinationEvaluator)

For decisions not tied to a specific issue, use the next available number.

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Proposed** | Decision is under discussion, not yet approved |
| **Accepted** | Decision has been approved but not yet implemented |
| **Implemented** | Decision has been coded and merged to main |
| **Deprecated** | Decision is no longer relevant (but kept for historical context) |
| **Superseded** | Decision has been replaced by a newer ADR (link to the replacement) |

## When to Write an ADR

Create an ADR for decisions that:
- Introduce new architectural patterns or components
- Choose between multiple valid implementation approaches
- Modify existing contracts or interfaces
- Have non-obvious trade-offs or consequences
- Future contributors will need to understand

## ADR Workflow

1. **During /start-work**: If the task requires architectural decisions, create `ADR-NNN-title.md` in `.decisions/`
2. **Before implementation**: Document context, proposed solution, principles, and alternatives
3. **After implementation**: Update status to "Implemented" and add implementation details
4. **Update README**: Add the new ADR to the table above

## References

- [Architecture Decision Records (Michael Nygard)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
