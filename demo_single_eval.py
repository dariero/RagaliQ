"""Standalone learning script: run one RAGTestCase through RagaliQ.

Run it two ways:

    .venv/bin/python demo_single_eval.py            # clean (faithful) answer
    .venv/bin/python demo_single_eval.py --corrupt  # answer with 1 unsupported claim

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import sys

from ragaliq import RagaliQ, RAGTestCase

# --- The shared "retrieval" side of a RAG test case --------------------------
# In a real RAG system, `context` is whatever your retriever pulled from the
# vector store for this question. Here we hand-write it so we control exactly
# what the answer is *allowed* to be grounded in.
QUERY = "What is the capital of France?"
CONTEXT = [
    "France is a country in Western Europe.",
    "The capital city of France is Paris, home to the Eiffel Tower.",
]

# --- Two versions of the generated answer -----------------------------------
# CLEAN: every statement is backed by the context above.
CLEAN_RESPONSE = "The capital of France is Paris."

# CORRUPT: same correct first claim, PLUS one on-topic but UNSUPPORTED claim.
# The population figure appears nowhere in CONTEXT, so the judge cannot ground it.
CORRUPT_RESPONSE = (
    "The capital of France is Paris, and Paris has a population of 30 million people."
)


def build_test_case(corrupt: bool) -> RAGTestCase:
    """Construct the RAGTestCase, choosing the clean or corrupted answer."""
    return RAGTestCase(
        id="demo-france-1",  # unique id (like a test name)
        name="Capital of France" + (" (corrupted)" if corrupt else ""),
        query=QUERY,  # the user's question
        context=CONTEXT,  # retrieved docs the answer may use
        response=CORRUPT_RESPONSE if corrupt else CLEAN_RESPONSE,  # the answer under test
    )


def main() -> None:
    corrupt = "--corrupt" in sys.argv

    test_case = build_test_case(corrupt)

    # RagaliQ() with defaults = Claude judge, evaluators [faithfulness, relevance],
    # passing threshold 0.7. .evaluate() runs synchronously (wraps asyncio internally).
    tester = RagaliQ(judge="claude")

    print(f"\n=== Variant: {'CORRUPTED' if corrupt else 'CLEAN'} ===")
    print(f"Question : {test_case.query}")
    print(f"Answer   : {test_case.response}")
    print(f"Context  : {test_case.context}")

    result = tester.evaluate(test_case)

    print(f"\nStatus       : {result.status}  (passed={result.passed})")
    print(f"Tokens used  : {result.judge_tokens_used}")
    print(f"Time         : {result.execution_time_ms} ms")

    print("\nScores (threshold 0.70):")
    for metric, score in result.scores.items():
        mark = "PASS" if score >= 0.70 else "FAIL"
        print(f"  [{mark}] {metric:<13} {score:.2f}")

    # Faithfulness exposes its per-claim breakdown in the 'raw' detail —
    # this is what makes the score explainable rather than a black box.
    faith = result.details["faithfulness"]
    print("\nFaithfulness reasoning:")
    print(f"  {faith['reasoning']}")
    claims = faith["raw"].get("claims", [])
    if claims:
        print("  Per-claim verdicts:")
        for c in claims:
            print(f"    - [{c['verdict']}] {c['claim']}")
            if c.get("evidence"):
                print(f"        evidence: {c['evidence']}")

    print("\nRelevance reasoning:")
    print(f"  {result.details['relevance']['reasoning']}")
    print()


if __name__ == "__main__":
    main()
