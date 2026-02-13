"""Example: Using ContextRecallEvaluator to measure retrieval completeness.

This example demonstrates how to use the ContextRecallEvaluator to verify
that retrieved context contains all the necessary information to answer the query.
"""

import asyncio

from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.context_recall import ContextRecallEvaluator
from ragaliq.judges.claude import ClaudeJudge


async def main() -> None:
    """Run context recall evaluation examples."""

    # Initialize evaluator and judge
    evaluator = ContextRecallEvaluator(threshold=0.8)
    judge = ClaudeJudge(api_key="your-api-key-here")

    # Example 1: Complete recall - all facts covered
    print("=" * 60)
    print("Example 1: Complete Recall")
    print("=" * 60)
    test_case_complete = RAGTestCase(
        id="ex1",
        name="Complete Recall Example",
        query="What is the capital of France and when was the Eiffel Tower built?",
        context=[
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower was built in 1889 for the World's Fair.",
        ],
        response="Paris is the capital of France. The Eiffel Tower was built in 1889.",
        expected_facts=[
            "Paris is the capital of France",
            "The Eiffel Tower was built in 1889",
        ],
    )

    result = await evaluator.evaluate(test_case_complete, judge)
    print(f"Score: {result.score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Tokens used: {result.tokens_used}")
    print()

    # Example 2: Partial recall - missing information
    print("=" * 60)
    print("Example 2: Partial Recall (Missing Information)")
    print("=" * 60)
    test_case_partial = RAGTestCase(
        id="ex2",
        name="Partial Recall Example",
        query="What is the population and area of Paris?",
        context=[
            "Paris has a population of approximately 2.2 million people.",
        ],
        response="Paris has a population of about 2.2 million people.",
        expected_facts=[
            "Paris has a population of about 2.2 million",
            "Paris covers an area of 105 square kilometers",
        ],
    )

    result = await evaluator.evaluate(test_case_partial, judge)
    print(f"Score: {result.score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
    print()

    # Inspect per-fact details
    print("Fact Coverage Details:")
    for i, fact in enumerate(result.raw_response["fact_coverage"]):
        print(f"  Fact {i+1}: {fact['fact']}")
        print(f"    Verdict: {fact['verdict']}")
        print(f"    Evidence: {fact['evidence'][:100]}...")
        print()


if __name__ == "__main__":
    # Note: This example requires ANTHROPIC_API_KEY environment variable
    # or replace "your-api-key-here" with actual API key
    asyncio.run(main())
