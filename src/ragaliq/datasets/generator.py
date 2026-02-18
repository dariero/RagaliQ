"""Test case generator for RagaliQ datasets.

This module provides TestCaseGenerator, which synthesizes RAGTestCase objects
from raw documents by using an LLM judge to generate questions and answers.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from ragaliq.core.test_case import RAGTestCase

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge


class TestCaseGenerator:
    """
    Generates RAGTestCase objects from documents using an LLM judge.

    The generator uses the judge to produce questions grounded in the provided
    documents, then generates reference answers for those questions. This
    creates a synthetic test dataset without requiring manual annotation.

    The judge is injected via method parameter (consistent with the Evaluator
    pattern) enabling flexible use across different judge implementations.

    Example:
        generator = TestCaseGenerator()
        judge = ClaudeJudge(api_key="...")
        test_cases = await generator.generate_from_documents(
            documents=["Python lists support append(), extend()..."],
            n=5,
            judge=judge,
        )
    """

    async def generate_from_documents(
        self,
        documents: list[str],
        n: int,
        judge: LLMJudge,
    ) -> list[RAGTestCase]:
        """
        Generate n test cases from the provided documents.

        Generates questions grounded in the documents, then produces
        corresponding answers using only the document content. Answer
        generation runs in parallel for all questions.

        Args:
            documents: Source documents to generate test cases from.
            n: Number of test cases to generate.
            judge: LLM judge instance used for generation.

        Returns:
            List of RAGTestCase objects (may be fewer than n if the judge
            returns fewer questions than requested).

        Raises:
            ValueError: If documents is empty or n is less than 1.
        """
        if not documents:
            raise ValueError("documents must not be empty")
        if n < 1:
            raise ValueError("n must be at least 1")

        # Step 1: Generate n questions from all documents
        questions_result = await judge.generate_questions(documents, n)
        questions = questions_result.questions[:n]  # Trim if LLM over-generated

        # Step 2: Generate an answer for each question in parallel
        answer_tasks = [judge.generate_answer(question=q, context=documents) for q in questions]
        answer_results = await asyncio.gather(*answer_tasks)

        # Step 3: Assemble RAGTestCase objects
        return [
            RAGTestCase(
                id=str(uuid.uuid4()),
                name=_derive_name(question, i),
                query=question,
                context=documents,
                response=answer_result.answer,
                tags=["generated"],
            )
            for i, (question, answer_result) in enumerate(zip(questions, answer_results, strict=True), start=1)
        ]

    def generate_from_documents_sync(
        self,
        documents: list[str],
        n: int,
        judge: LLMJudge,
    ) -> list[RAGTestCase]:
        """
        Synchronous wrapper for generate_from_documents.

        Args:
            documents: Source documents to generate test cases from.
            n: Number of test cases to generate.
            judge: LLM judge instance used for generation.

        Returns:
            List of RAGTestCase objects.
        """
        return asyncio.run(self.generate_from_documents(documents=documents, n=n, judge=judge))


def _derive_name(question: str, index: int) -> str:
    """
    Derive a short display name from a question string.

    Args:
        question: The question text.
        index: Sequential index for the name prefix.

    Returns:
        A short name string in the form "<index>. <truncated question>".
    """
    max_length = 60
    name = question.rstrip("?").strip()
    if len(name) > max_length:
        truncated = name[:max_length].rsplit(" ", 1)[0]
        name = truncated + "..."
    return f"{index}. {name}"
