"""Tests for prompt template loading and formatting."""

from pathlib import Path

import pytest

import ragaliq.judges.prompts as prompts_pkg
from ragaliq.judges.prompts import (
    PromptTemplate,
    get_prompt,
)

SHIPPED = sorted(f.stem for f in Path(prompts_pkg.__file__).parent.glob("*.yaml"))


class TestShippedTemplates:
    """Tests for the set of prompt templates shipped in the package."""

    def test_shipped_templates(self) -> None:
        """The package ships exactly these five templates."""
        assert SHIPPED == [
            "extract_claims",
            "generate_answer",
            "generate_questions",
            "relevance",
            "verify_claim",
        ]


class TestGetPrompt:
    """Tests for loading individual prompt templates."""

    def test_get_prompt_returns_template(self) -> None:
        """get_prompt should return a PromptTemplate object."""
        template = get_prompt("relevance")
        assert isinstance(template, PromptTemplate)

    def test_get_prompt_nonexistent_raises(self) -> None:
        """get_prompt should raise FileNotFoundError for missing templates."""
        with pytest.raises(FileNotFoundError, match="not found"):
            get_prompt("nonexistent_template")

    def test_get_prompt_has_required_fields(self) -> None:
        """Templates should have all required fields."""
        for name in SHIPPED:
            template = get_prompt(name)
            assert template.name, f"{name}: missing name"
            assert template.system_prompt, f"{name}: missing system_prompt"
            assert template.user_template, f"{name}: missing user_template"

    def test_get_prompt_caching(self) -> None:
        """get_prompt should cache loaded templates."""
        # Load twice - second call should use cache
        template1 = get_prompt("relevance")
        template2 = get_prompt("relevance")
        # Both should have identical content (cache hit)
        assert template1.name == template2.name
        assert template1.system_prompt == template2.system_prompt


class TestRelevanceTemplate:
    """Tests for the relevance prompt template."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load the relevance template."""
        return get_prompt("relevance")

    def test_name(self, template: PromptTemplate) -> None:
        """Template should have correct name."""
        assert template.name == "relevance"

    def test_user_template_placeholders(self, template: PromptTemplate) -> None:
        """User template should have required placeholders."""
        assert "{query}" in template.user_template
        assert "{response}" in template.user_template

    def test_format_user_prompt(self, template: PromptTemplate) -> None:
        """format_user_prompt should substitute placeholders."""
        prompt = template.format_user_prompt(
            query="What is Python?",
            response="Python is a programming language.",
        )
        assert "What is Python?" in prompt
        assert "Python is a programming language." in prompt
        assert "{query}" not in prompt
        assert "{response}" not in prompt

    def test_format_user_prompt_missing_var_raises(self, template: PromptTemplate) -> None:
        """format_user_prompt should raise KeyError for a missing variable."""
        with pytest.raises(KeyError, match="response"):
            template.format_user_prompt(query="Only query")

    def test_has_examples(self, template: PromptTemplate) -> None:
        """Template should have few-shot examples."""
        assert len(template.examples) > 0


class TestExtractClaimsTemplate:
    """Tests for the extract_claims prompt template."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load the extract_claims template."""
        return get_prompt("extract_claims")

    def test_name(self, template: PromptTemplate) -> None:
        """Template should have correct name."""
        assert template.name == "extract_claims"

    def test_user_template_placeholders(self, template: PromptTemplate) -> None:
        """User template should have required placeholders."""
        assert "{response}" in template.user_template

    def test_output_format_claims_array(self, template: PromptTemplate) -> None:
        """Output format should specify claims as array."""
        assert template.output_format is not None
        assert "claims" in template.output_format["schema"]

    def test_has_examples(self, template: PromptTemplate) -> None:
        """Template should have few-shot examples."""
        assert len(template.examples) > 0
        # Examples should show claims extraction
        for example in template.examples:
            assert "claims" in example.output
            assert isinstance(example.output["claims"], list)


class TestVerifyClaimTemplate:
    """Tests for the verify_claim prompt template."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load the verify_claim template."""
        return get_prompt("verify_claim")

    def test_name(self, template: PromptTemplate) -> None:
        """Template should have correct name."""
        assert template.name == "verify_claim"

    def test_user_template_placeholders(self, template: PromptTemplate) -> None:
        """User template should have required placeholders."""
        assert "{claim}" in template.user_template
        assert "{context}" in template.user_template

    def test_output_format_verdict(self, template: PromptTemplate) -> None:
        """Output format should specify verdict enum."""
        assert template.output_format is not None
        assert "verdict" in template.output_format["schema"]

    def test_has_examples(self, template: PromptTemplate) -> None:
        """Template should have few-shot examples."""
        assert len(template.examples) > 0
        # Examples should show verdict values
        for example in template.examples:
            assert "verdict" in example.output
            assert example.output["verdict"] in [
                "SUPPORTED",
                "CONTRADICTED",
                "NOT_ENOUGH_INFO",
            ]


class TestPromptTemplateFormatContext:
    """Tests for context formatting helper."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load any template for testing."""
        return get_prompt("verify_claim")

    def test_format_context_single_doc(self, template: PromptTemplate) -> None:
        """format_context should handle single document."""
        result = template.format_context(["First document content."])
        assert "Document 1:" in result
        assert "First document content." in result

    def test_format_context_multiple_docs(self, template: PromptTemplate) -> None:
        """format_context should number and separate multiple documents."""
        result = template.format_context(
            [
                "First document.",
                "Second document.",
                "Third document.",
            ]
        )
        assert "Document 1:" in result
        assert "Document 2:" in result
        assert "Document 3:" in result
        assert "---" in result

    def test_format_context_empty(self, template: PromptTemplate) -> None:
        """format_context should handle empty list."""
        result = template.format_context([])
        assert result == ""


class TestPromptTemplateGetExamplesText:
    """Tests for examples text formatting."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load template with examples."""
        return get_prompt("relevance")

    def test_get_examples_text_includes_all(self, template: PromptTemplate) -> None:
        """get_examples_text should include every example."""
        text = template.get_examples_text()
        assert "Example 1:" in text
        assert f"Example {len(template.examples)}:" in text

    def test_get_examples_text_empty(self) -> None:
        """get_examples_text should return empty string for no examples."""
        template = PromptTemplate(
            name="test",
            system_prompt="test",
            user_template="test",
            examples=[],
        )
        text = template.get_examples_text()
        assert text == ""
