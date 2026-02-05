"""Tests for prompt template loading and formatting."""

import pytest

from ragaliq.judges.prompts import (
    PromptExample,
    PromptTemplate,
    get_prompt,
    list_prompts,
)


class TestListPrompts:
    """Tests for listing available prompt templates."""

    def test_list_prompts_returns_list(self) -> None:
        """list_prompts should return a list of template names."""
        prompts = list_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_list_prompts_contains_expected_templates(self) -> None:
        """list_prompts should include core template names."""
        prompts = list_prompts()
        expected = ["faithfulness", "relevance", "extract_claims", "verify_claim"]
        for name in expected:
            assert name in prompts, f"Missing expected template: {name}"

    def test_list_prompts_sorted(self) -> None:
        """list_prompts should return sorted list."""
        prompts = list_prompts()
        assert prompts == sorted(prompts)


class TestGetPrompt:
    """Tests for loading individual prompt templates."""

    def test_get_prompt_returns_template(self) -> None:
        """get_prompt should return a PromptTemplate object."""
        template = get_prompt("faithfulness")
        assert isinstance(template, PromptTemplate)

    def test_get_prompt_nonexistent_raises(self) -> None:
        """get_prompt should raise FileNotFoundError for missing templates."""
        with pytest.raises(FileNotFoundError, match="not found"):
            get_prompt("nonexistent_template")

    def test_get_prompt_has_required_fields(self) -> None:
        """Templates should have all required fields."""
        for name in list_prompts():
            template = get_prompt(name)
            assert template.name, f"{name}: missing name"
            assert template.system_prompt, f"{name}: missing system_prompt"
            assert template.user_template, f"{name}: missing user_template"

    def test_get_prompt_caching(self) -> None:
        """get_prompt should cache loaded templates."""
        # Load twice - second call should use cache
        template1 = get_prompt("faithfulness")
        template2 = get_prompt("faithfulness")
        # Both should have identical content (cache hit)
        assert template1.name == template2.name
        assert template1.system_prompt == template2.system_prompt


class TestFaithfulnessTemplate:
    """Tests for the faithfulness prompt template."""

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Load the faithfulness template."""
        return get_prompt("faithfulness")

    def test_name(self, template: PromptTemplate) -> None:
        """Template should have correct name."""
        assert template.name == "faithfulness"

    def test_version(self, template: PromptTemplate) -> None:
        """Template should have version."""
        assert template.version == "1.0"

    def test_system_prompt_content(self, template: PromptTemplate) -> None:
        """System prompt should contain faithfulness criteria."""
        assert "faithful" in template.system_prompt.lower()
        assert "context" in template.system_prompt.lower()
        assert "JSON" in template.system_prompt

    def test_user_template_placeholders(self, template: PromptTemplate) -> None:
        """User template should have required placeholders."""
        assert "{context}" in template.user_template
        assert "{response}" in template.user_template

    def test_format_user_prompt(self, template: PromptTemplate) -> None:
        """format_user_prompt should substitute placeholders."""
        prompt = template.format_user_prompt(
            context="Test context",
            response="Test response",
        )
        assert "Test context" in prompt
        assert "Test response" in prompt
        assert "{context}" not in prompt
        assert "{response}" not in prompt

    def test_format_user_prompt_missing_var_raises(self, template: PromptTemplate) -> None:
        """format_user_prompt should raise KeyError for missing variables."""
        with pytest.raises(KeyError):
            template.format_user_prompt(context="Only context")

    def test_has_examples(self, template: PromptTemplate) -> None:
        """Template should have few-shot examples."""
        assert len(template.examples) > 0

    def test_examples_structure(self, template: PromptTemplate) -> None:
        """Examples should have input and output fields."""
        for example in template.examples:
            assert isinstance(example, PromptExample)
            assert "context" in example.input or "response" in example.input
            assert "score" in example.output
            assert "reasoning" in example.output

    def test_output_format(self, template: PromptTemplate) -> None:
        """Template should specify output format."""
        assert template.output_format is not None
        assert template.output_format.type == "json"


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
        assert "claims" in template.output_format.schema_

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
        assert "verdict" in template.output_format.schema_

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
        return get_prompt("faithfulness")

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
        return get_prompt("faithfulness")

    def test_get_examples_text_includes_all(self, template: PromptTemplate) -> None:
        """get_examples_text should include all examples by default."""
        text = template.get_examples_text()
        assert "Example 1:" in text
        assert "Example 2:" in text

    def test_get_examples_text_limit(self, template: PromptTemplate) -> None:
        """get_examples_text should respect max_examples limit."""
        text = template.get_examples_text(max_examples=1)
        assert "Example 1:" in text
        assert "Example 2:" not in text

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
