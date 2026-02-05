"""
Prompt template loader for RagaliQ judges.

This module provides functionality to load and manage YAML-based prompt templates
for LLM judge operations. Templates include system prompts, user templates,
output format specifications, and few-shot examples.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OutputSchema(BaseModel):
    """Schema definition for a single output field."""

    type: str = Field(..., description="Data type of the field")
    description: str = Field(default="", description="Field description")
    min: float | None = Field(default=None, description="Minimum value for numeric types")
    max: float | None = Field(default=None, description="Maximum value for numeric types")
    enum: list[str] | None = Field(default=None, description="Allowed values for enum types")
    items: dict[str, Any] | None = Field(default=None, description="Schema for array items")

    model_config = {"extra": "allow"}


class OutputFormat(BaseModel):
    """Output format specification for a prompt template."""

    type: str = Field(..., description="Output type (e.g., 'json')")
    schema_: dict[str, OutputSchema] = Field(
        default_factory=dict,
        alias="schema",
        description="Schema definition for output fields",
    )

    model_config = {"extra": "allow", "populate_by_name": True}


class PromptExample(BaseModel):
    """A few-shot example for a prompt template."""

    input: dict[str, Any] = Field(..., description="Example input variables")
    output: dict[str, Any] = Field(..., description="Expected output")

    model_config = {"extra": "allow"}


class PromptTemplate(BaseModel):
    """
    A prompt template for LLM judge operations.

    Prompt templates define the structure of prompts sent to LLM judges,
    including system prompts, user message templates, output format
    specifications, and few-shot examples.

    Attributes:
        name: Unique identifier for the template.
        version: Template version for tracking changes.
        description: Human-readable description of the template's purpose.
        system_prompt: The system message defining the LLM's role.
        user_template: Template string for user messages with {placeholders}.
        output_format: Specification for expected output structure.
        examples: Few-shot examples for the template.
    """

    name: str = Field(..., description="Template identifier")
    version: str = Field(default="1.0", description="Template version")
    description: str = Field(default="", description="Template description")
    system_prompt: str = Field(..., description="System prompt for the LLM")
    user_template: str = Field(..., description="User message template with placeholders")
    output_format: OutputFormat | None = Field(default=None, description="Expected output format")
    examples: list[PromptExample] = Field(default_factory=list, description="Few-shot examples")

    model_config = {"extra": "allow"}

    def format_user_prompt(self, **kwargs: Any) -> str:
        """
        Format the user template with provided variables.

        Args:
            **kwargs: Variables to substitute in the template.

        Returns:
            Formatted user prompt string.

        Raises:
            KeyError: If a required template variable is missing.
        """
        return self.user_template.format(**kwargs)

    def format_context(self, context: list[str]) -> str:
        """
        Format a list of context documents for insertion into prompts.

        Args:
            context: List of context document strings.

        Returns:
            Formatted context string with document separators.
        """
        return "\n\n---\n\n".join(f"Document {i + 1}:\n{doc}" for i, doc in enumerate(context))

    def get_examples_text(self, max_examples: int | None = None) -> str:
        """
        Format few-shot examples as text for inclusion in prompts.

        Args:
            max_examples: Maximum number of examples to include.

        Returns:
            Formatted examples string.
        """
        examples = self.examples[:max_examples] if max_examples else self.examples

        if not examples:
            return ""

        lines = ["Examples:", ""]
        for i, example in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {example.input}")
            lines.append(f"Output: {example.output}")
            lines.append("")

        return "\n".join(lines)


def _get_prompts_dir() -> Path:
    """Get the directory containing prompt template YAML files."""
    return Path(__file__).parent


@lru_cache(maxsize=32)
def _load_template_file(name: str) -> dict[str, Any]:
    """
    Load and parse a YAML template file.

    Args:
        name: Template name (without .yaml extension).

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        FileNotFoundError: If template file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    prompts_dir = _get_prompts_dir()
    file_path = prompts_dir / f"{name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")

    with file_path.open() as f:
        content: dict[str, Any] = yaml.safe_load(f)
        return content


def get_prompt(name: str) -> PromptTemplate:
    """
    Load a prompt template by name.

    Args:
        name: Template name (e.g., 'faithfulness', 'relevance').

    Returns:
        PromptTemplate object with loaded template data.

    Raises:
        FileNotFoundError: If template doesn't exist.
        ValidationError: If template structure is invalid.

    Example:
        >>> template = get_prompt("faithfulness")
        >>> prompt = template.format_user_prompt(
        ...     context="Document 1:\\nParis is in France.",
        ...     response="Paris is the capital of France."
        ... )
    """
    data = _load_template_file(name)
    return PromptTemplate.model_validate(data)


def list_prompts() -> list[str]:
    """
    List all available prompt template names.

    Returns:
        List of template names (without .yaml extension).

    Example:
        >>> list_prompts()
        ['extract_claims', 'faithfulness', 'relevance', 'verify_claim']
    """
    prompts_dir = _get_prompts_dir()
    return sorted(f.stem for f in prompts_dir.glob("*.yaml") if f.is_file())
