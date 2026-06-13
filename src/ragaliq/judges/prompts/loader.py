"""
Prompt template loader for RagaliQ judges.

This module provides functionality to load and manage YAML-based prompt templates
for LLM judge operations. Templates include system prompts, user templates,
output format specifications, and few-shot examples.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PromptExample(BaseModel):
    """A few-shot example for a prompt template."""

    input: dict[str, Any] = Field(..., description="Example input variables")
    output: dict[str, Any] = Field(..., description="Expected output")

    model_config = {"extra": "forbid"}


class PromptTemplate(BaseModel):
    """A judge prompt template: system prompt, user template, output format, examples."""

    name: str = Field(..., description="Template identifier")
    version: str = Field(default="1.0", description="Template version")
    description: str = Field(default="", description="Template description")
    system_prompt: str = Field(..., description="System prompt for the LLM")
    user_template: str = Field(..., description="User message template with placeholders")
    output_format: dict[str, Any] | None = Field(default=None, description="Expected output format")
    examples: list[PromptExample] = Field(default_factory=list, description="Few-shot examples")

    model_config = {"extra": "forbid"}

    def format_user_prompt(self, **kwargs: Any) -> str:
        """Format the user template, escaping braces in values to block format-string injection.

        Escaping prevents constructs like ``{__class__}`` in user content from
        being interpreted by ``str.format``.

        Raises:
            KeyError: If a required template variable is missing.
        """
        sanitized = {
            k: v.replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v
            for k, v in kwargs.items()
        }
        return self.user_template.format(**sanitized)

    def format_context(self, context: list[str]) -> str:
        """Join context documents into one string with numbered `Document N:` separators."""
        return "\n\n---\n\n".join(f"Document {i + 1}:\n{doc}" for i, doc in enumerate(context))

    def get_examples_text(self, max_examples: int | None = None) -> str:
        """Render up to `max_examples` few-shot examples as prompt text ("" if none)."""
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

    def build_system_prompt(self, max_examples: int | None = None) -> str:
        """Return the system prompt with a few-shot "Examples:" section appended when present.

        Falls back to the bare ``system_prompt`` when the template has no examples.

        Args:
            max_examples: Cap on examples to include (all if None).
        """
        examples_text = self.get_examples_text(max_examples)
        if not examples_text:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{examples_text}"


def _get_prompts_dir() -> Path:
    """Get the directory containing prompt template YAML files."""
    return Path(__file__).parent


@lru_cache(maxsize=32)
def _load_template_file(name: str) -> dict[str, Any]:
    """Load and parse `<name>.yaml` from the prompts directory (cached).

    Raises:
        FileNotFoundError: If the template file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    prompts_dir = _get_prompts_dir()
    file_path = prompts_dir / f"{name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")

    try:
        with file_path.open() as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse prompt template '{name}': {e}") from e

    if not isinstance(content, dict):
        raise ValueError(
            f"Prompt template '{name}' must be a YAML mapping, got {type(content).__name__}"
        )

    return content


def get_prompt(name: str) -> PromptTemplate:
    """Load and validate the prompt template named `name` (e.g. 'faithfulness').

    Raises:
        FileNotFoundError: If the template doesn't exist.
        ValidationError: If the template structure is invalid.
    """
    data = _load_template_file(name)
    return PromptTemplate.model_validate(data)


def list_prompts() -> list[str]:
    """Return the sorted names of all available prompt templates (without `.yaml`)."""
    prompts_dir = _get_prompts_dir()
    return sorted(f.stem for f in prompts_dir.glob("*.yaml") if f.is_file())
