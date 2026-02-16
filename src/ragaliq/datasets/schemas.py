"""Dataset schemas for RagaliQ."""

from pydantic import BaseModel, Field, field_validator

from ragaliq.core.test_case import RAGTestCase


class DatasetSchema(BaseModel):
    """
    Schema for a dataset containing multiple RAG test cases.

    Attributes:
        version: Dataset format version (for future compatibility).
        test_cases: List of RAG test cases to evaluate.
        metadata: Optional metadata about the dataset (tags, description, etc.).
    """

    version: str = Field(default="1.0", description="Dataset format version")
    test_cases: list[RAGTestCase] = Field(..., min_length=1, description="List of test cases")
    metadata: dict[str, str] = Field(default_factory=dict, description="Dataset metadata")

    @field_validator("test_cases")
    @classmethod
    def validate_unique_ids(cls, v: list[RAGTestCase]) -> list[RAGTestCase]:
        """Ensure all test case IDs are unique."""
        ids = [tc.id for tc in v]
        if len(ids) != len(set(ids)):
            duplicates = {tc_id for tc_id in ids if ids.count(tc_id) > 1}
            raise ValueError(f"Duplicate test case IDs found: {duplicates}")
        return v

    model_config = {"frozen": False, "extra": "forbid"}
