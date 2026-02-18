"""Dataset loader for JSON, YAML, and CSV formats."""

import csv
import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from ragaliq.core.test_case import RAGTestCase
from ragaliq.datasets.schemas import DatasetSchema


class DatasetLoadError(Exception):
    """Base exception for dataset loading errors."""

    pass


class DatasetLoader:
    """
    Auto-detecting dataset loader supporting JSON, YAML, and CSV formats.

    Detects format from file extension and loads test cases into DatasetSchema.
    """

    @staticmethod
    def load(path: str | Path) -> DatasetSchema:
        """
        Load a dataset from file, auto-detecting format.

        Args:
            path: Path to dataset file (.json, .yaml, .yml, or .csv)

        Returns:
            Validated DatasetSchema with test cases.

        Raises:
            DatasetLoadError: If file not found, format invalid, or validation fails.

        Examples:
            >>> loader = DatasetLoader()
            >>> dataset = loader.load("tests/fixtures/sample_dataset.json")
            >>> len(dataset.test_cases)
            2
        """
        file_path = Path(path)

        # Check file exists
        if not file_path.exists():
            raise DatasetLoadError(
                f"Dataset file not found: {file_path}\nChecked path: {file_path.absolute()}"
            )

        # Auto-detect format
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".json":
                return DatasetLoader._load_json(file_path)
            elif suffix in {".yaml", ".yml"}:
                return DatasetLoader._load_yaml(file_path)
            elif suffix == ".csv":
                return DatasetLoader._load_csv(file_path)
            else:
                raise DatasetLoadError(
                    f"Unsupported file format: {suffix}\n"
                    f"Supported formats: .json, .yaml, .yml, .csv"
                )
        except ValidationError as e:
            raise DatasetLoadError(
                f"Dataset validation failed for {file_path.name}:\n"
                f"{DatasetLoader._format_validation_error(e)}"
            ) from e
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise DatasetLoadError(f"Failed to parse {suffix} file {file_path.name}: {e}") from e

    @staticmethod
    def _load_json(path: Path) -> DatasetSchema:
        """Load dataset from JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return DatasetSchema.model_validate(data)

    @staticmethod
    def _load_yaml(path: Path) -> DatasetSchema:
        """Load dataset from YAML file."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return DatasetSchema.model_validate(data)

    @staticmethod
    def _load_csv(path: Path) -> DatasetSchema:
        """
        Load dataset from CSV file.

        Supports two formats:
        1. Pipe-separated lists for context and expected_facts columns
        2. JSON arrays in string form for complex fields

        Required columns: id, name, query, context, response
        Optional columns: expected_answer, expected_facts, tags
        """
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DatasetLoadError(f"CSV file {path.name} has no header row")

            # Validate required columns
            required = {"id", "name", "query", "context", "response"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise DatasetLoadError(
                    f"CSV file {path.name} missing required columns: {missing}\n"
                    f"Required: {required}"
                )

            test_cases = []
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header=1)
                try:
                    test_case = DatasetLoader._parse_csv_row(row)
                    test_cases.append(test_case)
                except (ValueError, json.JSONDecodeError) as e:
                    raise DatasetLoadError(
                        f"CSV row {row_num} parse error in {path.name}: {e}"
                    ) from e

        if not test_cases:
            raise DatasetLoadError(f"CSV file {path.name} contains no test cases (only header row)")

        return DatasetSchema(test_cases=test_cases)

    @staticmethod
    def _parse_csv_row(row: dict[str, str]) -> RAGTestCase:
        """
        Parse a CSV row into a RAGTestCase.

        Handles pipe-separated lists and JSON arrays.
        """

        def parse_list_field(value: str) -> list[str]:
            """Parse pipe-separated or JSON array field."""
            if not value or value.strip() == "":
                return []
            value = value.strip()
            # Try JSON array first
            if value.startswith("["):
                return cast(list[str], json.loads(value))
            # Fallback to pipe-separated
            return [item.strip() for item in value.split("|") if item.strip()]

        return RAGTestCase(
            id=row["id"].strip(),
            name=row["name"].strip(),
            query=row["query"].strip(),
            context=parse_list_field(row["context"]),
            response=row["response"].strip(),
            expected_answer=row.get("expected_answer", "").strip() or None,
            expected_facts=parse_list_field(row.get("expected_facts", "")),
            tags=parse_list_field(row.get("tags", "")),
        )

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        """Format Pydantic validation errors for user-friendly output."""
        lines = []
        for err in error.errors():
            field = " -> ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            lines.append(f"  • {field}: {msg}")
        return "\n".join(lines)
