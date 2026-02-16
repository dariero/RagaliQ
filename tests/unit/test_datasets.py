"""Unit tests for dataset loading functionality."""

import json
from pathlib import Path

import pytest
import yaml

from ragaliq.core.test_case import RAGTestCase
from ragaliq.datasets import DatasetLoader, DatasetLoadError, DatasetSchema


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def valid_json_dataset(fixtures_dir: Path) -> Path:
    """Return path to valid JSON dataset."""
    return fixtures_dir / "sample_dataset.json"


@pytest.fixture
def valid_yaml_dataset(fixtures_dir: Path) -> Path:
    """Return path to valid YAML dataset."""
    return fixtures_dir / "sample_dataset.yaml"


@pytest.fixture
def valid_csv_dataset(fixtures_dir: Path) -> Path:
    """Return path to valid CSV dataset."""
    return fixtures_dir / "sample_dataset.csv"


class TestDatasetSchema:
    """Test DatasetSchema validation."""

    def test_valid_dataset_schema(self):
        """Test valid dataset schema creation."""
        test_case = RAGTestCase(
            id="test_1",
            name="Test case",
            query="What is Python?",
            context=["Python is a programming language."],
            response="Python is a high-level programming language.",
            tags=["python"],
        )
        schema = DatasetSchema(test_cases=[test_case])
        assert schema.version == "1.0"
        assert len(schema.test_cases) == 1
        assert schema.test_cases[0].id == "test_1"

    def test_duplicate_ids_rejected(self):
        """Test that duplicate test case IDs are rejected."""
        test_case_1 = RAGTestCase(
            id="duplicate",
            name="First",
            query="Query 1",
            context=["Context 1"],
            response="Response 1",
        )
        test_case_2 = RAGTestCase(
            id="duplicate",
            name="Second",
            query="Query 2",
            context=["Context 2"],
            response="Response 2",
        )
        with pytest.raises(ValueError, match="Duplicate test case IDs"):
            DatasetSchema(test_cases=[test_case_1, test_case_2])

    def test_empty_test_cases_rejected(self):
        """Test that empty test cases list is rejected."""
        with pytest.raises(ValueError, match="at least 1 item"):
            DatasetSchema(test_cases=[])

    def test_metadata_optional(self):
        """Test that metadata is optional."""
        test_case = RAGTestCase(
            id="test_1",
            name="Test",
            query="Query",
            context=["Context"],
            response="Response",
        )
        schema = DatasetSchema(test_cases=[test_case])
        assert schema.metadata == {}


class TestDatasetLoaderJSON:
    """Test JSON dataset loading."""

    def test_load_valid_json(self, valid_json_dataset: Path):
        """Test loading a valid JSON dataset."""
        dataset = DatasetLoader.load(valid_json_dataset)
        assert isinstance(dataset, DatasetSchema)
        assert len(dataset.test_cases) == 2
        assert dataset.test_cases[0].id == "test_001"
        assert dataset.test_cases[1].id == "test_002"
        assert dataset.metadata["domain"] == "general_qa"

    def test_load_json_with_all_fields(self, valid_json_dataset: Path):
        """Test that all fields are correctly loaded from JSON."""
        dataset = DatasetLoader.load(valid_json_dataset)
        tc = dataset.test_cases[0]
        assert tc.name == "Python list methods"
        assert tc.query == "How do I add an item to a list in Python?"
        assert len(tc.context) == 2
        assert tc.response.startswith("You can add an item")
        assert tc.expected_answer == "Use the append() method to add items to a list."
        assert len(tc.expected_facts) == 2
        assert tc.tags == ["python", "lists", "basic"]

    def test_load_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON file."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{invalid json}")
        with pytest.raises(DatasetLoadError, match="Failed to parse .json file"):
            DatasetLoader.load(invalid_json)

    def test_load_json_missing_required_field(self, tmp_path: Path):
        """Test loading JSON with missing required field."""
        invalid_data = {
            "version": "1.0",
            "test_cases": [
                {
                    "id": "test_1",
                    "name": "Test",
                    "query": "Query?",
                    # Missing 'context' field
                    "response": "Response",
                }
            ],
        }
        invalid_json = tmp_path / "invalid_schema.json"
        invalid_json.write_text(json.dumps(invalid_data))
        with pytest.raises(DatasetLoadError, match="validation failed"):
            DatasetLoader.load(invalid_json)


class TestDatasetLoaderYAML:
    """Test YAML dataset loading."""

    def test_load_valid_yaml(self, valid_yaml_dataset: Path):
        """Test loading a valid YAML dataset."""
        dataset = DatasetLoader.load(valid_yaml_dataset)
        assert isinstance(dataset, DatasetSchema)
        assert len(dataset.test_cases) == 2
        assert dataset.test_cases[0].id == "test_001"
        assert dataset.test_cases[1].id == "test_002"

    def test_load_yaml_with_yml_extension(self, tmp_path: Path):
        """Test loading YAML file with .yml extension."""
        yml_file = tmp_path / "test.yml"
        data = {
            "version": "1.0",
            "test_cases": [
                {
                    "id": "test_1",
                    "name": "Test",
                    "query": "Query?",
                    "context": ["Context"],
                    "response": "Response",
                }
            ],
        }
        yml_file.write_text(yaml.dump(data))
        dataset = DatasetLoader.load(yml_file)
        assert len(dataset.test_cases) == 1

    def test_load_invalid_yaml(self, tmp_path: Path):
        """Test loading invalid YAML file."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content:")
        with pytest.raises(DatasetLoadError, match="Failed to parse .yaml file"):
            DatasetLoader.load(invalid_yaml)


class TestDatasetLoaderCSV:
    """Test CSV dataset loading."""

    def test_load_valid_csv(self, valid_csv_dataset: Path):
        """Test loading a valid CSV dataset."""
        dataset = DatasetLoader.load(valid_csv_dataset)
        assert isinstance(dataset, DatasetSchema)
        assert len(dataset.test_cases) == 2
        assert dataset.test_cases[0].id == "test_001"
        assert dataset.test_cases[1].id == "test_002"

    def test_load_csv_pipe_separated_lists(self, valid_csv_dataset: Path):
        """Test that pipe-separated lists are parsed correctly."""
        dataset = DatasetLoader.load(valid_csv_dataset)
        tc = dataset.test_cases[0]
        assert len(tc.context) == 2
        assert len(tc.expected_facts) == 2
        assert tc.tags == ["python", "lists", "basic"]

    def test_load_csv_optional_fields(self, valid_csv_dataset: Path):
        """Test CSV row with empty optional fields."""
        dataset = DatasetLoader.load(valid_csv_dataset)
        tc = dataset.test_cases[1]
        # expected_answer is empty in CSV row 2
        assert tc.expected_answer is None

    def test_load_csv_missing_header(self, tmp_path: Path):
        """Test CSV file with no header row."""
        no_header_csv = tmp_path / "no_header.csv"
        no_header_csv.write_text("")
        with pytest.raises(DatasetLoadError, match="has no header row"):
            DatasetLoader.load(no_header_csv)

    def test_load_csv_missing_required_column(self, tmp_path: Path):
        """Test CSV file missing required columns."""
        invalid_csv = tmp_path / "missing_column.csv"
        invalid_csv.write_text("id,name,query\ntest_1,Test,Query?")
        with pytest.raises(
            DatasetLoadError, match="missing required columns:"
        ):
            DatasetLoader.load(invalid_csv)

    def test_load_csv_empty_after_header(self, tmp_path: Path):
        """Test CSV file with only header, no data rows."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("id,name,query,context,response\n")
        with pytest.raises(DatasetLoadError, match="contains no test cases"):
            DatasetLoader.load(empty_csv)

    def test_load_csv_json_array_format(self, tmp_path: Path):
        """Test CSV with JSON array format for list fields."""
        csv_content = """id,name,query,context,response,tags
test_1,Test,Query?,"[""Context 1"", ""Context 2""]",Response,"[""tag1"", ""tag2""]"
"""
        json_csv = tmp_path / "json_format.csv"
        json_csv.write_text(csv_content)
        dataset = DatasetLoader.load(json_csv)
        assert len(dataset.test_cases[0].context) == 2
        assert dataset.test_cases[0].tags == ["tag1", "tag2"]

    def test_load_csv_parse_error(self, tmp_path: Path):
        """Test CSV with invalid JSON in a field."""
        invalid_csv = tmp_path / "parse_error.csv"
        invalid_csv.write_text(
            'id,name,query,context,response\ntest_1,Test,Query?,"[invalid json]",Response'
        )
        with pytest.raises(DatasetLoadError, match="CSV row 2 parse error"):
            DatasetLoader.load(invalid_csv)


class TestDatasetLoaderErrors:
    """Test error handling in dataset loader."""

    def test_file_not_found(self, tmp_path: Path):
        """Test loading non-existent file."""
        missing_file = tmp_path / "missing.json"
        with pytest.raises(DatasetLoadError, match="Dataset file not found"):
            DatasetLoader.load(missing_file)

    def test_file_not_found_shows_path(self, tmp_path: Path):
        """Test that error message includes full path."""
        missing_file = tmp_path / "missing.json"
        with pytest.raises(DatasetLoadError, match="Checked path:"):
            DatasetLoader.load(missing_file)

    def test_unsupported_file_format(self, tmp_path: Path):
        """Test loading file with unsupported extension."""
        unsupported = tmp_path / "data.txt"
        unsupported.write_text("some data")
        with pytest.raises(DatasetLoadError, match="Unsupported file format: .txt"):
            DatasetLoader.load(unsupported)

    def test_validation_error_formatting(self, tmp_path: Path):
        """Test that validation errors are formatted clearly."""
        invalid_data = {
            "version": "1.0",
            "test_cases": [
                {
                    "id": "test_1",
                    "name": "Test",
                    "query": "Query?",
                    "context": ["Context"],
                    "response": "Response",
                    "tags": "invalid_not_a_list",  # Invalid: should be list, not string
                }
            ],
        }
        invalid_json = tmp_path / "validation_error.json"
        invalid_json.write_text(json.dumps(invalid_data))
        with pytest.raises(DatasetLoadError) as exc_info:
            DatasetLoader.load(invalid_json)
        # Check that error message is formatted with bullet points
        assert "•" in str(exc_info.value)


class TestDatasetLoaderPathTypes:
    """Test that loader accepts both string and Path objects."""

    def test_load_with_string_path(self, valid_json_dataset: Path):
        """Test loading with string path."""
        dataset = DatasetLoader.load(str(valid_json_dataset))
        assert len(dataset.test_cases) == 2

    def test_load_with_path_object(self, valid_json_dataset: Path):
        """Test loading with Path object."""
        dataset = DatasetLoader.load(valid_json_dataset)
        assert len(dataset.test_cases) == 2


class TestDatasetLoaderFormatEquivalence:
    """Test that all formats produce equivalent results."""

    def test_json_yaml_csv_equivalence(
        self, valid_json_dataset: Path, valid_yaml_dataset: Path, valid_csv_dataset: Path
    ):
        """Test that JSON, YAML, and CSV produce same test cases."""
        json_dataset = DatasetLoader.load(valid_json_dataset)
        yaml_dataset = DatasetLoader.load(valid_yaml_dataset)
        csv_dataset = DatasetLoader.load(valid_csv_dataset)

        # All should have 2 test cases
        assert len(json_dataset.test_cases) == 2
        assert len(yaml_dataset.test_cases) == 2
        assert len(csv_dataset.test_cases) == 2

        # Compare first test case across formats
        for dataset in [json_dataset, yaml_dataset, csv_dataset]:
            tc = dataset.test_cases[0]
            assert tc.id == "test_001"
            assert tc.name == "Python list methods"
            assert len(tc.context) == 2
            assert len(tc.expected_facts) == 2
            assert tc.tags == ["python", "lists", "basic"]
