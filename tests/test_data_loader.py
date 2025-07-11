"""
Test data loading and managing KARA test scenarios.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Scenario:
    """Represents a test scenario with all necessary data."""

    name: str
    description: str
    test_type: str
    parameters: Dict[str, Any]
    tags: List[str]

    # Single document fields
    initial_text: Optional[str] = None
    updated_text: Optional[str] = None

    # Multi-document fields
    initial_documents: Optional[List[str]] = None
    updated_documents: Optional[List[str]] = None

    # Exception handling fields
    expect_failure: bool = False
    expected_results: Optional[Dict[str, Any]] = None
    expected_exception: Optional[str] = None
    expected_exception_message: Optional[str] = None
    exception_on_create: bool = False
    exception_on_update: bool = False

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Scenario":
        """Create Scenario from JSON data."""
        return cls(
            name=data["name"],
            description=data["description"],
            test_type=data["test_type"],
            parameters=data["parameters"],
            expected_results=data["expected_results"],
            tags=data["tags"],
            initial_text=data.get("initial_text"),
            updated_text=data.get("updated_text"),
            initial_documents=data.get("initial_documents"),
            updated_documents=data.get("updated_documents"),
            expected_exception=data.get("expected_exception"),
            expected_exception_message=data.get("expected_exception_message"),
            expect_failure=data.get("expect_failure", False),
        )

    def is_single_document(self) -> bool:
        """Check if this is a single document scenario."""
        return self.initial_text is not None and self.updated_text is not None

    def is_multi_document(self) -> bool:
        """Check if this is a multi-document scenario."""
        return self.initial_documents is not None and self.updated_documents is not None

    def expects_exception(self) -> bool:
        """Check if this scenario expects an exception to be raised."""
        return self.expected_exception is not None or self.expect_failure


class DataLoader:
    """Utility class for loading test scenarios."""

    def __init__(self, test_data_dir: Optional[Union[str, Path]] = None):
        """Initialize with test data directory."""
        if test_data_dir is None:
            # Default to test_data directory relative to this file
            self.test_data_dir = Path(__file__).parent / "test_data"
        else:
            self.test_data_dir = Path(test_data_dir)

        self.scenarios_dir = self.test_data_dir / "scenarios"

    def load_scenario(self, scenario_name: str) -> Scenario:
        """Load a specific test scenario by name."""
        scenario_file = self.scenarios_dir / f"{scenario_name}.json"
        if not scenario_file.exists():
            raise FileNotFoundError(f"Test scenario '{scenario_name}' not found at {scenario_file}")

        with open(scenario_file, encoding="utf-8") as f:
            data = json.load(f)

        return Scenario.from_json(data)

    def load_all_scenarios(self) -> List[Scenario]:
        """Load all available test scenarios."""
        scenarios: List[Scenario] = []

        if not self.scenarios_dir.exists():
            return scenarios

        for scenario_file in self.scenarios_dir.glob("*.json"):
            with open(scenario_file, encoding="utf-8") as f:
                data = json.load(f)
            scenarios.append(Scenario.from_json(data))

        return scenarios

    def load_scenarios_by_tag(self, tag: str) -> List[Scenario]:
        """Load scenarios that have a specific tag."""
        all_scenarios = self.load_all_scenarios()
        return [scenario for scenario in all_scenarios if tag in scenario.tags]

    def load_scenarios_by_type(self, test_type: str) -> List[Scenario]:
        """Load scenarios of a specific test type."""
        all_scenarios = self.load_all_scenarios()
        return [scenario for scenario in all_scenarios if scenario.test_type == test_type]

    def list_available_scenarios(self) -> List[str]:
        """List names of all available scenarios."""
        if not self.scenarios_dir.exists():
            return []

        return [f.stem for f in self.scenarios_dir.glob("*.json")]

    def list_available_tags(self) -> List[str]:
        """List all unique tags from available scenarios."""
        all_scenarios = self.load_all_scenarios()
        all_tags = set()
        for scenario in all_scenarios:
            all_tags.update(scenario.tags)
        return sorted(all_tags)

    def list_available_types(self) -> List[str]:
        """List all unique test types from available scenarios."""
        all_scenarios = self.load_all_scenarios()
        return sorted({scenario.test_type for scenario in all_scenarios})


def get_test_data_loader() -> DataLoader:
    """Get a test data loader instance."""
    return DataLoader()
