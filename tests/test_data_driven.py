"""
Data-driven tests for KARA core functionality using test scenarios.
"""

import pytest
from test_data_loader import DataLoader, Scenario

from kara.core import KARAUpdater, UpdateResult
from kara.splitters import RecursiveCharacterChunker


class TestKARADataDriven:
    """Data-driven tests using predefined test scenarios."""

    @pytest.fixture
    def test_data_loader(self) -> DataLoader:
        """Provide test data loader."""
        return DataLoader()

    def _create_updater_from_scenario(self, scenario: Scenario) -> KARAUpdater:
        """Create a KARAUpdater from scenario parameters."""
        chunker = RecursiveCharacterChunker(
            chunk_size=scenario.parameters["chunk_size"],
            separators=scenario.parameters["separators"],
            keep_separator=scenario.parameters["keep_separator"],
        )
        return KARAUpdater(chunker=chunker, epsilon=scenario.parameters["epsilon"])

    def _run_scenario_with_exception_handling(self, scenario: Scenario) -> None:
        """Run a scenario and handle expected exceptions."""
        if scenario.expects_exception():
            # Handle scenarios that expect exceptions
            if scenario.expected_exception:
                # Get the exception class from string name
                exception_class = getattr(__builtins__, scenario.expected_exception, Exception)
                if hasattr(Exception, scenario.expected_exception):
                    import sys

                    exception_class = getattr(sys.modules["builtins"], scenario.expected_exception)

                with pytest.raises(exception_class) as exc_info:
                    self._execute_scenario_logic(scenario)

                # Check exception message if specified
                if scenario.expected_exception_message:
                    assert scenario.expected_exception_message in str(exc_info.value), (
                        f"Expected exception message '{scenario.expected_exception_message}' "
                        f"not found in '{str(exc_info.value)}'"
                    )
            else:
                # Generic failure expectation - use a more specific exception
                with pytest.raises((ValueError, TypeError, RuntimeError)):
                    self._execute_scenario_logic(scenario)
        else:
            # Normal scenario execution
            self._execute_scenario_logic(scenario)

    def _execute_scenario_logic(self, scenario: Scenario) -> None:
        """Execute the core logic of a scenario."""
        updater = self._create_updater_from_scenario(scenario)

        if scenario.is_single_document():
            # Create initial knowledge base
            assert scenario.initial_text is not None
            assert scenario.updated_text is not None
            initial_result = updater.create_knowledge_base([scenario.initial_text])
            initial_kb = initial_result.new_chunked_doc
            assert initial_kb is not None

            # Update with new text
            update_result = updater.update_knowledge_base(initial_kb, [scenario.updated_text])
            assert update_result.new_chunked_doc is not None

            # Validate expected results
            if scenario.expected_results is not None:
                self._validate_results(update_result, scenario.expected_results, scenario.name)

        elif scenario.is_multi_document():
            # Create initial knowledge base
            assert scenario.initial_documents is not None
            assert scenario.updated_documents is not None
            initial_result = updater.create_knowledge_base(scenario.initial_documents)
            initial_kb = initial_result.new_chunked_doc
            assert initial_kb is not None

            # Update with new documents
            update_result = updater.update_knowledge_base(initial_kb, scenario.updated_documents)
            assert update_result.new_chunked_doc is not None

            # Validate expected results
            if scenario.expected_results is not None:
                self._validate_results(update_result, scenario.expected_results, scenario.name)

    def _validate_results(
        self, update_result: UpdateResult, expected: dict, scenario_name: str
    ) -> None:
        """Validate update results against expected outcomes."""
        # Check minimum reused chunks
        if "min_reused_chunks" in expected:
            min_reused = expected["min_reused_chunks"]
            actual_reused = update_result.num_reused
            assert actual_reused >= min_reused, (
                f"Scenario {scenario_name}: Expected at least {min_reused} "
                f"reused chunks, got {actual_reused}"
            )

        # Check efficiency ratio
        if "efficiency_ratio_threshold" in expected:
            min_ratio = expected["efficiency_ratio_threshold"]
            actual_ratio = update_result.efficiency_ratio
            assert actual_ratio >= min_ratio, (
                f"Scenario {scenario_name}: Expected efficiency ratio >= {min_ratio}, "
                f"got {actual_ratio}"
            )

        # Check total chunks range
        if "total_chunks_range" in expected:
            assert update_result.new_chunked_doc is not None
            total_chunks = len(update_result.new_chunked_doc.chunks)
            min_chunks, max_chunks = expected["total_chunks_range"]
            assert min_chunks <= total_chunks <= max_chunks, (
                f"Scenario {scenario_name}: Expected {min_chunks}-{max_chunks} "
                f"total chunks, got {total_chunks}"
            )

        # Check that new chunks were created when expected
        if expected.get("new_chunks_expected", False):
            assert update_result.num_added > 0, (
                f"Scenario {scenario_name}: Expected new chunks to be created"
            )

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "simple_addition",
            "middle_insertion",
            "complete_replacement",
            "high_epsilon_small_change",
            "low_epsilon_small_change",
            "sentence_separators",
            "paragraph_separators",
            "wikipedia_style",
        ],
    )
    def test_single_document_scenarios(
        self, test_data_loader: DataLoader, scenario_name: str
    ) -> None:
        """Test single document scenarios."""
        scenario = test_data_loader.load_scenario(scenario_name)
        assert scenario.is_single_document(), (
            f"Scenario {scenario_name} is not a single document scenario"
        )

        # Run scenario with exception handling
        self._run_scenario_with_exception_handling(scenario)

    @pytest.mark.parametrize(
        "scenario_name", ["multi_doc_one_changed", "multi_doc_all_changed", "multi_doc_removal"]
    )
    def test_multi_document_scenarios(
        self, test_data_loader: DataLoader, scenario_name: str
    ) -> None:
        """Test multi-document scenarios."""
        scenario = test_data_loader.load_scenario(scenario_name)
        assert scenario.is_multi_document(), (
            f"Scenario {scenario_name} is not a multi-document scenario"
        )

        # Run scenario with exception handling
        self._run_scenario_with_exception_handling(scenario)

    @pytest.mark.parametrize(
        "scenario_name", ["empty_document", "very_large_chunks", "very_small_chunks"]
    )
    def test_edge_case_scenarios(self, test_data_loader: DataLoader, scenario_name: str) -> None:
        """Test edge case scenarios."""
        scenario = test_data_loader.load_scenario(scenario_name)

        # Run scenario with exception handling
        self._run_scenario_with_exception_handling(scenario)

    def test_epsilon_effect_comprehensive(self, test_data_loader: DataLoader) -> None:
        """Test epsilon effect using multiple scenarios."""
        epsilon_scenarios = test_data_loader.load_scenarios_by_tag("epsilon")

        for scenario in epsilon_scenarios:
            # Create KARA updater
            updater = self._create_updater_from_scenario(scenario)

            # Run the scenario
            assert scenario.initial_text is not None
            assert scenario.updated_text is not None
            initial_result = updater.create_knowledge_base([scenario.initial_text])
            assert initial_result.new_chunked_doc is not None
            update_result = updater.update_knowledge_base(
                initial_result.new_chunked_doc, [scenario.updated_text]
            )

            # Validate that epsilon behavior matches expectations
            expected = scenario.expected_results
            if expected is not None and "efficiency_ratio_threshold" in expected:
                min_ratio = expected["efficiency_ratio_threshold"]
                actual_ratio = update_result.efficiency_ratio
                assert actual_ratio >= min_ratio, (
                    f"Epsilon scenario {scenario.name}: Expected ratio >= {min_ratio}, "
                    f"got {actual_ratio}"
                )

    def test_all_scenarios_run_successfully(self, test_data_loader: DataLoader) -> None:
        """Ensure all test scenarios can be loaded and run without errors."""
        all_scenarios = test_data_loader.load_all_scenarios()
        assert len(all_scenarios) > 0, "No test scenarios found"

        for scenario in all_scenarios:
            # Basic validation that scenario is well-formed
            assert scenario.name
            assert scenario.description
            assert scenario.test_type
            assert scenario.parameters
            # Exception scenarios may have empty expected_results
            if not scenario.expects_exception():
                assert scenario.expected_results
            assert scenario.tags

            # Verify scenario has the right content based on type
            single_doc_types = ["single_document", "edge_case", "realistic"]
            if scenario.test_type in single_doc_types:
                assert scenario.is_single_document() or scenario.is_multi_document()
            elif scenario.test_type == "multi_document":
                assert scenario.is_multi_document()

    def test_scenario_tags_functionality(self, test_data_loader: DataLoader) -> None:
        """Test that scenario filtering by tags works correctly."""
        # Test basic tags
        basic_scenarios = test_data_loader.load_scenarios_by_tag("basic")
        assert len(basic_scenarios) > 0

        # Test epsilon tags
        epsilon_scenarios = test_data_loader.load_scenarios_by_tag("epsilon")
        assert len(epsilon_scenarios) > 0

        # Test single_doc tags
        single_doc_scenarios = test_data_loader.load_scenarios_by_tag("single_doc")
        assert len(single_doc_scenarios) > 0

        # Verify all scenarios with single_doc tag are actually single document
        for scenario in single_doc_scenarios:
            if scenario.test_type != "multi_document":
                assert scenario.is_single_document()

    def test_scenario_types_functionality(self, test_data_loader: DataLoader) -> None:
        """Test that scenario filtering by type works correctly."""
        # Test single document scenarios
        single_doc_scenarios = test_data_loader.load_scenarios_by_type("single_document")
        for scenario in single_doc_scenarios:
            assert scenario.is_single_document()

        # Test multi-document scenarios
        multi_doc_scenarios = test_data_loader.load_scenarios_by_type("multi_document")
        for scenario in multi_doc_scenarios:
            assert scenario.is_multi_document()

        # Test edge case scenarios
        edge_case_scenarios = test_data_loader.load_scenarios_by_type("edge_case")
        for scenario in edge_case_scenarios:
            # Edge cases can be either single or multi-document
            assert scenario.is_single_document() or scenario.is_multi_document()

    @pytest.mark.parametrize("scenario_name", ["invalid_parameters"])
    def test_exception_scenarios(self, test_data_loader: DataLoader, scenario_name: str) -> None:
        """Test scenarios that are expected to raise exceptions."""
        scenario = test_data_loader.load_scenario(scenario_name)
        assert scenario.expects_exception(), f"Scenario {scenario_name} should expect an exception"

        # This scenario should raise an exception
        self._run_scenario_with_exception_handling(scenario)
