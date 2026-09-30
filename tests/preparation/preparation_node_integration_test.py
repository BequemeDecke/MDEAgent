"""
Integration test for the prepare_node using create_preparation_node.

This test simulates a real case by invoking the preparation node with actual model files
and checks the resulting state and workspace contents.
It follows a similar structure to mdeagent_test.py.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

import pytest

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation import EvaluationExecutor, implementations
from mdeagent.preparation.graph import build_preparation_graph
from mdeagent.preparation.node import create_preparation_node
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)

TEST_ENVIRONMENT = Path(".mdeagent-tests")
TEST_SETUP_FILES = TEST_ENVIRONMENT / "setup-files"
TEST_EXECUTION_RUNS = TEST_ENVIRONMENT / "test-executions" / "preparation-node"


def create_workspace_folder() -> Path:
    """Create a unique workspace folder for the test execution."""
    workspace = TEST_EXECUTION_RUNS / datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


class TestPreparationNodeIntegration(TestCase):
    """Integration test case for the preparation node.

    This test spins up the actual preparation subgraph and invokes it through
    the create_preparation_node function with real model files.
    """

    @pytest.fixture(autouse=True)
    def _setup_test(self, enable_langfuse):
        """Set up the test environment with workspace and preparation agent."""
        # Create a unique workspace for the test
        self.workspace_path = create_workspace_folder()
        logger.info(f"Created test workspace at {self.workspace_path}")

        self.enable_langfuse = enable_langfuse

        # Check if the setup files exist
        self.source_model_path = TEST_SETUP_FILES / "Families"
        self.target_model_path = TEST_SETUP_FILES / "Persons"
        if not self.source_model_path.exists() or not self.target_model_path.exists():
            pytest.skip(
                f"Setup files not found. Skipping test. Please ensure that {self.source_model_path} "
                f"and {self.target_model_path} exist."
            )

        source_file_count = len(list(self.source_model_path.glob("*.java")))
        target_file_count = len(list(self.target_model_path.glob("*.java")))
        if source_file_count != 4:
            pytest.skip(
                f"Expected 4 source model files in {self.source_model_path}, "
                f"but found {source_file_count}."
            )
        if target_file_count != 3:
            pytest.skip(
                f"Expected 3 target model files in {self.target_model_path}, "
                f"but found {target_file_count}."
            )

        # Build the preparation agent with evaluation executor.
        self.evaluation_executor = EvaluationExecutor(
            evaluations={
                "workspace_structure": {
                    "evaluation": implementations.WorkspaceStructureEvaluation(),
                    "evaluation_schema": implementations.WorkspaceStructureSchema,
                },
                "tools_installed": {
                    "evaluation": implementations.ToolInstalledEvaluation(),
                    "evaluation_schema": implementations.ToolInstalledSchema,
                },
            }
        )
        self.preparation_agent = build_preparation_graph(
            evaluation_executor=self.evaluation_executor,
            max_iterations=10,
        ).compile()

        # Required tools for the preparation
        self.required_tools = ["mvn", "java", "javac", "jar"]

        # Group and artifact IDs for the Maven project
        self.group_id = "de.hofuniversity"
        self.artifact_id = "PreparationNodeIntegrationTest"

    def test_preparation_node__full_workflow(self):
        """Test the preparation node with a full workflow invocation.

        Note: Only ainvoke can be used here, because some nodes are executed
        asynchronously and the test needs to wait for them to finish.
        The test will fail if the workflow is not completed successfully.
        """
        # 1. Create the initial state for the agent
        initial_state = MDEAgentState(
            source_model_path=self.source_model_path,
            target_model_path=self.target_model_path,
            group_id=self.group_id,
            artifact_id=self.artifact_id,
        )

        # 2. Create the preparation node with required parameters
        preparation_node = create_preparation_node(
            preparation_agent=self.preparation_agent,
            workspace_path=self.workspace_path,
            required_tools=self.required_tools,
        )

        # 3. Invoke the preparation node with the initial state
        output = asyncio.run(preparation_node(initial_state))

        # 4. Check the output state for expected values
        self.check_output_state(output)

        # 5. Check the contents of the workspace for expected files
        self.check_workspace_contents()

    def check_output_state(self, output: MDEAgentState):
        """Check the output state for expected values."""
        # Check that workspace_path is set correctly
        self.assertIsNotNone(
            output.get("workspace_path"),
            "Workspace path should not be None.",
        )
        self.assertEqual(
            output["workspace_path"],
            self.workspace_path,
            "Workspace path should match the created workspace path.",
        )

        # Check that required_tools is set correctly
        self.assertIsNotNone(
            output.get("required_tools"),
            "Required tools should not be None.",
        )
        self.assertEqual(
            output["required_tools"],
            self.required_tools,
            "Required tools should match the provided tools.",
        )

        # Check that maven_project_path is set correctly
        self.assertIsNotNone(
            output.get("maven_project_path"),
            "Maven project path should not be None.",
        )

        # Check that the transformation class path is set
        self.assertIsNotNone(
            output.get("transformation_class_path"),
            "Transformation class path should not be None.",
        )
        # The preparation node creates the transformation class file via touch()
        self.assertTrue(
            output["transformation_class_path"].exists(),
            "Transformation class file should exist (created by prepare_workspace).",
        )

        # Check that the bxtool path is set and the file exists
        self.assertIsNotNone(
            output.get("bxtool_path"),
            "BXT tool path should not be None.",
        )
        self.assertTrue(
            output["bxtool_path"].exists(),
            "BXT tool file should exist.",
        )

        # Check that the transformation plan is set
        self.assertIsNotNone(
            output.get("transformation_plan"),
            "Transformation plan should not be None.",
        )
        self.assertIsInstance(
            output["transformation_plan"],
            dict,
            "Transformation plan should be a serialized dictionary.",
        )
        
        # Check that the transformation plan contains model implementations
        tp = TransformationPlan.from_dict(output["transformation_plan"])
        tp_data = tp.data

        self.assertIn(
            "source_model_implementation",
            tp_data,
            "Transformation plan should contain source_model_implementation.",
        )
        self.assertIn(
            "target_model_implementation",
            tp_data,
            "Transformation plan should contain target_model_implementation.",
        )
        self.assertIn(
            "source_model_package",
            tp_data,
            "Transformation plan should contain source_model_package.",
        )
        self.assertIn(
            "target_model_package",
            tp_data,
            "Transformation plan should contain target_model_package.",
        )

        # Verify package names are correct
        self.assertEqual(
            tp_data["source_model_package"],
            "Families",
            "Source model package name should be 'Families'.",
        )
        self.assertEqual(
            tp_data["target_model_package"],
            "Persons",
            "Target model package name should be 'Persons'.",
        )
        self.assertEqual(
            output.get("transformation_package_path"),
            "de.hofuniversity.PreparationNodeIntegrationTest",
            "Transformation package path should be 'de.hofuniversity.PreparationNodeIntegrationTest'.",
        )

        # Verify model implementations contain content from actual files
        source_files_content = [
            f.read_text() for f in self.source_model_path.glob("*.java")
        ]
        target_files_content = [
            f.read_text() for f in self.target_model_path.glob("*.java")
        ]

        for content in source_files_content:
            self.assertIn(
                content,
                tp_data["source_model_implementation"],
                "Transformation plan should contain all source model file contents.",
            )

        for content in target_files_content:
            self.assertIn(
                content,
                tp_data["target_model_implementation"],
                "Transformation plan should contain all target model file contents.",
            )

    def check_workspace_contents(self):
        """Check the contents of the workspace for expected files and structure."""
        # Check that the workspace directory structure is correct
        expected_dirs = [
            self.workspace_path
            / "PreparationNodeIntegrationTest/src/main/java/de/hofuniversity/PreparationNodeIntegrationTest/",
            self.workspace_path / self.artifact_id,
        ]
        for dir_path in expected_dirs:
            self.assertTrue(
                dir_path.exists(),
                f"Expected directory {dir_path} to exist.",
            )

        # Check that the workspace contains the BXT tool file. The adapter name is
        # derived deterministically from the source/target model folder names
        # (``Families`` -> ``Persons`` => ``FamiliesToPersonsBxToolAdapter``).
        bxtool_path = (
            self.workspace_path
            / "PreparationNodeIntegrationTest/src/main/java/de/hofuniversity/PreparationNodeIntegrationTest/"
            / "FamiliesToPersonsBxToolAdapter.java"
        )
        self.assertTrue(
            bxtool_path.exists(),
            f"Expected BXT tool file {bxtool_path.name} to exist in the workspace.",
        )

        # Check that the workspace contains the TRANSFORMATION.md file
        transformation_plan_path = (
            self.workspace_path / self.artifact_id / "TRANSFORMATION.md"
        )
        self.assertTrue(
            transformation_plan_path.exists(),
            f"Expected TRANSFORMATION.md file at {transformation_plan_path}.",
        )

        # Verify TRANSFORMATION.md is not empty
        transformation_content = transformation_plan_path.read_text()
        self.assertGreater(
            len(transformation_content),
            0,
            "TRANSFORMATION.md should not be empty.",
        )

        # Verify TRANSFORMATION.md contains model information
        self.assertIn(
            "Families",
            transformation_content,
            "TRANSFORMATION.md should reference the source model (Families).",
        )
        self.assertIn(
            "Persons",
            transformation_content,
            "TRANSFORMATION.md should reference the target model (Persons).",
        )

        # Check that Maven pom.xml was created
        pom_path = self.workspace_path / "pom.xml"
        self.assertTrue(
            pom_path.exists(),
            f"Expected pom.xml at {pom_path}.",
        )

        # Verify pom.xml contains group and artifact IDs
        pom_content = pom_path.read_text()
        self.assertIn(
            self.group_id,
            pom_content,
            f"pom.xml should contain group ID '{self.group_id}'.",
        )
        self.assertIn(
            self.artifact_id,
            pom_content,
            f"pom.xml should contain artifact ID '{self.artifact_id}'.",
        )

    def tearDown(self):
        """Clean up the workspace after each test."""
        # Optional: Uncomment to clean up after tests
        # if self.workspace_path.exists():
        #     rmtree(self.workspace_path)
        #     logger.info(f"Cleaned up test workspace at {self.workspace_path}")
        pass
