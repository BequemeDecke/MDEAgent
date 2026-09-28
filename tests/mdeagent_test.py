"""
This test file is the final test for the MDEAgent.

The inputs are some test files found in `./.mdeagent-tests/setup-files` and the required user input to create a maven package including the transformation class.
It uses the provided model in the .env file.
The output should be a functioning workspace. This should be checked by the tests.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

import pytest
from langgraph.types import GraphOutput

from mdeagent.agent import build_mdeagent
from mdeagent.monitoring import build_langfuse_client
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)

TEST_ENVIRONMENT = Path(".mdeagent-tests")
TEST_SETUP_FILES = TEST_ENVIRONMENT / "setup"
TEST_EXECUTION_RUNS = TEST_ENVIRONMENT / "test-executions"


def create_workspace_folder() -> Path:
    workspace = TEST_EXECUTION_RUNS / datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    workspace.mkdir(parents=True)
    return workspace


class TestMDEAgent(TestCase):
    """Test case for the workflow architecture approach."""

    @pytest.fixture(autouse=True)
    def _setup_langfuse(self, enable_langfuse):
        """Set up Langfuse monitoring conditionally based on --enable-langfuse flag."""
        # Create a unique workspace for the test
        self.workspace_path = create_workspace_folder()
        logger.info(f"Created test workspace at {self.workspace_path}")

        # Check if the setup files exist
        self.source_model_path = TEST_SETUP_FILES / "metamodels" / "Families"
        self.target_model_path = TEST_SETUP_FILES / "metamodels" / "Persons"
        if not self.source_model_path.exists() or not self.target_model_path.exists():
            self.fail(
                f"Setup files not found. Please ensure that {self.source_model_path} and {self.target_model_path} exist."
            )

        # Check if the Git-Submodule metamodels have been downloaded.
        # When the submodule is not initialized (e.g. after a fresh clone), the
        # directory exists but is empty and the tests would fail with cryptic
        # errors. Detect this up front and give the user an actionable hint.
        for metamodel_name in ("Families", "Persons"):
            metamodel_dir = TEST_SETUP_FILES / "metamodels" / metamodel_name
            try:
                ecore_files = list((metamodel_dir / "model").glob("*.ecore"))
            except OSError:
                ecore_files = []
            if not ecore_files:
                self.fail(
                    f"Git-Submodule metamodel '{metamodel_name}' was not downloaded "
                    f"(no *.ecore files found in {metamodel_dir / 'model'}).\n"
                    "Please initialize the submodule:"
                    "\n    git submodule update --init --recursive"
                )

        # Build the workflow agent without BenchmarX support
        self.agent = build_mdeagent(self.workspace_path, benchmarx_path=None).compile(
            name="MDEAgent-F2P-Test"
        )

        # Build the Langfuse client for monitoring (optional)
        self.enable_langfuse = enable_langfuse
        if enable_langfuse:
            self.langfuse_client, self.langfuse_callback_handler = (
                build_langfuse_client()
            )
        else:
            self.langfuse_client = None
            self.langfuse_callback_handler = None

    def test_mdeagent_f2p(self):
        """The test method for the MDEAgent workflow.

        Note: Only ainvoke can be used here, because some nodes are executed asynchronously and the test needs to wait for them to finish. The test will fail if the workflow is not completed successfully.
        """
        # 1. Create the initial state for the agent
        initial_state = MDEAgentState(
            source_model_path=self.source_model_path,
            target_model_path=self.target_model_path,
            group_id="de.hofuniversity",
            artifact_id="MDEAgentFamilyToPerson",
            task_specification="Transform the Families model to the Persons model, ensure that the transformation keeps the consistency of the data synchronized.",
            iteration=1,
        )

        # 2. Invoke the agent with the initial state
        callbacks = (
            [self.langfuse_callback_handler] if self.langfuse_callback_handler else []
        )
        output: GraphOutput[MDEAgentState] = asyncio.run(
            self.agent.ainvoke(
                initial_state, config={"callbacks": callbacks}, version="v2"
            )
        )
        if self.langfuse_client:
            self.langfuse_client.flush()

        # 3. Check the output state for expected values
        self.check_output_state(output.value)

        # 4. Check the contents of the workspace for expected files
        self.check_workspace_contents(self.artifact_id)

    def check_output_state(self, output: MDEAgentState):
        """Check the output state for expected values."""
        self.assertIsInstance(
            output, MDEAgentState, "Output state is not of type MDEAgentState."
        )

        # Check that the transformation class path is set
        self.assertIsNotNone(
            output.get("transformation_class_path"),
            "Transformation class path should not be None.",
        )
        self.assertTrue(
            output["transformation_class_path"].exists(),
            "Transformation class file does not exist.",
        )

        # Check that the bxtool path is set
        # self.assertIsNotNone(
        #     output.get("bxtool_path"), "BXT tool path should not be None."
        # )
        # self.assertTrue(output["bxtool_path"].exists(), "BXT tool file does not exist.")

        # Check that the written files list is not empty
        self.assertGreater(
            len(output.get("written_files", [])),
            0,
            "No files were written by the implementation node.",
        )

        # Check that the latest evaluation runs list is not empty
        self.assertGreater(
            len(output.get("latest_evaluation_runs", [])),
            0,
            "No evaluation runs were recorded.",
        )

    def check_workspace_contents(self, artifact_id: str):
        """Check the contents of the workspace for expected files."""
        # Check that the workspace contains the expected files
        expected_files = ["pom.xml", f"{artifact_id}/pom.xml", "Persons/pom.xml", "Families/pom.xml"]
        for file_name in expected_files:
            file_path = self.workspace_path / file_name
            self.assertTrue(
                file_path.exists(),
                f"Expected file {file_name} does not exist in the workspace.",
            )
