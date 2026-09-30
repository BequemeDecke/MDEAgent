"""
This test checks if the preparation node correctly prepares the workspace.
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation import EvaluationExecutor, implementations
from mdeagent.preparation.graph import build_preparation_graph
from mdeagent.preparation.node import (
    create_preparation_node,
)
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)


def create_test_model_package(temp_dir: Path, package_name: str):
    package_path = temp_dir / package_name
    package_path.mkdir()

    # Create a minimal pom.xml so explore_models can copy the model as a Maven project
    pom_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>{package_name}</artifactId>
    <version>1.0.0</version>
</project>
"""
    (package_path / "pom.xml").write_text(pom_content)

    source_file = package_path / f"{package_name}.java"
    source_register_file = package_path / f"{package_name}Register.java"
    source_package_file = package_path / f"{package_name}Package.java"
    source_factory_file = package_path / f"{package_name}Factory.java"

    source_file.write_text(f"public interface {package_name} {{ }}")
    source_register_file.write_text(f"public interface {package_name}Register {{ }}")
    source_package_file.write_text(f"public interface {package_name}Package {{ }}")
    source_factory_file.write_text(f"public interface {package_name}Factory {{ }}")
    return (
        package_path,
        source_file,
        source_register_file,
        source_package_file,
        source_factory_file,
    )


class TestPreparationNodeIntegration(TestCase):
    def setUp(self):
        mvn_path = shutil.which("mvn")
        if mvn_path is None:
            import os

            logger.warning(
                f"Maven is not installed or not in PATH. "
                f"Current PATH: {os.environ.get('PATH', 'Not set')[:200]}. "
                f"Skipping integration tests."
            )
            self.skipTest("Maven is not found in PATH. Skipping integration tests.")

        self.preparation_agent = build_preparation_graph(
            evaluation_executor=EvaluationExecutor(
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
            ),
            max_iterations=10,
        ).compile()

    def test_preparation_node__invoke_subgraph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            workspace_path.mkdir()
            models_path = Path(temp_dir) / "models"
            models_path.mkdir()
            group_id = "de.example"
            artifact_id = "mdagent"
            transformation_plan_path = (
                workspace_path / artifact_id / "TRANSFORMATION.md"
            )
            required_tools = ["mvn", "git"]

            (
                (source_model_path, source_file, *_),
                (target_model_path, target_file, *_),
            ) = (
                create_test_model_package(models_path, "Source"),
                create_test_model_package(models_path, "Target"),
            )

            # Create the preparation node with required parameters
            call_preparation_node = create_preparation_node(
                self.preparation_agent, workspace_path, required_tools
            )

            initial_state = MDEAgentState(
                transformation_plan=None,
                group_id=group_id,
                artifact_id=artifact_id,
                source_model_path=source_model_path,
                target_model_path=target_model_path,
            )

            output: MDEAgentState = asyncio.run(call_preparation_node(initial_state))
            logger = logging.getLogger(__name__)
            logger.debug(f"Output state: {output}")

            # Check that the preparation node set workspace_path and required_tools in the output
            self.assertIsNotNone(
                output.get("workspace_path"),
                "The preparation node should set workspace_path in the output state.",
            )
            self.assertEqual(
                output["workspace_path"],
                workspace_path,
                "The output workspace_path should match the expected workspace path.",
            )
            self.assertIsNotNone(
                output.get("required_tools"),
                "The preparation node should set required_tools in the output state.",
            )
            self.assertEqual(
                output["required_tools"],
                required_tools,
                "The output required_tools should match the expected tools.",
            )
            self.assertEqual(
                output.get("maven_project_path"),
                workspace_path / artifact_id,
                "The output state should contain the maven project path.",
            )

            # Check that the transformation plan file was created
            self.assertTrue(
                transformation_plan_path.exists(),
                "The preparation node should create a TRANSFORMATION.md file in the workspace.",
            )

            # Check that the output state contains a transformation plan
            self.assertIsNotNone(
                output.get("transformation_plan"),
                "The output state should contain a transformation plan.",
            )
            self.assertIsInstance(
                output["transformation_plan"],
                dict,
                "The transformation plan in the output state should be a serialized dictionary.",
            )
            tp = TransformationPlan.from_dict(output["transformation_plan"])
            tp_data = tp.data

            # Check that the transformation plan contains the model implementations
            self.assertIn(
                source_file.read_text(),
                tp_data["source_model_implementation"],
                "The transformation plan should contain the source model implementation.",
            )
            self.assertIn(
                target_file.read_text(),
                tp_data["target_model_implementation"],
                "The transformation plan should contain the target model implementation.",
            )

            # Check that the transformation plan contains the correct package names
            self.assertEqual(
                tp_data["source_model_package"],
                "Source",
                "The transformation plan should contain the correct source model package.",
            )
            self.assertEqual(
                tp_data["target_model_package"],
                "Target",
                "The transformation plan should contain the correct target model package.",
            )

            # Check that the transformation_class_path is set and the file exists
            self.assertIsNotNone(
                output.get("transformation_class_path"),
                "The output state should contain the path to the transformation class file.",
            )
            self.assertTrue(
                output.get("transformation_class_path").exists(),
                "The preparation node should create the transformation class file.",
            )

            # Check that the bxtool_path is set and the file exists
            self.assertIsNotNone(
                output.get("bxtool_path"),
                "The output state should contain the path to the BxAgentJavaBxTool.java file.",
            )
            self.assertTrue(
                output.get("bxtool_path").exists(),
                "The preparation node should create a BxAgentJavaBxTool.java file.",
            )
