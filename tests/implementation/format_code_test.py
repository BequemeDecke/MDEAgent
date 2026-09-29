import asyncio
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from mdeagent.implementation.format_code import create_format_code_node
from mdeagent.implementation.state import ImplementationState
from mdeagent.preparation.maven import MavenProject


class TestFormatCodeNode(TestCase):
    def setUp(self):
        self.format_code_node = create_format_code_node(
            workspace=Path("/fake/workspace")
        )

    @patch("mdeagent.preparation.maven.MavenProject.format")
    def test_format_code_node__calls_format_java_files(self, mock_format):
        """Test that the format_code node calls MavenProject.format_code and returns the state unchanged"""
        from mdeagent.implementation.types import TransformationClass
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "example"
            workspace_path.mkdir(parents=True, exist_ok=True)

            project = MavenProject.create(
                workspace_path, group_id="com.example", artifact_id="example-artifact"
            )
            bxtool_path = project.add_java_class(
                package="com.example", class_name="BxTool", content="public class BxTool {}"
            )
            transformation_class_path = project.add_java_class(
                package="com.example", class_name="Transformation", content="public class Transformation {}"
            )

            state: ImplementationState = {
                "transformation_plan": None,  # type: ignore
                "transformation_class": {
                    "name": "Transformation",
                    "package": "com.example",
                    "path": transformation_class_path,
                    "code": "public class Transformation { /* test */ }",
                },
                "task_specification": "Test task",
                "maven_project_path": workspace_path,
                "bxtool_path": bxtool_path,
                "written_files": [],
                "latest_evaluation_runs": {},
                "iteration": 1,
            }

            result = asyncio.run(self.format_code_node(state))

            mock_format.assert_called_once()
            self.assertEqual(result, state)


class TestFormatJavaFiles(TestCase):
    @patch("subprocess.run")
    def test_format_java_files__success(self, mock_run):
        """Test that format_java_files runs successfully when mvn spotless:apply succeeds"""
        from mdeagent.preparation.maven import MavenProject

        mock_run.return_value = subprocess.CompletedProcess(
            args=["mvn", "spotless:apply"], returncode=0, stdout="", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            maven_project = MavenProject.create(
                workspace, group_id="com.example", artifact_id="example-artifact"
            )
            maven_project.format()  # This should call the mocked subprocess.run

            mock_run.assert_called_once_with(
                ["mvn", "spotless:apply"],
                capture_output=True,
                cwd=workspace,
                check=False,
                text=True,
            )

    @patch("subprocess.run")
    def test_format_java_files__failure_returns_error_code(self, mock_run):
        """Test that format_java_files raises an error when mvn spotless:apply fails"""
        from mdeagent.preparation.maven import MavenProject

        mock_run.return_value = subprocess.CompletedProcess(
            args=["mvn", "spotless:apply"],
            returncode=1,
            stdout="Some output",
            stderr="Some error",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project = MavenProject.create(
                workspace, group_id="com.example", artifact_id="example-artifact"
            )

            has_formatting_succeeded = (
                project.format()
            )  # This should call the mocked subprocess.run

            self.assertFalse(has_formatting_succeeded)
