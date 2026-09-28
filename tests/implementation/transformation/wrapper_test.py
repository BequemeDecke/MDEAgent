from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

from mdeagent.evaluation.types import EvaluationResult, EvaluationRun
from mdeagent.implementation.transformation.react.wrapper import (
    TransformationClassAgentWrapper,
)
from mdeagent.implementation.types import TransformationClass
from mdeagent.util import real_to_virtual, virtual_to_real


class TestReactWrapper(TestCase):
    def setUp(self):
        pass

    def test_virtualize_paths_in_class(self):
        workspace = Path("/mock/workspace")
        virtual_root = Path("/workspace")

        # Create a mock transformation class with a real path
        transformation_class = TransformationClass(
            name="TestClass",
            path=workspace / "subdir" / "test_file.py",
            package="test_package",
        )

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to virtualize paths
        virtualized_class = wrapper.virtualize_paths_in_class(transformation_class)

        # Assert that the path has been virtualized correctly
        expected_virtual_path = real_to_virtual(
            transformation_class["path"], workspace, virtual_root
        )
        self.assertEqual(virtualized_class["path"], expected_virtual_path)

    def test_realize_paths_in_class(self):
        workspace = Path("/mock/workspace")
        virtual_root = Path("/workspace")

        # Create a mock transformation class with a virtual path
        transformation_class = TransformationClass(
            name="TestClass",
            path=virtual_root / "subdir" / "test_file.py",
            package="test_package",
        )

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to realize paths
        realized_class = wrapper.realize_paths_in_class(transformation_class)

        # Assert that the path has been realized correctly
        expected_real_path = virtual_to_real(
            transformation_class["path"], virtual_root, workspace
        )
        self.assertEqual(realized_class["path"], expected_real_path)

    def test_virtualize_paths_in_evaluation_results(self):
        workspace = Path("/mock/workspace")
        virtual_root = Path("/workspace")

        # Create a mock evaluation results dictionary with a file existence run
        evaluation_results = {
            "file_existence": EvaluationRun(
                started_at=datetime.now(UTC),
                execution_time_ms=100,
                iteration=1,
                errors=[],
                results=[
                    EvaluationResult(
                        content="File exists: /mock/workspace/subdir/test_file.py",
                        metadata={
                            "file": str(workspace / "subdir" / "test_file.py"),
                            "success": True,
                            "include_in_report": False,
                        },
                    )
                ]
            )
        }

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to virtualize paths in evaluation results
        virtualized_results = wrapper.virtualize_paths_in_evaluation_results(
            evaluation_results
        )

        # Assert that the file path in the evaluation results has been virtualized correctly
        expected_virtual_path = real_to_virtual(
            Path(evaluation_results["file_existence"].results[0].metadata["file"]),
            workspace,
            virtual_root,
        )
        self.assertEqual(
            Path(virtualized_results["file_existence"].results[0].metadata["file"]),
            expected_virtual_path,
        )

    def test_realize_paths_in_evaluation_results(self):
        workspace = Path("/mock/workspace")
        virtual_root = Path("/workspace")

        # Create a mock evaluation results dictionary with a file existence run
        evaluation_results = {
            "file_existence": EvaluationRun(
                started_at=datetime.now(UTC),
                execution_time_ms=100,
                iteration=1,
                errors=[],
                results=[
                    EvaluationResult(
                        content="File exists: /workspace/subdir/test_file.py",
                        metadata={
                            "file": str(virtual_root / "subdir" / "test_file.py"),
                            "success": True,
                            "include_in_report": False,
                        },
                    )
                ],
            )
        }

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to realize paths in evaluation results
        realized_results = wrapper.realize_paths_in_evaluation_results(
            evaluation_results
        )

        # Assert that the file path in the evaluation results has been realized correctly
        expected_real_path = virtual_to_real(
            Path(evaluation_results["file_existence"].results[0].metadata["file"]),
            virtual_root,
            workspace,
        )
        self.assertEqual(
            Path(realized_results["file_existence"].results[0].metadata["file"]),
            expected_real_path,
        )
