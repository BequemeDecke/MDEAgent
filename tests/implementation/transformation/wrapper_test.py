import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, Mock

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
        virtual_root = Path("/")

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
        virtual_root = Path("/")

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
        virtual_root = Path("/")

        # Create mock evaluation results with file paths
        evaluation_results = [
            EvaluationResult(
                content="File exists: /mock/workspace/subdir/test_file.py",
                metadata={
                    "file": str(workspace / "subdir" / "test_file.py"),
                    "success": True,
                    "include_in_report": False,
                },
            ),
            EvaluationResult(
                content="No file metadata",
                metadata={
                    "success": True,
                    "include_in_report": False,
                },
            ),
        ]

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to virtualize paths in evaluation results
        virtualized_results = wrapper.virtualize_paths_in_evaluation_results(
            evaluation_results
        )

        # Assert that the file path in the evaluation results has been virtualized correctly
        expected_virtual_path = real_to_virtual(
            Path(evaluation_results[0].metadata["file"]),
            workspace,
            virtual_root,
        )
        self.assertEqual(
            Path(virtualized_results[0].metadata["file"]),
            expected_virtual_path,
        )
        # Assert that results without file metadata are unchanged
        self.assertEqual(len(virtualized_results), 2)
        self.assertNotIn("file", virtualized_results[1].metadata)

    def test_realize_paths_in_evaluation_results(self):
        workspace = Path("/mock/workspace")
        virtual_root = Path("/")

        # Create mock evaluation results with virtual file paths
        evaluation_results = [
            EvaluationResult(
                content="File exists: /subdir/test_file.py",
                metadata={
                    "file": str(virtual_root / "subdir" / "test_file.py"),
                    "success": True,
                    "include_in_report": False,
                },
            ),
            EvaluationResult(
                content="No file metadata",
                metadata={
                    "success": True,
                    "include_in_report": False,
                },
            ),
        ]

        # Create an instance of the wrapper with a mock workspace
        wrapper = TransformationClassAgentWrapper(workspace, graph=None)

        # Call the method to realize paths in evaluation results
        realized_results = wrapper.realize_paths_in_evaluation_results(
            evaluation_results
        )

        # Assert that the file path in the evaluation results has been realized correctly
        expected_real_path = virtual_to_real(
            Path(evaluation_results[0].metadata["file"]),
            virtual_root,
            workspace,
        )
        self.assertEqual(
            Path(realized_results[0].metadata["file"]),
            expected_real_path,
        )
        # Assert that results without file metadata are unchanged
        self.assertEqual(len(realized_results), 2)
        self.assertNotIn("file", realized_results[1].metadata)

    def test_synthesize_transformation_class_returns_all_written_files(self):
        workspace = Path("/mock/workspace")
        transformation_class_path = workspace / "src" / "TestClass.java"
        transformation_class = TransformationClass(
            name="TestClass",
            path=transformation_class_path,
            package="test_package",
        )
        plan_data = {
            "source_model_package": "source",
            "target_model_package": "target",
            "iteration": 1,
            "source_model_implementation": "source implementation",
            "target_model_implementation": "target implementation",
            "transformation_direction": "source to target",
            "difficulties": "none",
            "implementation_steps": "implement the transformation",
        }
        transformation_plan = Mock()
        transformation_plan.data = plan_data
        transformation_plan.to_dict.return_value = {
            "data": plan_data,
            "parser": {},
            "template": Path("templates"),
        }

        graph = Mock()
        graph.ainvoke = AsyncMock(
            return_value=SimpleNamespace(
                value={
                    "written_files": [
                        "/src/TestClass.java",
                        "/src/Source.java",
                        "/src/Target.java",
                    ]
                }
            )
        )
        wrapper = TransformationClassAgentWrapper(workspace, graph)

        written_files = asyncio.run(
            wrapper.synthesize_transformation_class(
                transformation_plan,
                transformation_class,
            )
        )

        self.assertEqual(
            set(written_files),
            {
                transformation_class_path,
                workspace / "src" / "Source.java",
                workspace / "src" / "Target.java",
            },
        )
        graph.ainvoke.assert_awaited_once()
