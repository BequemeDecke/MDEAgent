"""Tests for the implementation_to_transformation_path mapper function."""

from pathlib import Path
from unittest import TestCase

from mdeagent.implementation.state import ImplementationState
from mdeagent.mapping import implementation_to_transformation_path


class TestImplementationToTransformationPathMapper(TestCase):
    """Test cases for implementation_to_transformation_path mapping function."""

    def _make_state_with_transformation_class(
        self, path: Path
    ) -> ImplementationState:
        """Helper to build an ImplementationState with a transformation_class."""
        return ImplementationState(
            transformation_plan={
                "data": {},
                "parser": {"type": "FileTransformationPlanParser", "args": {}},
                "template": Path("/tmp"),
            },
            transformation_class={"name": "Transform", "package": "com.example", "path": path, "code": "class Transform {}"},
            task_specification="Test task",
            maven_project_path=Path("/tmp/project"),
            bxtool_path=Path("/tmp/bxtool"),
            written_files=[path],
        )

    def test_mapping_returns_transformation_class_path(self):
        """Mapper should return dict with transformation_class_path key."""
        state = self._make_state_with_transformation_class(
            Path("/tmp/ws/src/main/java/Transform.java")
        )
        result = implementation_to_transformation_path(state)
        self.assertIn("transformation_class_path", result)
        self.assertEqual(
            result["transformation_class_path"],
            "/tmp/ws/src/main/java/Transform.java",
        )

    def test_mapping_without_transformation_class(self):
        """Mapper should handle missing transformation_class gracefully."""
        state = ImplementationState(
            transformation_plan={
                "data": {},
                "parser": {"type": "FileTransformationPlanParser", "args": {}},
                "template": Path("/tmp"),
            },
            transformation_class=None,
            task_specification="Test task",
            maven_project_path=Path("/tmp/project"),
            bxtool_path=Path("/tmp/bxtool"),
            written_files=[],
        )
        result = implementation_to_transformation_path(state)
        self.assertIn("transformation_class_path", result)
        self.assertEqual(result["transformation_class_path"], "")

    def test_mapping_only_includes_transformation_class_path(self):
        """Mapper should only return transformation_class_path, not all state."""
        state = self._make_state_with_transformation_class(
            Path("/tmp/Transform.java")
        )
        result = implementation_to_transformation_path(state)
        self.assertEqual(len(result), 1)
        self.assertIn("transformation_class_path", result)
        self.assertNotIn("transformation_plan", result)
        self.assertNotIn("task_specification", result)
        self.assertNotIn("maven_project_path", result)

    def test_mapping_with_nested_path(self):
        """Mapper should handle deeply nested paths correctly."""
        nested_path = Path(
            "/tmp/workspace/module/src/main/java/de/hofuniversity/transform/Transform.java"
        )
        state = self._make_state_with_transformation_class(nested_path)
        result = implementation_to_transformation_path(state)
        self.assertIn("transformation_class_path", result)
        self.assertEqual(result["transformation_class_path"], str(nested_path))

    def test_mapping_preserves_path_type_conversion(self):
        """Mapper should convert Path to string."""
        state = self._make_state_with_transformation_class(Path("/tmp/Transform.java"))
        result = implementation_to_transformation_path(state)
        self.assertIsInstance(result["transformation_class_path"], str)
        self.assertEqual(result["transformation_class_path"], "/tmp/Transform.java")
