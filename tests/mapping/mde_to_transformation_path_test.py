"""Tests for the mde_to_transformation_path mapper function."""

from pathlib import Path
from unittest import TestCase

from mdeagent.mapping import mde_to_transformation_path
from mdeagent.state import MDEAgentState


class TestMDEToTransformationPathMapper(TestCase):
    """Test cases for mde_to_transformation_path mapping function."""

    def test_mapping_returns_transformation_class_path(self):
        """Mapper should return dict with transformation_class_path key."""
        state = MDEAgentState(
            workspace_path=Path("/tmp/ws"),
            source_model_path=Path("/tmp/source"),
            target_model_path=Path("/tmp/target"),
            transformation_class_path=Path("/tmp/Transform.java"),
        )

        result = mde_to_transformation_path(state)

        self.assertIn("transformation_class_path", result)
        self.assertEqual(result["transformation_class_path"], "/tmp/Transform.java")

    def test_mapping_without_transformation_class_path(self):
        """Mapper should handle missing transformation_class_path gracefully."""
        state = MDEAgentState(
            workspace_path=Path("/tmp/ws"),
            source_model_path=Path("/tmp/source"),
            target_model_path=Path("/tmp/target"),
        )

        result = mde_to_transformation_path(state)

        self.assertIn("transformation_class_path", result)
        self.assertEqual(result["transformation_class_path"], "")

    def test_mapping_only_includes_transformation_class_path(self):
        """Mapper should only return transformation_class_path, not all state."""
        state = MDEAgentState(
            workspace_path=Path("/tmp/ws"),
            source_model_path=Path("/tmp/source"),
            target_model_path=Path("/tmp/target"),
            transformation_class_path=Path("/tmp/Transform.java"),
            group_id="test.group",
            artifact_id="test-artifact",
            written_files=[Path("/tmp/file1.java")],
        )

        result = mde_to_transformation_path(state)

        self.assertEqual(len(result), 1)
        self.assertIn("transformation_class_path", result)
        self.assertNotIn("workspace_path", result)
        self.assertNotIn("group_id", result)
        self.assertNotIn("written_files", result)

    def test_mapping_with_nested_path(self):
        """Mapper should handle nested paths correctly."""
        state = MDEAgentState(
            workspace_path=Path("/tmp/ws"),
            source_model_path=Path("/tmp/source"),
            target_model_path=Path("/tmp/target"),
            transformation_class_path=Path("/tmp/ws/src/main/java/com/example/Transform.java"),
        )

        result = mde_to_transformation_path(state)

        self.assertIn("transformation_class_path", result)
        self.assertEqual(
            result["transformation_class_path"],
            "/tmp/ws/src/main/java/com/example/Transform.java",
        )

    def test_mapping_with_none_value(self):
        """Mapper should handle None value explicitly."""
        state = MDEAgentState(
            workspace_path=Path("/tmp/ws"),
            source_model_path=Path("/tmp/source"),
            target_model_path=Path("/tmp/target"),
            transformation_class_path=None,
        )

        result = mde_to_transformation_path(state)

        self.assertIn("transformation_class_path", result)
        self.assertEqual(result["transformation_class_path"], "")
