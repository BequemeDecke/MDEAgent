"""Tests for the transformation_code evaluation implementation."""

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase

from mdeagent.evaluation.implementations.transformation_code import (
    TransformationCodeEvaluation,
    TransformationCodeSchema,
)


class TestTransformationCodeSchema(TestCase):
    """Test cases for TransformationCodeSchema."""

    def test_schema_accepts_valid_path(self):
        """Schema should accept a valid path string."""
        data = {"transformation_class_path": "/path/to/Transform.java"}
        schema = TransformationCodeSchema.model_validate(data)
        self.assertEqual(schema.transformation_class_path, "/path/to/Transform.java")

    def test_schema_rejects_empty_path(self):
        """Schema should reject an empty path string."""
        data = {"transformation_class_path": ""}
        with self.assertRaises(Exception):
            TransformationCodeSchema.model_validate(data)

    def test_schema_accepts_relative_path(self):
        """Schema should accept a relative path string."""
        data = {"transformation_class_path": "src/main/java/Transform.java"}
        schema = TransformationCodeSchema.model_validate(data)
        self.assertEqual(
            schema.transformation_class_path, "src/main/java/Transform.java"
        )

    def test_schema_rejects_missing_path(self):
        """Schema should reject when transformation_class_path is missing."""
        data = {}
        with self.assertRaises(Exception):
            TransformationCodeSchema.model_validate(data)


class TestTransformationCodeEvaluationSetup(TestCase):
    """Test cases for TransformationCodeEvaluation.setup()."""

    def test_setup_returns_none(self):
        """Setup should be a no-op returning None."""
        evaluation = TransformationCodeEvaluation()
        result = asyncio.run(evaluation.setup())
        self.assertIsNone(result)


class TestTransformationCodeEvaluationRun(TestCase):
    """Test cases for TransformationCodeEvaluation.run()."""

    def test_run_no_path_provided(self):
        """Evaluation should return failure when no path is provided."""
        evaluation = TransformationCodeEvaluation()
        results, errors = asyncio.run(evaluation.run(transformation_class_path=""))
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].metadata["success"])
        self.assertIn("No transformation class path", results[0].content)

    def test_run_file_does_not_exist(self):
        """Evaluation should return failure when file does not exist."""
        evaluation = TransformationCodeEvaluation()
        results, errors = asyncio.run(
            evaluation.run(transformation_class_path="/nonexistent/path/Transform.java")
        )
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].metadata["success"])
        self.assertIn("does not exist", results[0].content)

    def test_run_file_is_empty(self):
        """Evaluation should return failure when file is empty."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
            f.write(b"")
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].metadata["success"])
            self.assertIn("empty", results[0].content)
        finally:
            Path(temp_path).unlink()

    def test_run_file_is_whitespace_only(self):
        """Evaluation should return failure when file contains only whitespace (treated as empty)."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("   \n\n  \t  ")
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].metadata["success"])
            # Whitespace-only files are treated as empty
            self.assertIn("empty", results[0].content)
        finally:
            Path(temp_path).unlink()

    def test_run_file_only_comments(self):
        """Evaluation should return failure when file contains only comments."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("// This is just a comment\n/* Block comment */")
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].metadata["success"])
            self.assertIn("does not contain code", results[0].content)
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_class_keyword(self):
        """Evaluation should pass when file contains Java class keyword."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
// Comment
public class Transform {
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
            self.assertIn("contains code", results[0].content)
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_interface_keyword(self):
        """Evaluation should pass when file contains Java interface keyword."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
public interface Transformable {
    void transform();
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_method_with_return(self):
        """Evaluation should pass when file contains method with return statement."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
public class Transform {
    public String getName() {
        return "test";
    }
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_for_loop(self):
        """Evaluation should pass when file contains control flow keyword."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
public class Transform {
    public void process() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_class_declaration(self):
        """Evaluation should pass when file contains class declaration pattern."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
class SimpleClass {
    int value = 42;
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
        finally:
            Path(temp_path).unlink()

    def test_run_file_contains_variable_declaration(self):
        """Evaluation should pass when file contains variable declaration."""
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="w") as f:
            f.write("""
public class Transform {
    private String fieldName;
    public void setField(String fieldName) {
        this.fieldName = fieldName;
    }
}
            """)
            f.flush()
            temp_path = f.name

        try:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=temp_path)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].metadata["success"])
        finally:
            Path(temp_path).unlink()

    def test_run_path_is_directory(self):
        """Evaluation should return failure when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluation = TransformationCodeEvaluation()
            results, errors = asyncio.run(
                evaluation.run(transformation_class_path=tmpdir)
            )
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].metadata["success"])
            self.assertIn("not a file", results[0].content)
