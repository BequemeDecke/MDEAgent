"""
Unit tests for the MDEAgent benchmark module.

These tests verify the benchmark functionality without hitting a real LLM.
Tests cover model loading, result saving, and trace collection mocking.
"""

import asyncio
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmark.test_models import (
    collect_trace_info,
    load_models,
    run_benchmark_model,
)


class TestLoadModels(TestCase):
    """Tests for the load_models function."""

    def test_load_models_from_csv(self):
        """Test loading models from a CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,object,created,owned_by,mode\n")
            f.write("model1,model,1234,openai,\n")
            f.write("model2,model,5678,openai,\n")
            f.write("model3,embedding,9012,openai,\n")
            temp_path = Path(f.name)

        try:
            models = load_models(temp_path)
            self.assertEqual(len(models), 3)
            self.assertEqual(models[0], "model1")
            self.assertEqual(models[1], "model2")
            self.assertEqual(models[2], "model3")
        finally:
            temp_path.unlink()

    def test_load_models_handles_bom(self):
        """Test loading models from a CSV file with UTF-8 BOM."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as f:
            f.write("id,object,created,owned_by,mode\n")
            f.write("model1,model,1234,openai,\n")
            temp_path = Path(f.name)

        try:
            models = load_models(temp_path)
            # Should not have BOM character in model ID
            self.assertEqual(models[0], "model1")
            self.assertNotIn("\ufeff", models[0])
        finally:
            temp_path.unlink()

    def test_load_models_empty_file(self):
        """Test loading models from an empty CSV file (only header)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,object,created,owned_by,mode\n")
            temp_path = Path(f.name)

        try:
            models = load_models(temp_path)
            self.assertEqual(len(models), 0)
        finally:
            temp_path.unlink()

    def test_load_models_empty_rows(self):
        """Test loading models from a CSV file with empty rows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,object,created,owned_by,mode\n")
            f.write("model1,model,1234,openai,\n")
            f.write(",model,5678,openai,\n")  # Empty ID
            f.write("model2,model,9012,openai,\n")
            temp_path = Path(f.name)

        try:
            models = load_models(temp_path)
            # Should skip empty IDs
            self.assertEqual(len(models), 2)
            self.assertEqual(models[0], "model1")
            self.assertEqual(models[1], "model2")
        finally:
            temp_path.unlink()

    def test_load_models_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_models(Path("/nonexistent/path/models.csv"))


class TestRunBenchmarkModel(TestCase):
    """Tests for the run_benchmark_model function."""

    def test_run_benchmark_model_success(self):
        """Test successful benchmark execution."""
        async def _test():
            # Create mock objects
            mock_langfuse_client = Mock()
            mock_langfuse_callback = Mock()
            mock_langfuse_callback.trace_id = "test-trace-id"

            with tempfile.TemporaryDirectory(prefix="mdeagent-benchmark-") as temp_dir:
                workspace_path = Path(temp_dir)

                # Mock the build_model and build_mdeagent functions
                mock_model = Mock()
                mock_agent = AsyncMock()
                mock_agent.ainvoke = AsyncMock()

                # Create mock output with expected state
                mock_output = Mock()
                mock_output.value = {
                    "transformation_class_path": workspace_path / "test.class",
                    "bxtool_path": workspace_path / "bxtool",
                    "written_files": [workspace_path / "file1.py", workspace_path / "file2.java"],
                    "latest_evaluation_runs": {"eval1": Mock(), "eval2": Mock()},
                    "iteration": 2,
                }
                mock_agent.ainvoke.return_value = mock_output

                with (
                    patch("benchmark.test_models.build_model", return_value=mock_model),
                    patch("benchmark.test_models.build_mdeagent", return_value=Mock(compile=Mock(return_value=mock_agent))),
                ):
                    result = await run_benchmark_model(
                        model_id="test-model",
                        workspace_path=workspace_path,
                        langfuse_client=mock_langfuse_client,
                        langfuse_callback=mock_langfuse_callback,
                    )

                # Verify result structure
                self.assertTrue(result["success"])
                self.assertEqual(result["model_id"], "test-model")
                self.assertEqual(result["trace_id"], "test-trace-id")
                self.assertEqual(result["iterations_completed"], 2)
                self.assertTrue(result["transformation_class_path"].endswith("test.class"))
                self.assertEqual(len(result["written_files"]), 2)
                self.assertEqual(len(result["evaluation_runs"]), 2)

                # Verify LangFuse was flushed
                mock_langfuse_client.flush.assert_called_once()

        asyncio.run(_test())

    def test_run_benchmark_model_exception_handling(self):
        """Test that exceptions are caught and logged."""
        async def _test():
            mock_langfuse_client = Mock()
            mock_langfuse_callback = Mock()

            with tempfile.TemporaryDirectory(prefix="mdeagent-benchmark-") as temp_dir:
                workspace_path = Path(temp_dir)

                # Mock build_model to raise an exception
                mock_model = Mock()
                mock_agent = AsyncMock()
                mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("Test error"))

                with (
                    patch("benchmark.test_models.build_model", return_value=mock_model),
                    patch("benchmark.test_models.build_mdeagent", return_value=Mock(compile=Mock(return_value=mock_agent))),
                ):
                    result = await run_benchmark_model(
                        model_id="test-model",
                        workspace_path=workspace_path,
                        langfuse_client=mock_langfuse_client,
                        langfuse_callback=mock_langfuse_callback,
                    )

                # Verify error handling
                self.assertFalse(result["success"])
                self.assertIsNotNone(result["error"])
                self.assertEqual(result["error"]["type"], "RuntimeError")
                self.assertEqual(result["error"]["message"], "Test error")
                self.assertIsNone(result["trace_id"])

                # Verify LangFuse was still flushed
                mock_langfuse_client.flush.assert_called_once()

        asyncio.run(_test())


class TestCollectTraceInfo(TestCase):
    """Tests for the collect_trace_info function."""

    def test_collect_trace_info_found(self):
        """Test trace collection when trace is found."""
        mock_trace = Mock()
        mock_trace.id = "test-trace-id"
        mock_traces = Mock()
        mock_traces.data = [mock_trace]

        mock_client = Mock()
        mock_client.api.trace.list.return_value = mock_traces
        mock_client.get_trace_url.return_value = "http://localhost:3000/traces/test-trace-id"

        result = collect_trace_info(mock_client, "test-model")

        self.assertEqual(result["trace_id"], "test-trace-id")
        self.assertEqual(result["trace_url"], "http://localhost:3000/traces/test-trace-id")
        mock_client.api.trace.list.assert_called_once()

    def test_collect_trace_info_not_found(self):
        """Test trace collection when no trace is found."""
        mock_traces = Mock()
        mock_traces.data = []

        mock_client = Mock()
        mock_client.api.trace.list.return_value = mock_traces

        result = collect_trace_info(mock_client, "test-model")

        self.assertIsNone(result["trace_id"])
        self.assertIsNone(result["trace_url"])

    def test_collect_trace_info_error(self):
        """Test trace collection when an error occurs."""
        mock_client = Mock()
        mock_client.api.trace.list.side_effect = Exception("API Error")

        result = collect_trace_info(mock_client, "test-model")

        # Should return empty dict on error, not raise
        self.assertIsNone(result["trace_id"])
        self.assertIsNone(result["trace_url"])


class TestModelIdFormat(TestCase):
    """Tests for model ID format handling."""

    def test_model_id_with_slashes(self):
        """Test that model IDs with slashes are handled correctly."""
        model_id = "Qwen/Qwen3.8-27B-FP8"
        # The model ID should be usable in agent names after replacement
        agent_name = f"MDEAgent-Benchmark-{model_id.replace('/', '-')}"
        self.assertNotIn("/", agent_name)
        self.assertEqual(agent_name, "MDEAgent-Benchmark-Qwen-Qwen3.8-27B-FP8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
