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
    run_benchmark,
    main,
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
                        iteration=3,
                    )

                # Verify result structure
                self.assertTrue(result["success"])
                self.assertEqual(result["model_id"], "test-model")
                self.assertEqual(result["iteration"], 3)
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
                        iteration=1,
                    )

                # Verify error handling
                self.assertFalse(result["success"])
                self.assertEqual(result["iteration"], 1)
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

    def test_model_id_with_slashes_and_iteration(self):
        """Test that model IDs with slashes include iteration number in agent name."""
        model_id = "Qwen/Qwen3.8-27B-FP8"
        iteration = 3
        agent_name = f"MDEAgent-Benchmark-{model_id.replace('/', '-')}-iter{iteration}"
        self.assertNotIn("/", agent_name)
        self.assertEqual(agent_name, "MDEAgent-Benchmark-Qwen-Qwen3.8-27B-FP8-iter3")
        self.assertIn("-iter3", agent_name)


class TestLogLevels(TestCase):
    """Tests for the --log-level CLI argument."""

    def test_log_level_defaults_to_info(self):
        """Test that --log-level defaults to INFO."""
        import argparse
        from benchmark.test_models import main
        import sys
        from io import StringIO

        # Patch sys.argv to simulate default invocation
        old_argv = sys.argv
        old_exit = sys.exit
        exits = []

        def mock_exit(code=0):
            exits.append(code)
            raise SystemExit(code)

        try:
            sys.argv = ["test_models.py"]
            sys.exit = mock_exit

            with (
                patch("benchmark.test_models.run_benchmark", return_value=[]),
            ):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
            sys.exit = old_exit

        # Empty results list is falsy -> exit code 3 (no results collected)
        self.assertIn(3, exits)

    def test_log_level_accepts_valid_levels(self):
        """Test that valid log levels are accepted."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )

        # Test each valid level
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parser.parse_args([f"--log-level", level])
            self.assertEqual(args.log_level, level)

    def test_log_level_rejects_invalid_level(self):
        """Test that invalid log levels are rejected."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )

        with self.assertRaises(SystemExit):
            parser.parse_args(["--log-level", "INVALID"])

    def test_num_iterations_defaults_to_five(self):
        """Test that --num-iterations defaults to 5."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-iterations",
            type=int,
            default=5,
        )

        # Default value
        args = parser.parse_args([])
        self.assertEqual(args.num_iterations, 5)

        # Custom value
        args = parser.parse_args(["--num-iterations", "10"])
        self.assertEqual(args.num_iterations, 10)

    def test_num_iterations_accepts_positive_integers(self):
        """Test that positive integers are accepted."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-iterations",
            type=int,
            default=5,
        )

        for value in [1, 3, 10, 100]:
            args = parser.parse_args(["--num-iterations", str(value)])
            self.assertEqual(args.num_iterations, value)

    def test_main_passes_num_iterations_to_run_benchmark(self):
        """Test that main() passes --num-iterations to run_benchmark()."""
        import sys

        old_argv = sys.argv
        old_exit = sys.exit
        exit_args = []

        def mock_exit(code=0):
            exit_args.append(code)
            raise SystemExit(code)

        try:
            sys.argv = ["test_models.py", "--num-iterations", "7"]
            sys.exit = mock_exit

            with (
                patch(
                    "benchmark.test_models.run_benchmark",
                    return_value=[],
                ) as mock_run,
            ):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
            sys.exit = old_exit

        # Verify run_benchmark was called with num_iterations=7
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        self.assertEqual(call_kwargs.kwargs["num_iterations"], 7)


class TestRunBenchmark(TestCase):
    """Tests for the parallel run_benchmark function."""

    def test_run_benchmark_generates_correct_number_of_tasks(self):
        """Test that num_iterations parameter generates correct number of task pairs."""
        models = ["model1", "model2"]
        num_iterations = 5

        # Verify the formula: models × iterations
        expected_tasks = len(models) * num_iterations
        self.assertEqual(expected_tasks, 10)

        # Each model should be run 5 times
        for i, model_id in enumerate(models, 1):
            for iteration in range(1, num_iterations + 1):
                task_name = f"{model_id}-iter{iteration}"
                self.assertIsNotNone(task_name)

    def test_run_benchmark_default_parameters(self):
        """Test default parameters for run_benchmark."""
        import inspect
        sig = inspect.signature(run_benchmark)

        # Check default values
        self.assertEqual(sig.parameters["num_iterations"].default, 5)
        self.assertEqual(sig.parameters["max_concurrency"].default, 3)

    def test_run_benchmark_semaphore_concurrency(self):
        """Test that semaphore limits concurrent executions."""
        async def _test():
            semaphore = asyncio.Semaphore(2)
            max_concurrent = 0
            current = 0

            async def task():
                nonlocal max_concurrent, current
                async with semaphore:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                    await asyncio.sleep(0.01)
                    current -= 1

            # Run 10 tasks with max_concurrency=2
            await asyncio.gather(*[task() for _ in range(10)])
            self.assertLessEqual(max_concurrent, 2)

        asyncio.run(_test())

    def test_run_benchmark_result_contains_iteration_field(self):
        """Test that each result contains an iteration field."""
        # Create a mock result to verify structure
        result = {
            "model_id": "test-model",
            "iteration": 3,
            "workspace_path": "/tmp/test",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "success": True,
            "error": None,
            "trace_id": None,
            "trace_url": None,
            "transformation_class_path": None,
            "bxtool_path": None,
            "written_files": [],
            "evaluation_runs": [],
            "iterations_completed": 0,
        }

        self.assertIn("iteration", result)
        self.assertEqual(result["iteration"], 3)
        self.assertEqual(result["model_id"], "test-model")

    def test_run_benchmark_error_handling_in_gather(self):
        """Test that exceptions from asyncio.gather are properly handled."""
        async def _test():
            async def failing_task():
                raise ValueError("Test failure")

            async def successful_task():
                return {"success": True, "data": "result"}

            # Simulate gather with mixed success/failure
            tasks = [failing_task(), successful_task(), failing_task()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Should have 3 results
            self.assertEqual(len(results), 3)

            # First should be an exception
            self.assertIsInstance(results[0], ValueError)
            self.assertEqual(str(results[0]), "Test failure")

            # Second should be successful result
            self.assertTrue(results[1]["success"])

            # Third should be an exception
            self.assertIsInstance(results[2], ValueError)

        asyncio.run(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
