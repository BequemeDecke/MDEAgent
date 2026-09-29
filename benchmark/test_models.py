"""
Benchmark script for the MDEAgent.

This script reads models from benchmark/models.csv, creates a temporary workspace,
builds and runs the MDEAgent with each model, collects results via LangFuse,
and saves results to benchmark/results.json.
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from mdeagent.agent import build_mdeagent
from mdeagent.monitoring import build_langfuse_client
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_CSV = PROJECT_ROOT / ".mdeagent-benchmark" / "models.csv"
RESULTS_DIR = PROJECT_ROOT / ".mdeagent-benchmark" / "results"
RESULTS_JSON = PROJECT_ROOT / ".mdeagent-benchmark" / "results.json"
TEST_SETUP_FILES = PROJECT_ROOT / ".mdeagent-tests" / "setup"
SOURCE_MODEL_PATH = TEST_SETUP_FILES / "metamodels" / "Families"
TARGET_MODEL_PATH = TEST_SETUP_FILES / "metamodels" / "Persons"

# Task specification (same as in mdeagent_test.py)
TASK_SPECIFICATION = (
    "Transform the Families model to the Persons model, "
    "ensure that the transformation keeps the consistency of the data synchronized."
)


def load_models(csv_path: str | Path) -> list[str]:
    """Load model IDs from the models CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Models file not found: {csv_path}")

    models = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get("id", "").strip()
            if model_id:
                models.append(model_id)
    return models


def build_model(model_id: str) -> Any:
    """Build a LangChain chat model for the given model ID."""
    from mdeagent.config import Config

    config = Config.get_instance()
    model_config = config.MODEL

    return init_chat_model(
        model_provider="openai",
        base_url=model_config.BASE_URL,
        api_key=model_config.API_KEY.get_secret_value(),
        model=model_id,
        request_timeout=model_config.REQUEST_TIMEOUT,
        max_retries=model_config.MAX_RETRIES,
    )


def _run_single_iteration_sync(model_id: str, iteration: int) -> dict[str, Any]:
    """Run a single benchmark iteration in a new asyncio event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_single_iteration(model_id, iteration))
    finally:
        loop.close()


async def _run_single_iteration(model_id: str, iteration: int) -> dict[str, Any]:
    """Run a single benchmark iteration for one model with its own workspace."""
    with tempfile.TemporaryDirectory(prefix="mdeagent-benchmark-") as temp_dir:
        workspace_path = Path(temp_dir)
        langfuse_client_instance, langfuse_callback = build_langfuse_client()

        try:
            result = await run_benchmark_model(
                model_id=model_id,
                workspace_path=workspace_path,
                langfuse_client=langfuse_client_instance,
                langfuse_callback=langfuse_callback,
                iteration=iteration,
            )
            trace_info = collect_trace_info(langfuse_client_instance, model_id)
            result["trace_id"] = trace_info.get("trace_id")
            result["trace_url"] = trace_info.get("trace_url")
            return result
        finally:
            langfuse_client_instance.flush()
async def run_benchmark_model(
    model_id: str,
    workspace_path: Path,
    langfuse_client: Langfuse,
    langfuse_callback: LangfuseCallbackHandler,
    iteration: int = 1,
) -> dict[str, Any]:
    """Run a single benchmark iteration for one model."""
    result: dict[str, Any] = {
        "model_id": model_id,
        "iteration": iteration,
        "workspace_path": str(workspace_path),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "success": False,
        "error": None,
        "trace_id": None,
        "trace_url": None,
        "transformation_class_path": None,
        "bxtool_path": None,
        "written_files": [],
        "evaluation_runs": [],
        "iterations_completed": 0,
    }

    try:
        logger.info(f"Building model: {model_id}")
        model = build_model(model_id)

        logger.info(f"Building MDEAgent for model: {model_id}")
        agent_builder = build_mdeagent(
            workspace_path=workspace_path,
            benchmarx_path=None,
            model=model,
        )
        agent = agent_builder.compile(name=f"MDEAgent-Benchmark-{model_id.replace('/', '-')}-iter{iteration}")

        initial_state = MDEAgentState(
            source_model_path=SOURCE_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
            group_id="de.hofuniversity",
            artifact_id=f"MDEAgentBenchmark-{model_id.replace('/', '-')}",
            task_specification=TASK_SPECIFICATION,
            iteration=1,
        )

        callbacks = [langfuse_callback]
        logger.info(f"Running MDEAgent for model: {model_id}")
        output = await agent.ainvoke(
            initial_state, config={"callbacks": callbacks}, version="v2"
        )

        result["success"] = True
        result["trace_id"] = langfuse_callback.trace_id if hasattr(langfuse_callback, "trace_id") else None
        result["iterations_completed"] = output.value.get("iteration", 1)

        if output.value.get("transformation_class_path"):
            result["transformation_class_path"] = str(output.value["transformation_class_path"])
        if output.value.get("bxtool_path"):
            result["bxtool_path"] = str(output.value["bxtool_path"])
        if output.value.get("written_files"):
            result["written_files"] = [str(f) for f in output.value["written_files"]]
        if output.value.get("latest_evaluation_runs"):
            result["evaluation_runs"] = list(output.value["latest_evaluation_runs"].keys())

        logger.info(f"Successfully completed benchmark for model: {model_id}")

    except Exception as e:
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": str(e.__traceback__) if hasattr(e, "__traceback__") else None,
        }
        logger.exception(f"Benchmark failed for model {model_id}: {e}")

    finally:
        if langfuse_client:
            langfuse_client.flush()

    return result


def collect_trace_info(langfuse_client: Langfuse, model_id: str) -> dict[str, str]:
    """Collect trace information from LangFuse for a specific model benchmark."""
    trace_info = {"trace_id": None, "trace_url": None}

    try:
        agent_name = f"MDEAgent-Benchmark-{model_id.replace('/', '-')}"
        traces = langfuse_client.api.trace.list(name=agent_name, limit=1)

        if traces.data and len(traces.data) > 0:
            trace = traces.data[0]
            trace_info["trace_id"] = trace.id
            trace_info["trace_url"] = langfuse_client.get_trace_url(trace_id=trace.id)
            logger.info(f"Found trace for {model_id}: {trace_info['trace_url']}")
        else:
            logger.warning(f"No traces found for model {model_id}")

    except Exception as e:
        logger.warning(f"Failed to collect trace info for {model_id}: {e}")

    return trace_info


def run_benchmark(num_iterations: int = 5, max_concurrency: int = 3) -> list[dict[str, Any]]:
    """Run the full benchmark across all models using threading.

    All iterations across all models run in parallel, controlled by max_concurrency.

    Args:
        num_iterations: Number of times each model should be run (default: 5)
        max_concurrency: Maximum number of tasks to run in parallel (default: 3)

    Returns:
        List of result dictionaries, one per (model, iteration) pair
    """
    logger.info(f"Loading models from {MODELS_CSV}")
    models = load_models(MODELS_CSV)
    logger.info(f"Loaded {len(models)} models: {models[:5]}..." if len(models) > 5 else f"Loaded {len(models)} models: {models}")

    if not models:
        logger.error("No models found in models.csv")
        return []

    if not SOURCE_MODEL_PATH.exists():
        logger.error(f"Source model path not found: {SOURCE_MODEL_PATH}")
        return []
    if not TARGET_MODEL_PATH.exists():
        logger.error(f"Target model path not found: {TARGET_MODEL_PATH}")
        return []

    # Build task list: one task per (model, iteration)
    total_tasks = len(models) * num_iterations
    tasks = []
    for model_id in models:
        for iteration in range(1, num_iterations + 1):
            tasks.append((model_id, iteration))

    logger.info(f"Running {total_tasks} tasks with max_concurrency={max_concurrency}")

    results: list[dict[str, Any]] = []
    completed = 0
    completed_lock = threading.Lock()  # thread-safe completed counter
    results_lock = threading.Lock()  # thread-safe results collection

    def _run_and_track(model_id: str, iteration: int) -> dict[str, Any]:
        """Run a single task, write results to its own file, track progress thread-safely."""
        nonlocal completed
        safe_name = model_id.replace("/", "-")
        output_path = RESULTS_DIR / f"{safe_name}_iter{iteration}.json"

        try:
            result = _run_single_iteration_sync(model_id, iteration)

        except Exception as e:
            logger.exception(f"Unexpected error for {model_id} iter {iteration}: {e}")
            result = {
                "model_id": model_id,
                "iteration": iteration,
                "workspace_path": None,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "trace_id": None,
                "trace_url": None,
                "transformation_class_path": None,
                "bxtool_path": None,
                "written_files": [],
                "evaluation_runs": [],
                "iterations_completed": 0,
            }

        # Common: write file, log progress, collect results
        with completed_lock:
            completed += 1
            status = "✓ SUCCESS" if result["success"] else f"✗ FAILED: {result.get('error', {}).get('message', 'Unknown error')}"
            logger.info(f"[Run {completed}/{total_tasks}] {model_id} (iter {iteration}): {status}")

        with results_lock:
            results.append(result)

        # Write results to the iteration-specific file (thread-safe: each task has its own file)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"✓ {model_id} iter {iteration}: saved to {output_path}")
        except Exception as e:
            logger.exception(f"Failed to write result for {model_id} iter {iteration}: {e}")

        return result

    # Submit all tasks and wait for completion
    with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="bench") as executor:
        future_to_task = {
            executor.submit(_run_and_track, model_id, iteration): (model_id, iteration)
            for model_id, iteration in tasks
        }

        for future in as_completed(future_to_task):
            try:
                future.result()  # Re-raise exceptions
            except Exception as e:
                logger.exception(f"Future error: {e}")

    # Summary statistics
    logger.info("Benchmark complete.")
    success_count = sum(1 for r in results if r["success"])
    failure_count = len(results) - success_count
    logger.info(f"Summary: {success_count} succeeded, {failure_count} failed out of {len(results)} runs")

    return results


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="MDEAgent Benchmark")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of times each model should be run (default: 5)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of tasks to run in parallel (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("MDEAgent Benchmark Started")
    logger.info("=" * 80)

    results = run_benchmark(num_iterations=args.num_iterations, max_concurrency=args.max_concurrency)

    if results:
        success_count = sum(1 for r in results if r["success"])
        if success_count == len(results):
            logger.info("All benchmarks passed!")
            sys.exit(0)
        elif success_count > 0:
            logger.warning(f"Some benchmarks failed: {len(results) - success_count}/{len(results)}")
            sys.exit(1)
        else:
            logger.error("All benchmarks failed!")
            sys.exit(2)
    else:
        logger.error("No results collected!")
        sys.exit(3)


if __name__ == "__main__":
    main()
