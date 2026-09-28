"""
Benchmark script for the MDEAgent.

This script reads models from benchmark/models.csv, creates a temporary workspace,
builds and runs the MDEAgent with each model, collects results via LangFuse,
and saves results to benchmark/results.json.
"""

import asyncio
import csv
import json
import logging
import sys
import tempfile
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
    """Load model IDs from the models CSV file.

    Args:
        csv_path: Path to models.csv

    Returns:
        List of model IDs (from the 'id' column)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Models file not found: {csv_path}")

    models = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig to handle BOM
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get("id", "").strip()
            if model_id:
                models.append(model_id)
    return models


def build_model(model_id: str) -> Any:
    """Build a LangChain chat model for the given model ID.

    Uses openai provider with the base URL and API key from config.

    Args:
        model_id: The model identifier (e.g., 'Qwen/Qwen3.8-27B-FP8')

    Returns:
        A BaseChatModel instance
    """
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


async def run_benchmark_model(
    model_id: str,
    workspace_path: Path,
    langfuse_client: Langfuse,
    langfuse_callback: LangfuseCallbackHandler,
    iteration: int = 1,
) -> dict[str, Any]:
    """Run a single benchmark iteration for one model.

    Args:
        model_id: The model ID to benchmark
        workspace_path: Path to the temporary workspace
        langfuse_client: Langfuse client for trace collection
        langfuse_callback: Langfuse callback handler
        iteration: Iteration number (for repeated runs)

    Returns:
        Dictionary with benchmark results
    """
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
        # 1. Build the model
        logger.info(f"Building model: {model_id}")
        model = build_model(model_id)

        # 2. Build the MDEAgent with the model
        logger.info(f"Building MDEAgent for model: {model_id}")
        agent_builder = build_mdeagent(
            workspace_path=workspace_path,
            benchmarx_path=None,
            model=model,
        )
        agent = agent_builder.compile(name=f"MDEAgent-Benchmark-{model_id.replace('/', '-')}-iter{iteration}")

        # 3. Create initial state
        initial_state = MDEAgentState(
            source_model_path=SOURCE_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
            group_id="de.hofuniversity",
            artifact_id=f"MDEAgentBenchmark-{model_id.replace('/', '-')}",
            task_specification=TASK_SPECIFICATION,
            iteration=1,
        )

        # 4. Run the agent with LangFuse callback
        callbacks = [langfuse_callback]
        logger.info(f"Running MDEAgent for model: {model_id}")
        output = await agent.ainvoke(
            initial_state, config={"callbacks": callbacks}, version="v2"
        )

        # 5. Collect results from output state
        result["success"] = True
        result["trace_id"] = langfuse_callback.trace_id if hasattr(langfuse_callback, "trace_id") else None
        result["iterations_completed"] = output.value.get("iteration", 1)

        # Extract state values
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
        # Always flush LangFuse
        if langfuse_client:
            langfuse_client.flush()

    return result


def collect_trace_info(langfuse_client: Langfuse, model_id: str) -> dict[str, str]:
    """Collect trace information from LangFuse for a specific model benchmark.

    Args:
        langfuse_client: Langfuse client instance
        model_id: The model ID to find traces for

    Returns:
        Dictionary with trace_id and trace_url
    """
    trace_info = {"trace_id": None, "trace_url": None}

    try:
        # Search for traces by name (the compiled agent name contains model_id)
        # The agent name format is "MDEAgent-Benchmark-{model_id}"
        agent_name = f"MDEAgent-Benchmark-{model_id.replace('/', '-')}"

        traces = langfuse_client.api.trace.list(
            name=agent_name,
            limit=1,
        )

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


async def _run_single_iteration(
    model_id: str,
    iteration: int,
) -> dict[str, Any]:
    """Run a single benchmark iteration for one model with its own workspace.

    Each iteration gets its own temporary workspace and LangFuse callback.

    Args:
        model_id: The model ID to benchmark
        iteration: Iteration number

    Returns:
        Dictionary with benchmark results
    """
    # Create temporary workspace (will be deleted automatically)
    with tempfile.TemporaryDirectory(prefix="mdeagent-benchmark-") as temp_dir:
        workspace_path = Path(temp_dir)

        # Build LangFuse callback for this iteration
        langfuse_client_instance, langfuse_callback = build_langfuse_client()

        try:
            # Run the benchmark
            result = await run_benchmark_model(
                model_id=model_id,
                workspace_path=workspace_path,
                langfuse_client=langfuse_client_instance,
                langfuse_callback=langfuse_callback,
                iteration=iteration,
            )

            # Collect trace information from LangFuse
            trace_info = collect_trace_info(langfuse_client_instance, model_id)
            result["trace_id"] = trace_info.get("trace_id")
            result["trace_url"] = trace_info.get("trace_url")

            return result

        finally:
            # Ensure flush happens even if result collection fails
            langfuse_client_instance.flush()


async def run_benchmark(num_iterations: int = 5, max_concurrency: int = 3) -> list[dict[str, Any]]:
    """Run the full benchmark across all models.

    Each model is run `num_iterations` times in separate workspaces.
    Models run in parallel with controlled concurrency.

    Args:
        num_iterations: Number of times each model should be run (default: 5)
        max_concurrency: Maximum number of models to run in parallel (default: 3)

    Returns:
        List of result dictionaries, one per (model, iteration) pair
    """
    # 1. Load models
    logger.info(f"Loading models from {MODELS_CSV}")
    models = load_models(MODELS_CSV)
    logger.info(f"Loaded {len(models)} models: {models[:5]}..." if len(models) > 5 else f"Loaded {len(models)} models: {models}")

    if not models:
        logger.error("No models found in models.csv")
        return []

    # 3. Verify source model paths exist
    if not SOURCE_MODEL_PATH.exists():
        logger.error(f"Source model path not found: {SOURCE_MODEL_PATH}")
        return []
    if not TARGET_MODEL_PATH.exists():
        logger.error(f"Target model path not found: {TARGET_MODEL_PATH}")
        return []

    # 4. Run benchmark with parallelization
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    total_tasks = len(models) * num_iterations
    completed = 0

    async def run_with_semaphore(model_id: str, iteration: int) -> dict[str, Any]:
        """Run benchmark with concurrency limit."""
        nonlocal completed
        async with semaphore:
            logger.info(f"[Run {completed + 1}/{total_tasks}] Starting: {model_id} (iter {iteration})")
            result = await _run_single_iteration(
                model_id=model_id,
                iteration=iteration,
            )
            completed += 1
            status = "✓ SUCCESS" if result["success"] else f"✗ FAILED: {result.get('error', {}).get('message', 'Unknown error')}"
            logger.info(f"[Run {completed}/{total_tasks}] {model_id} (iter {iteration}): {status}")
            return result

    # Create all tasks: each (model, iteration) pair
    tasks = []
    for model_id in models:
        for iteration in range(1, num_iterations + 1):
            tasks.append(run_with_semaphore(model_id, iteration))

    # Execute all tasks concurrently (semaphore limits actual concurrency)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions from gather
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Unexpected error in benchmark task: {result}")
            processed_results.append({
                "model_id": "unknown",
                "iteration": 0,
                "success": False,
                "error": {"type": type(result).__name__, "message": str(result)},
                "trace_id": None,
                "trace_url": None,
                "workspace_path": None,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "transformation_class_path": None,
                "bxtool_path": None,
                "written_files": [],
                "evaluation_runs": [],
                "iterations_completed": 0,
            })
        else:
            processed_results.append(result)

    # 5. Save results to individual files per model
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    results_by_model: dict[str, list[dict[str, Any]]] = {}
    for result in processed_results:
        model_id = result["model_id"]
        results_by_model.setdefault(model_id, []).append(result)

    for model_id, model_results in results_by_model.items():
        # Sanitize model ID for filename: replace '/' with '-'
        safe_name = model_id.replace("/", "-")
        output_path = RESULTS_JSON.parent / f"results_{safe_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved {len(model_results)} results for {model_id} to {output_path}")

    logger.info("Benchmark complete.")

    # Summary statistics
    success_count = sum(1 for r in processed_results if r["success"])
    failure_count = len(processed_results) - success_count
    logger.info(f"Summary: {success_count} succeeded, {failure_count} failed out of {len(processed_results)} runs")

    return processed_results


def main():
    """Main entry point for the benchmark script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("MDEAgent Benchmark Started")
    logger.info("=" * 80)

    # Run the benchmark
    results = asyncio.run(run_benchmark())

    # Exit with appropriate code
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
