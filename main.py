import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from mdeagent.agent import build_mdeagent
from mdeagent.state import MDEAgentState


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MDEAgent CLI.

    Usage:
        uv run main.py -s <source_model_path> -t <target_model_path> -g <group_id> -a <artifact_id> <task_specification>

    Required arguments:
        -s, --source-model-path    Path to the source model (e.g., Families metamodel)
        -t, --target-model-path    Path to the target model (e.g., Persons metamodel)
        -g, --group-id             Maven group ID (e.g., de.hofuniversity)
        -a, --artifact-id          Maven artifact ID (e.g., MDEAgentFamilyToPerson)
        task_specification         Description of the transformation task

    Optional arguments:
        --log-level                Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
        --use-langfuse             Enable LangFuse for monitoring and tracing
    """
    parser = argparse.ArgumentParser(
        description="Run the MDEAgent to generate a model transformation."
    )
    parser.add_argument(
        "--source-model-path",
        "-s",
        type=str,
        help="Path to the source model (e.g., Families metamodel).",
        required=True,
    )
    parser.add_argument(
        "--target-model-path",
        "-t",
        type=str,
        help="Path to the target model (e.g., Persons metamodel).",
        required=True,
    )
    parser.add_argument(
        "--group-id",
        "-g",
        type=str,
        help="Maven group ID (e.g., de.hofuniversity).",
        required=True,
    )
    parser.add_argument(
        "--artifact-id",
        "-a",
        type=str,
        help="Maven artifact ID (e.g., MDEAgentFamilyToPerson).",
        required=True,
    )
    parser.add_argument(
        "task_specification",
        type=str,
        help="Description of the transformation task (e.g., transform the Families model to the Persons model).",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    parser.add_argument(
        "--use-langfuse",
        action="store_true",
        help="Enable LangFuse for logging and monitoring.",
    )
    return parser.parse_args()


async def run_agent(
    source_model_path: Path,
    target_model_path: Path,
    group_id: str,
    artifact_id: str,
    task_specification: str,
    log_level: str = "INFO",
    use_langfuse: bool = False,
) -> dict:
    """Build and asynchronously invoke the MDEAgent.

    Args:
        source_model_path: Path to the source model directory.
        target_model_path: Path to the target model directory.
        group_id: Maven group ID for the generated transformation.
        artifact_id: Maven artifact ID for the generated transformation.
        task_specification: Description of the transformation task.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        use_langfuse: Whether to enable LangFuse monitoring.

    Returns:
        The agent's response dict.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Source model path: {source_model_path}")
    logger.info(f"Target model path: {target_model_path}")
    logger.info(f"Group ID: {group_id}")
    logger.info(f"Artifact ID: {artifact_id}")
    logger.info(f"Task specification: {task_specification}")

    mde_agent = build_mdeagent(workspace_path=source_model_path)
    logger.debug("MDEAgent built successfully.")

    config: dict = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        },
    }

    if use_langfuse:
        from mdeagent.monitoring import (
            build_langfuse_client,  # Dynamic import to avoid unnecessary dependency
        )

        langfuse_client, langfuse_handler = build_langfuse_client()
        config["callbacks"] = [langfuse_handler]
        logger.debug("LangFuse client and handler initialized.")

    initial_state = MDEAgentState(
        source_model_path=source_model_path,
        target_model_path=target_model_path,
        group_id=group_id,
        artifact_id=artifact_id,
        task_specification=task_specification,
        iteration=1,
    )
    logger.debug("Initial state created.")

    response = await mde_agent.ainvoke(initial_state, config, version="v2")
    logger.info(f"MDEAgent completed. Iteration: {response.value.get('iteration', 'N/A')}")

    if use_langfuse:
        langfuse_client.flush()  # Ensure all events are sent to LangFuse

    return response


# --- Main Execution ---
def main():
    args = parse_arguments()

    # Validate that task_specification is provided (required)
    if not args.task_specification:
        print("Error: task_specification is required.")
        print("Usage: uv run main.py -s <source_model_path> -t <target_model_path> -g <group_id> -a <artifact_id> <task_specification>")
        sys.exit(1)

    asyncio.run(
        run_agent(
            source_model_path=Path(args.source_model_path),
            target_model_path=Path(args.target_model_path),
            group_id=args.group_id,
            artifact_id=args.artifact_id,
            task_specification=args.task_specification,
            log_level=args.log_level,
            use_langfuse=args.use_langfuse,
        )
    )


if __name__ == "__main__":
    main()
