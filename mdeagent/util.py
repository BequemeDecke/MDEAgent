import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from mdeagent.config import Config

logger = logging.getLogger(__name__)


def with_transformation[T](
    node: Callable[[T], dict | T],
    transform: Callable[[T], dict],
) -> Callable[[T], dict]:
    """Wraps a node with a state transformation function.

    The transformation is applied **before** the node executes, allowing
    decoupling of concerns that would otherwise be handled inside the node itself.

    Common use cases:
    - Path normalization between different components with different path conventions
    - Iteration control (incrementing iteration counters before node execution)
    - Injecting or stripping metadata before/after node processing

    The returned function merges the transform result with the node result
    (dict union, right-to-left: node values override transform values).

    Parameters
    ----------
    node : callable
        An async or sync node function: ``state -> dict | T``
    transform : callable
        A state transformation function: ``state -> dict``

    Returns
    -------
    callable
        An async function that applies ``transform`` to the incoming state,
        passes the result to ``node``, and returns the merged state update.

    Example
    -------
    >>> def increment_iteration(state):
    ...     return {"iteration": state.get("iteration", 0) + 1}
    >>>
    >>> graph.add_node(
    ...     "step",
    ...     with_transformation(some_node, increment_iteration),
    ... )
    """

    async def wrapped(state: T) -> dict:
        # Step 1: Transform the incoming state
        transformed_state = transform(state)

        # Step 2: Execute the node with the transformed state
        node_result = await node(transformed_state)

        # Step 3: Merge results — node values override transform values
        merged: dict = {**transformed_state, **node_result}

        return merged

    return wrapped


def get_all_namespaces(filename: Path):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=["start-ns"])])
    return namespaces


def log_workspace_structure(workspace: Path):
    """
    Log the structure of the workspace directory for debugging purposes.
    """
    logger.debug(f"Workspace structure at {workspace}:")
    for path in workspace.rglob("*"):
        logger.debug(f" - {path.relative_to(workspace)}")


def copy_workspace(workspace: Path, destination: Path):
    """
    Copy the entire workspace directory to a new destination.
    """
    import shutil

    if destination.exists():
        logger.warning(
            f"Destination {destination} already exists. It will be overwritten."
        )
        shutil.rmtree(destination)

    shutil.copytree(workspace, destination)
    logger.info(f"Workspace copied from {workspace} to destination: {destination}.")


def limit_text(text: str, limit: int = 100) -> str:
    """
    Limit the length of a text string to a specified number of characters.
    If the text exceeds the limit, it will be truncated and an ellipsis ("...") will be appended.
    """
    if len(text) <= limit:
        return text
    else:
        return text[:limit] + "..."

def cancel_if_iteration_exceeded(max_iteration: int | None = None):
    """Checks if the current iteration exceeds the maximum allowed iterations. If it does, a RuntimeError is raised to cancel further iterations.

    Args:
        max_iteration (int | None, optional): The maximum number of iterations allowed. If None, the default value from the configuration will be used.

    Raises:
        RuntimeError: If the current iteration exceeds the maximum allowed iterations.

    Returns:
        _type_: A transformation function that can be used in the workflow to check and control the iteration count.
    """
    if max_iteration is None:
        max_iteration = Config.get_instance().AGENT_CONTROL.WORKFLOW_MAX_ITERATIONS

    def transformation(state: dict) -> dict:
        iteration: int = state.get("iteration")

        if iteration is not None and iteration >= max_iteration:
            logger.warning(
                f"Maximum iteration count reached ({iteration}/{max_iteration}). Cancelling further iterations."
            )
            raise RuntimeError(
                f"Maximum iteration count reached ({iteration}/{max_iteration})."
            )

        return {**state} 

    return transformation


def real_to_virtual(path: Path, real_root: Path, virtual_root: Path) -> Path:
    """Mappt einen realen Pfad auf den virtuellen Workspace."""
    real_path = path.resolve(strict=False)
    resolved_real_root = real_root.resolve(strict=False)

    try:
        relative_path = real_path.relative_to(resolved_real_root)
    except ValueError as error:
        raise ValueError(
            f"Pfad liegt außerhalb von {resolved_real_root}: {real_path}"
        ) from error

    return virtual_root / relative_path


def virtual_to_real(path: Path, virtual_root: Path, real_root: Path) -> Path:
    """Mappt einen virtuellen Pfad auf den realen Workspace."""
    virtual_path = path.resolve(strict=False)
    resolved_virtual_root = virtual_root.resolve(strict=False)

    try:
        relative_path = virtual_path.relative_to(resolved_virtual_root)
    except ValueError as error:
        raise ValueError(
            f"Pfad liegt außerhalb von {resolved_virtual_root}: {virtual_path}"
        ) from error

    return real_root / relative_path