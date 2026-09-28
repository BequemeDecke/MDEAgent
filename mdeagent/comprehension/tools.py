import logging
from typing import Literal

from langchain.tools import ToolRuntime, tool

from mdeagent.comprehension import SerializedTransformationPlan, TransformationPlan

logger = logging.getLogger(__name__)

Section = Literal[
    "source_model_implementation",
    "target_model_implementation",
    "transformation_direction",
    "implementation_steps",
    "difficulties",
    "source_model_package",
    "target_model_package",
    "source_model_name",
    "target_model_name",
]


@tool
def read_transformation_plan(runtime: ToolRuntime, section: Section) -> str:
    """Tool to read the transformation plan from the runtime state. The transformation plan is stored in the runtime state as a serialized object, and this tool deserializes it and returns it as a TransformationPlan object.

    Args:
        runtime (ToolRuntime): The runtime of the agent, which contains the state where the transformation plan is stored.
        section (Section): The section of the transformation plan to read. Can be one of "source_model_implementation", "target_model_implementation", "transformation_direction", "implementation_steps", or "difficulties".

    Raises:
        ValueError: If the transformation plan is not found in the runtime state.

    Returns:
        str: The requested section of the transformation plan as a string.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    logger.debug(f"Serialized transformation plan: {serialized_tp}")
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)
    return tp.data.get(section, "Section not found in the transformation plan.")


@tool
def update_model_implementation(
    runtime: ToolRuntime,
    source_model_implementation: str | None = None,
    target_model_implementation: str | None = None,
):
    """Update the implementation details of the source and target models in the transformation plan. You can update either one or both implementations.

    Args:
        source_model_implementation (str | None, optional): The implementation details of the source model.
        target_model_implementation (str | None, optional): The implementation details of the target model.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    logger.debug(f"Serialized transformation plan: {serialized_tp}")
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)

    if (
        source_model_implementation is not None
        and target_model_implementation is not None
    ):
        tp.update_model_implementation(
            source_model_implementation, target_model_implementation
        )

    if source_model_implementation is not None and target_model_implementation is None:
        tp.update_model_implementation(
            source_model_implementation, tp.data["target_model_implementation"]
        )

    if source_model_implementation is None and target_model_implementation is not None:
        tp.update_model_implementation(
            tp.data["source_model_implementation"], target_model_implementation
        )


@tool
def update_transformation_direction(
    runtime: ToolRuntime, transformation_direction: str
):
    """Update the transformation direction in the transformation plan.

    Args:
        transformation_direction (str): The transformation direction to be updated in the transformation plan.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    logger.debug(f"Serialized transformation plan: {serialized_tp}")
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_transformation_direction(transformation_direction)


@tool
def update_difficulties(runtime: ToolRuntime, difficulties: str):
    """Update the identified difficulties in the transformation plan.

    Args:
        difficulties (str): The identified difficulties to be updated in the transformation plan.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    logger.debug(f"Serialized transformation plan: {serialized_tp}")
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_transformation_difficulties(difficulties)


@tool
def update_implementation_steps(runtime: ToolRuntime, implementation_steps: str):
    """Update the implementation steps in the transformation plan.

    Args:
        implementation_steps (str): The implementation steps in markdown to be updated in the transformation plan.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    logger.debug(f"Serialized transformation plan: {serialized_tp}")
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_implementation_steps(implementation_steps)


transformation_plan_tools = [
    read_transformation_plan,
    update_transformation_direction,
    update_difficulties,
    update_implementation_steps,
]
