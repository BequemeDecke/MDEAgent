import logging
from pathlib import Path

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.evaluation.utils import (
    filter_execution_results,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import (
    TransformationClassGenerator,
)

logger = logging.getLogger(__name__)

def create_implement_transformation_node(
    workspace: Path,
    generator: TransformationClassGenerator,
):
    """
    Creates the implement_transformation node for the implementation graph.

    This node calls the generator to synthesize the transformation class,
    then updates the transformation_class["code"] field with the generated code
    (so it can be used by implement_bx_tool if BenchmarX integration is enabled).
    """

    async def implement_transformation(
        state: ImplementationState,
    ) -> ImplementationState:
        transformation_class = state.get("transformation_class")
        if transformation_class is None:
            raise ValueError("Transformation class is not set in the state.")
        transformation_plan = state.get("transformation_plan")
        if transformation_plan is None:
            raise ValueError("Transformation plan is not set in the state.")
        latest_evaluation_runs = state.get("latest_evaluation_runs", {})
        if latest_evaluation_runs is None:
            raise ValueError("Latest evaluation runs are not set in the state.")
        task_specification = state.get("task_specification")
        if task_specification is None:
            raise ValueError("Task specification is not set in the state.")

        # Filter the evaluation results to only include those that are relevant for the current transformation class
        filtered_results: list[EvaluationResult] = filter_execution_results(latest_evaluation_runs)

        # Call the transformation class generator to synthesize the transformation class)
        written_files = await generator.synthesize_transformation_class(
            transformation_plan=TransformationPlan.from_dict(transformation_plan),
            transformation_class=transformation_class,
            specific_task=task_specification,
            evaluation_results=filtered_results,
        )

        # Update the transformation_class["code"] field with the generated code
        transformation_class_path = transformation_class.get("path")
        if transformation_class_path is None or not Path(transformation_class_path).exists():
            logger.error(f"Generated transformation class file does not exist: {transformation_class_path}")
        else:
            logger.info(f"Generated code for {transformation_class.get('name')}: {transformation_class.get('code', 'N/A')}")
            transformation_class["code"] = transformation_class_path.read_text()

        # Merge old and new written files
        written_files = [workspace / file for file in written_files]
        updated_written_files = list(set(state.get("written_files", [])) | set(written_files))

        return ImplementationState(
            transformation_class=transformation_class,
            written_files=updated_written_files,
        )

    return implement_transformation
