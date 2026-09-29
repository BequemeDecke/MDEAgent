import logging

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import TransformationClass
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)


def create_implementation_node(
    agent: CompiledStateGraph, benchmarx_path: str | None = None
):
    """Creates a function that calls the implementation agent with the necessary state and returns the updated state after the implementation agent has done its work.

    Args:
        agent (CompiledStateGraph): The implementation subgraph
        benchmarx_path (str | None): The path to the BenchmarX tool, if used. If None, the bxtool_path from the state will be used.
    """

    async def implementation_node(state: MDEAgentState) -> MDEAgentState:
        logger.info("Starting implementation node ... (For State look at LangFuse)")

        serialized_tp = state.get("transformation_plan")
        if serialized_tp is None:
            raise ValueError(
                "The Transformation Plan is required for the implementation agent!"
            )
        transformation_class_path = state.get("transformation_class_path")
        if transformation_class_path is None:
            raise ValueError(
                "Transformation class path is required for the implementation agent."
            )
        transformation_package_path = state.get("transformation_package_path")
        if transformation_package_path is None:
            raise ValueError(
                "Transformation package path is required for the implementation agent."
            )
        task_specification = state.get("task_specification", "")
        bxtool_path_from_state = state.get("bxtool_path")
        # When BenchmarX is being used, we don't need a separate bxtool adapter
        # The bxtool_path in state points to the adapter file location (used even with BenchmarX)
        if benchmarx_path is None and bxtool_path_from_state is None:
            raise ValueError(
                "BxTool file path is required for the implementation agent when BenchmarX is not being used."
            )

        maven_project_path = state.get("maven_project_path")
        if maven_project_path is None:
            raise ValueError(
                "Maven project path is required for the implementation agent."
            )

        # Create a TransformationClass object from the preparation phase data
        # The path and package information is stored in the TransformationClass

        transformation_class: TransformationClass = {
            "name": transformation_class_path.stem,
            "package": transformation_package_path,
            "path": transformation_class_path,
            "code": None,  # Will be populated by implement_transformation node
        }

        input_state = ImplementationState(
            transformation_plan=serialized_tp,
            transformation_class=transformation_class,
            task_specification=task_specification,
            maven_project_path=maven_project_path,
            bxtool_path=bxtool_path_from_state,  # Always provided per state definition
            written_files=[],
            latest_evaluation_runs={},
            iteration=0,
        )
        response: GraphOutput = await agent.ainvoke(input_state, version="v2")
        output_state: ImplementationState = response.value

        new_written_files = set(
            state.get("written_files", [])
        )  # Get existing written files from state
        new_written_files.update(
            output_state.get("written_files", [])
        )  # Add new written

        return MDEAgentState(written_files=list(new_written_files))

    return implementation_node
