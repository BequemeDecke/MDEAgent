import logging

from langgraph.graph.state import CompiledStateGraph

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.comprehension.state import ComprehensionState
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)


def create_comprehension_node(
    comprehension_subgraph: CompiledStateGraph[ComprehensionState],
):
    async def comprehension_node(state: MDEAgentState) -> MDEAgentState:
        """
        Calls the comprehension agent with the current workflow state.

        It needs a specific schema in order to parse the output of the subagent.
        It also gets the current results of the evaluations, which can be used to inform the comprehension agent about what has been tried already and what the results were.
        """
        logger.info("Starting comprehension node ... (For State look at LangFuse)")

        serialized_transformation = state.get("transformation_plan")
        if serialized_transformation is None:
            raise ValueError(
                "The comprehension node requires a transformation plan in the state."
            )
        transformation = TransformationPlan.from_dict(serialized_transformation)
        task_specification = state.get("task_specification", None)

        input_state = ComprehensionState(
            transformation_plan=transformation.to_dict(),
            task_specification=task_specification,
            latest_evaluation_runs={},  # TODO: Pass the latest evaluation runs of category "design"
            iteration=0,
        )

        # Call the comprehension subgraph with the current state
        output_state = await comprehension_subgraph.ainvoke(
            input=input_state, version="v2"
        )
        # Return the new state delta
        return MDEAgentState(
            transformation_plan=output_state.value.get("transformation_plan")
        )

    return comprehension_node
