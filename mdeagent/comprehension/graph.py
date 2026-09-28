import logging
from typing import Literal

from langchain.messages import HumanMessage
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph

from mdeagent.comprehension.state import ComprehensionState
from mdeagent.evaluation.executor import EvaluationExecutor
from mdeagent.evaluation.filter import IsErrorFilter
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.evaluation.pipefilter import EvaluationPipe
from mdeagent.util import cancel_if_iteration_exceeded, with_transformation

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
The specific requirements of the transformation by the user is:
--- BEGIN SPECIFIC TASK ---
{task_specification}
--- END SPECIFIC TASK ---

Use the following results to check if the transformation plan is complete and consistent:

--- BEGIN AUDIT RESULTS ---
{evaluation_results}
--- END AUDIT RESULTS ---
"""


def create_reflect_comprehension_node(comprehension_agent: CompiledStateGraph):
    async def reflect_comprehension(state: ComprehensionState) -> ComprehensionState:
        """
        Reflects on the current transformation plan and updates the state with the current iteration.
        """
        logger.debug("Reflecting on the current transformation plan ...")
        transformation = state.get("transformation_plan")
        task_specification = state.get(
            "task_specification",
            "No specific task provided. Think about the transformation plan and how to achieve the transformation!",
        )

        input_prompt = PROMPT_TEMPLATE.format(
            task_specification=task_specification,
            evaluation_results="\n".join(
                [str(run) for run in state.get("latest_evaluation_runs", {}).values()]
            ),
        )

        output = await comprehension_agent.ainvoke(
            input={
                "messages": [HumanMessage(content=input_prompt)],
                "transformation_plan": transformation,
            },
            version="v2",
        )

        return ComprehensionState(
            transformation_plan=output.value.get("transformation_plan")
        )

    return reflect_comprehension


ComprehensionEvaluationDecision = Literal["plan_complete", "plan_incomplete", "failure"]


def route_evaluation_decision(
    state: ComprehensionState,
) -> ComprehensionEvaluationDecision:
    """
    Routes the flow based on the evaluation results of the transformation plan.

    If the transformation plan is complete and consistent, the flow proceeds to the next stage.
    If the transformation plan is incomplete or inconsistent, the flow loops back to the reflect_comprehension node for further refinement.
    """
    evaluation_runs = state.get("latest_evaluation_runs", {})
    transformation_plan_run = evaluation_runs.get("plan_complete")

    if transformation_plan_run is None:
        return "failure"

    pipe = EvaluationPipe() | IsErrorFilter
    error_results = pipe.filter_results(transformation_plan_run.results)

    return "plan_complete" if len(error_results) == 0 else "plan_incomplete"


def build_comprehension_graph(
    evaluation_executor: EvaluationExecutor,
    comprehension_agent: CompiledStateGraph,
    max_iterations: int,
) -> StateGraph:
    """
    Builds the comprehension subgraph for the MDEAgent workflow.
    Nodes:
    - reflect_comprehension: This node is responsible for reflecting on the current transformation plan
    - evaluate_comprehension: This node evaluates the current transformation plan and provides feedback for improvement.
    Conditional Edges:
    - If the transformation plan is complete and consistent, the workflow proceeds to the next stage.
    - If the transformation plan is incomplete or inconsistent, the workflow loops back to the reflect_comprehension node for further refinement.
    """
    # 1. Create the nodes
    reflect_comprehension = create_reflect_comprehension_node(comprehension_agent)
    evaluate_comprehension = create_evaluation_node(
        evaluation_executor,
        mapper={
            "plan_complete": lambda state: {
                "transformation_plan": state.get("transformation_plan")
            }
        },
        execution_mode="specific",
    )

    # 2. Wrap reflect_comprehension node with iteration control
    reflect_comprehension_iteration = with_transformation(
        node=with_transformation(
            node=reflect_comprehension,
            transform=lambda state: {
                **state,
                "iteration": state.get("iteration", 0) + 1,
            },
        ),
        transform=cancel_if_iteration_exceeded(max_iteration=max_iterations),
    )

    # 3. Build the comprehension subgraph
    graph = StateGraph(state_schema=ComprehensionState)
    graph.add_node("reflect_comprehension", reflect_comprehension_iteration)
    graph.add_node("evaluate_comprehension", evaluate_comprehension)

    graph.add_edge(START, "reflect_comprehension")
    graph.add_edge("reflect_comprehension", "evaluate_comprehension")

    graph.add_conditional_edges(
        "evaluate_comprehension",
        route_evaluation_decision,
        {
            "plan_incomplete": "reflect_comprehension",
            "plan_complete": END,
        },
    )
    return graph
