from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mdeagent.evaluation import (
    EvaluationExecutor,
)
from mdeagent.evaluation.node import (
    create_evaluation_node,
)
from mdeagent.preparation.evaluate_preparation import create_evaluate_preparation
from mdeagent.preparation.explore_models import create_explore_models_node
from mdeagent.preparation.implementations.clear_workspace import ClearWorkspaceStrategy
from mdeagent.preparation.prepare_workspace import create_prepare_workspace_node
from mdeagent.preparation.state import PreparationState
from mdeagent.util import cancel_if_iteration_exceeded, with_transformation


def build_preparation_graph(
    evaluation_executor: EvaluationExecutor,
    benchmarx_path: Path | None = None,
    download_benchmarx: bool = False,
) -> StateGraph:
    """Builder for the preparation Subgraph of the MDEAgent. This graph is responsible for preparing the workspace, exploring the source and target models, and evaluating the preparation state.

    The graph implements an agent loop: it first evaluates the current
    preparation state and then, based on a conditional edge, either (re-)prepares
    the workspace via ``prepare_workspace`` or terminates. After
    ``prepare_workspace`` the optional ``download_benchmarx`` node and the
    ``explore_models`` node are executed before the flow returns to the
    evaluation node.

    Args:
        evaluation_executor (EvaluationExecutor): The evaluation executor that will be used to evaluate the preparation state.
        benchmarx_path (Path | None, optional): The path to the benchmark data. Defaults to None.
        download_benchmarx (bool, optional): Whether to include the download_benchmarx node in the graph. Defaults to False. This is experimental.

    Returns:
        StateGraph: The preparation subgraph of the MDEAgent.
    """
    # 1. Create the nodes
    prepare_workspace = create_prepare_workspace_node(
        fix_strategy=ClearWorkspaceStrategy(),
        benchmarx_path=benchmarx_path,
        download_benchmarx=download_benchmarx,
    )
    explore_models = create_explore_models_node()
    evaluate_preparation = create_evaluation_node(
        evaluation_executor=evaluation_executor,
        mapper={
            "workspace_structure": lambda state: {
                "workspace_path": state.get("workspace_path"),
                "artifact_id": state.get("artifact_id"),
                "package_path": f"{state.get('group_id')}.{state.get('artifact_id')}",
            },
            "tools_installed": lambda state: {
                "tools": state.get("required_tools", []),
            },
        },
        execution_mode="specific",
    )
    route_preparation_decision = create_evaluate_preparation()
    if download_benchmarx:
        from mdeagent.preparation.benchmarx import create_download_benchmarx_node

        benchmarx_node = create_download_benchmarx_node()

    # 2. Add iteration control to prepare_workspace node. If three iterations are exceeded, the flow will be canceled.
    prepare_workspace_iteration = with_transformation(
        node=with_transformation(
            node=prepare_workspace,
            transform=lambda state: {
                **state,
                "iteration": state.get("iteration", 0) + 1,
            },
        ),
        transform=cancel_if_iteration_exceeded(max_iteration=3),
    )

    # 3. Build the preparation graph
    graph = StateGraph(PreparationState)
    graph.add_node("evaluate_preparation", evaluate_preparation)
    graph.add_node("prepare_workspace", prepare_workspace_iteration)
    graph.add_node("explore_models", explore_models)

    graph.add_edge(START, "evaluate_preparation")
    graph.add_edge("explore_models", "evaluate_preparation")
    graph.add_conditional_edges(
        "evaluate_preparation",
        route_preparation_decision,
        {
            "structure_incomplete": "prepare_workspace",
            "workspace_prepared": END,
        },
    )

    if download_benchmarx:
        graph.add_node("download_benchmarx", benchmarx_node)

        graph.add_edge("prepare_workspace", "download_benchmarx")
        graph.add_edge("download_benchmarx", "explore_models")
    else:
        graph.add_edge("prepare_workspace", "explore_models")

    return graph
