from pathlib import Path
from typing import Literal

from langchain.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from mdeagent.evaluation.executor import EvaluationExecutor
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.implementation.bxtool.implement_bx_tool import (
    create_implement_bx_tool_node,
)
from mdeagent.implementation.evaluation.evaluate_transformation_implementation import (
    create_route_implementation,
)
from mdeagent.implementation.format_code import create_format_code_node
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.transformation.factory import (
    create_transformation_class_generator,
)
from mdeagent.implementation.transformation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.mapping import (
    implementation_to_java_files,
    implementation_to_maven_project,
)
from mdeagent.models import build_base_model
from mdeagent.util import with_transformation


def build_implementation_graph(
    evaluation_executor: EvaluationExecutor,
    workspace_path: Path,
    implementation_strategy: Literal[
        "deep_agent", "hybrid_agent", "template_based"
    ] = "deep_agent",
    model: BaseChatModel | None = None,
    benchmarx_path: Path | None = None,
) -> StateGraph:
    """Build the implementation graph for the MDE agent."""
    # 1. Build the base model
    if model is None:
        model = build_base_model()

    # 2. Create the nodes
    implement_transformation = create_implement_transformation_node(
        workspace=workspace_path,
        generator=create_transformation_class_generator(
            strategy=implementation_strategy,
            workspace=workspace_path,
            model=model,
        ),
    )
    format_code = create_format_code_node(
        workspace=workspace_path,
    )
    evaluate_implementation_base = create_evaluation_node(
        evaluation_executor=evaluation_executor,
        mapper={
            "file_existence": implementation_to_java_files,
            "java_compilation": implementation_to_maven_project,
        },
        execution_mode="specific",
    )
    route_implementation = create_route_implementation(
        integration_enabled=bool(benchmarx_path)
    )
    if benchmarx_path:
        implement_bx_tool = create_implement_bx_tool_node(
            llm=model,
            workspace=workspace_path,
            benchmarx_path=benchmarx_path,
        )

    # 3. Wrap the evaluation node with a transformation to increment the iteration count
    evaluate_implementation = with_transformation(
        node=evaluate_implementation_base,
        transform=lambda state: {**state, "iteration": state.get("iteration", 0) + 1},
    )

    # 4. Build the graph
    graph = StateGraph(ImplementationState)
    graph.add_node("implement_transformation", implement_transformation, initial=True)
    graph.add_node("format_code", format_code)
    graph.add_node("evaluate_implementation", evaluate_implementation)

    graph.add_edge(START, "implement_transformation")
    graph.add_edge("format_code", "evaluate_implementation")

    # BxTool adapter flow only when BenchmarX is being used
    if benchmarx_path:
        graph.add_node("implement_bx_tool", implement_bx_tool)
        graph.add_edge("implement_transformation", "implement_bx_tool")
        graph.add_edge("implement_bx_tool", "format_code")
        graph.add_conditional_edges(
            "evaluate_implementation",
            route_implementation,
            {
                "implementation_error": "implement_transformation",
                "integration_error": "implement_bx_tool",
                "max_iteration_reached": END,  # TODO: Terminate the workflow with building a failure message
                "implementation_success": END,
            },
        )
    else:
        graph.add_edge("implement_transformation", "format_code")
        # `integration_error` cannot occur when the bx tool integration is not
        # part of the graph (see `create_evaluate_transformation_implementation`),
        # so it is intentionally not part of the routing map here.
        graph.add_conditional_edges(
            "evaluate_implementation",
            route_implementation,
            {
                "implementation_error": "implement_transformation",
                "max_iteration_reached": END,  # TODO: Terminate the workflow with building a failure message
                "implementation_success": END,
            },
        )

    return graph
