from pathlib import Path

from langchain.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from mdeagent.comprehension.agent import build_comprehension_agent
from mdeagent.comprehension.graph import build_comprehension_graph
from mdeagent.comprehension.node import create_comprehension_node
from mdeagent.config import Config
from mdeagent.evaluation import (
    EvaluationExecutor,
    FileExistenceEvaluation,
    FileExistenceSchema,
    JavaCompilationEvaluation,
    JavaCompilationSchema,
    PlanCompleteEvaluation,
    PlanCompleteSchema,
    ToolInstalledEvaluation,
    ToolInstalledSchema,
    WorkspaceStructureEvaluation,
    WorkspaceStructureSchema,
)
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.guardrails.transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from mdeagent.implementation.graph import build_implementation_graph
from mdeagent.implementation.node import create_implementation_node
from mdeagent.mapping import (
    mde_to_files,
    mde_to_maven_project,
    mde_to_tools,
    mde_to_transformation_plan,
    mde_to_workspace,
)
from mdeagent.models import build_base_model
from mdeagent.preparation.graph import build_preparation_graph
from mdeagent.preparation.node import create_preparation_node
from mdeagent.state import MDEAgentState
from mdeagent.tracking import control_iteration
from mdeagent.util import with_transformation


def build_mdeagent(
    workspace_path: Path,
    benchmarx_path: Path | None = None,
    download_benchmarx: bool = False,
    model: BaseChatModel | None = None,
) -> StateGraph[MDEAgentState]:
    config = Config.get_instance()
    if model is None:
        model = build_base_model()

    # 1. Initialize the core components of the MDEAgent
    check_transformation_iteration = create_check_transformation_iteration_function()
    agent_evaluator = EvaluationExecutor(
        evaluations={
            "workspace_structure": {
                "evaluation": WorkspaceStructureEvaluation(),
                "evaluation_schema": WorkspaceStructureSchema,
                "category": "preparation",
            },
            "tools_installed": {
                "evaluation": ToolInstalledEvaluation(),
                "evaluation_schema": ToolInstalledSchema,
                "category": "preparation",
            },
            "file_existence": {
                "evaluation": FileExistenceEvaluation(),
                "evaluation_schema": FileExistenceSchema,
                "category": "execution",
            },
            "java_compilation": {
                "evaluation": JavaCompilationEvaluation(),
                "evaluation_schema": JavaCompilationSchema,
                "category": "execution",
            },
            "plan_complete": {
                "evaluation": PlanCompleteEvaluation(),
                "evaluation_schema": PlanCompleteSchema,
                "category": "design",
            },
        }
    )

    # 2. Create the base nodes of the MDEAgent workflow
    preparation_node = create_preparation_node(
        preparation_agent=build_preparation_graph(
            evaluation_executor=agent_evaluator,
            max_iterations=config.AGENT_CONTROL.SUBGRAPH_MAX_ITERATIONS,
            benchmarx_path=benchmarx_path,
            download_benchmarx=download_benchmarx,
        ).compile(),
        workspace_path=workspace_path,
        required_tools=[
            "mvn",
            "java",
            "javac",
            "jar",
        ],
    )
    comprehension_node = create_comprehension_node(
        comprehension_subgraph=build_comprehension_graph(
            evaluation_executor=agent_evaluator,
            comprehension_agent=build_comprehension_agent(model=model),
            max_iterations=config.AGENT_CONTROL.SUBGRAPH_MAX_ITERATIONS,
        ).compile()
    )
    implementation_node = create_implementation_node(
        agent=build_implementation_graph(
            evaluation_executor=agent_evaluator,
            workspace_path=workspace_path,
            max_iterations=config.AGENT_CONTROL.SUBGRAPH_MAX_ITERATIONS,
            implementation_strategy=config.AGENT_CONTROL.TRANSFORMATION_IMPLEMENTATION_STRATEGY,
            model=model,
            benchmarx_path=benchmarx_path,
        ).compile()
    )
    evaluation_node = create_evaluation_node(
        evaluation_executor=agent_evaluator,
        mapper={
            "file_existence": mde_to_files,
            "java_compilation": mde_to_maven_project,
            "tools_installed": mde_to_tools,
            "workspace_structure": mde_to_workspace,
            "plan_complete": mde_to_transformation_plan,
        },
    )

    # 3. Wrap nodes within transformation functions to control iteration and state updates
    evaluation_incrementation_node = with_transformation(
        node=evaluation_node, transform=control_iteration
    )

    # 4. Build the StateGraph for the MDEAgent workflow
    builder = StateGraph(MDEAgentState)
    builder.add_node("preparation", preparation_node)
    builder.add_node("comprehension", comprehension_node)
    builder.add_node("implementation", implementation_node)
    builder.add_node("evaluation", evaluation_incrementation_node)

    builder.add_edge(START, "preparation")
    builder.add_edge("preparation", "comprehension")
    builder.add_edge("comprehension", "implementation")
    builder.add_edge("implementation", "evaluation")

    builder.add_conditional_edges(
        "evaluation",
        check_transformation_iteration,
        {
            "design_failed": "comprehension",
            "execution_failed": "implementation",
            "max_iteration_reached": END,
            "design_passed": END,
            "error": END,
        },
    )

    return builder
