from dataclasses import replace
from pathlib import Path

from langchain.agents import AgentState
from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from mdeagent.comprehension.plan import (
    SerializedTransformationPlan,
    TransformationPlan,
)
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)
from mdeagent.util import real_to_virtual, virtual_to_real


class TransformationClassAgentState(AgentState):
    """
    Expands the AgentState (messages) to include the transformation plan, the list of written files, the specific task, the transformation class, and the evaluation results.
    """

    written_files: list[str]
    specific_task: str | None
    transformation_plan: SerializedTransformationPlan
    transformation_class: TransformationClass
    evaluation_results: list[EvaluationResult]


INPUT_PROMPT_TEMPLATE = """
--- BEGIN VARIABLES ---
- Package of the source model: {source_model_package}
- Package of the target model: {target_model_package}
- Package of the transformation implementation: {transformation_package}
- Transformation class name: {transformation_class_name}
- Transformation class path: {transformation_class_path}
--- END VARIABLES ---

Use the read_transformation_plan tool to read the transformation plan and follow the steps described in it to implement the transformation class.

You specific task is: {specific_task}

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---
"""


def create_input_prompt(
    transformation_plan: TransformationPlan,
    transformation_class: TransformationClass,
    specific_task: str | None = None,
    evaluation_results: list[EvaluationResult] = None,
) -> TransformationClassAgentState:
    """
    Creates an input prompt for the coding agent based on the provided transformation plan, transformation class, specific task, and evaluation results.

    Args:
        transformation_plan (TransformationPlan): The transformation plan that describes the steps to be taken in order to perform the transformation.
        transformation_class (TransformationClass): The transformation class that contains the source and target model interfaces and the transformation rules.
        specific_task (str | None, optional): A specific task that the agent should focus on. Defaults to None.
        evaluation_results (list[EvaluationResult] | None, optional): Evaluation results from previous runs. Defaults to None.

    Returns:
        TransformationClassAgentState: The input prompt for the coding agent.
    """
    if evaluation_results is None:
        evaluation_results = []
    content = INPUT_PROMPT_TEMPLATE.format(
        source_model_package=transformation_plan.data["source_model_package"],
        target_model_package=transformation_plan.data["target_model_package"],
        transformation_package=transformation_class["package"],
        transformation_class_name=transformation_class["name"],
        transformation_class_path=transformation_class["path"],
        specific_task=specific_task or "No specific task provided.",
        evaluation_results_text=format_evaluation_results(evaluation_results)
        if evaluation_results
        else "No evaluation results provided.",
    )
    message = HumanMessage(content=content)
    return message


class TransformationClassAgentWrapper(TransformationClassGenerator):
    graph: CompiledStateGraph[TransformationClassAgentState]
    config: RunnableConfig
    workspace: Path
    _virtual_root: Path

    def __init__(
        self, workspace: Path, graph: CompiledStateGraph[TransformationClassAgentState]
    ):
        super().__init__()
        self.workspace = workspace
        self._virtual_root = Path("/")
        self.graph = graph
        self.config = {
            "configurable": {"thread_id": "transformation_class_agent"},
        }

    def virtualize_paths_in_class(
        self, transformation_class: TransformationClass
    ) -> TransformationClass:
        """
        Virtualizes the paths in the transformation class by replacing real paths with virtual paths.

        Args:
            transformation_class (TransformationClass): The transformation class to be virtualized.

        Returns:
            TransformationClass: The transformation class with virtualized paths.
        """
        v_class = transformation_class.copy()
        v_class["path"] = real_to_virtual(v_class["path"], self.workspace, self._virtual_root)
        return v_class

    def realize_paths_in_class(
        self, transformation_class: TransformationClass
    ) -> TransformationClass:
        """
        Realizes the paths in the transformation class by replacing virtual paths with real paths.

        Args:
            transformation_class (TransformationClass): The transformation class to be realized.

        Returns:
            TransformationClass: The transformation class with realized paths.
        """
        r_class = transformation_class.copy()
        r_class["path"] = virtual_to_real(r_class["path"], self._virtual_root, self.workspace)
        return r_class

    def virtualize_paths_in_evaluation_results(
        self, evaluation_results: list[EvaluationResult]
    ) -> list[EvaluationResult]:
        """
        Virtualizes the paths in the evaluation results by replacing real paths with virtual paths.

        Args:
            evaluation_results (list[EvaluationResult]): The evaluation results to be virtualized.

        Returns:
            list[EvaluationResult]: The evaluation results with virtualized paths.
        """
        virtualized = []
        for result in evaluation_results:
            metadata = result.metadata
            # Handle both dict and EvaluationMetadata (dataclass) cases
            if isinstance(metadata, dict):
                has_file = "file" in metadata
                if has_file:
                    metadata = dict(metadata)
                    metadata["file"] = str(
                        real_to_virtual(
                            Path(metadata["file"]), self.workspace, self._virtual_root
                        )
                    )
            else:
                # EvaluationMetadata dataclass - convert to dict first
                if hasattr(metadata, "__dict__"):
                    metadata_dict = dict(metadata.__dict__)
                else:
                    metadata_dict = {}
                if "file" in metadata_dict:
                    metadata_dict["file"] = str(
                        real_to_virtual(
                            Path(metadata_dict["file"]), self.workspace, self._virtual_root
                        )
                    )
                metadata = metadata_dict
            virtualized.append(replace(result, metadata=metadata))
        return virtualized

    def realize_paths_in_evaluation_results(
        self, evaluation_results: list[EvaluationResult]
    ) -> list[EvaluationResult]:
        """
        Realizes the paths in the evaluation results by replacing virtual paths with real paths.

        Args:
            evaluation_results (list[EvaluationResult]): The evaluation results to be realized.

        Returns:
            list[EvaluationResult]: The evaluation results with realized paths.
        """
        realized = []
        for result in evaluation_results:
            metadata = result.metadata
            # Handle both dict and EvaluationMetadata (dataclass) cases
            if isinstance(metadata, dict):
                has_file = "file" in metadata
                if has_file:
                    metadata = dict(metadata)
                    metadata["file"] = str(
                        virtual_to_real(
                            Path(metadata["file"]), self._virtual_root, self.workspace
                        )
                    )
            else:
                # EvaluationMetadata dataclass - convert to dict first
                if hasattr(metadata, "__dict__"):
                    metadata_dict = dict(metadata.__dict__)
                else:
                    metadata_dict = {}
                if "file" in metadata_dict:
                    metadata_dict["file"] = str(
                        virtual_to_real(
                            Path(metadata_dict["file"]), self._virtual_root, self.workspace
                        )
                    )
                metadata = metadata_dict
            realized.append(replace(result, metadata=metadata))
        return realized

    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: list[EvaluationResult] | None = None,
    ):
        # 1. Map real paths to virtual paths
        virtualized_class = self.virtualize_paths_in_class(transformation_class)
        virtualized_evaluation_results = self.virtualize_paths_in_evaluation_results(
            evaluation_results or []
        )

        input = TransformationClassAgentState(
            messages=[
                create_input_prompt(
                    transformation_plan,
                    virtualized_class,
                    specific_task=specific_task,
                    evaluation_results=virtualized_evaluation_results,
                )
            ],
            written_files=[],
            specific_task=specific_task,
            transformation_plan=transformation_plan.to_dict(),
            transformation_class=virtualized_class,
            evaluation_results=virtualized_evaluation_results,
        )
        output = await self.graph.ainvoke(input, config=self.config, version="v2")
        written_files = {
            virtual_to_real(Path(f), self._virtual_root, self.workspace) for f in output.value["written_files"]}
        written_files.add(transformation_class["path"])
        return list(written_files)
