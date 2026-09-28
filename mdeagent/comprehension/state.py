from typing import TypedDict

from mdeagent.comprehension.plan import SerializedTransformationPlan
from mdeagent.evaluation.types import EvaluationRun


class ComprehensionState(TypedDict):
    """
    The state of the comprehension subgraph. This state contains information and data that is relevant and needed for the comprehension subgraph to function.

    Important! Keep this updated and consistent all the time. This is the state that is passed between the nodes of the comprehension subgraph, and it is important that all nodes can read and write to this state as needed.
    Important! Only direct nodes are allowed to manipulate this state.
    """

    # === Transformation Plan ===
    transformation_plan: SerializedTransformationPlan  # Has to be in the state
    task_specification: str # Has to be in the state
    # === Evaluation Results ===
    latest_evaluation_runs: dict[str, EvaluationRun]
    # === Tracking ===
    iteration: int


