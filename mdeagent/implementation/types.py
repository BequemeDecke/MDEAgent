from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun


class TransformationClass(TypedDict):
    """Represents a generated transformation class with its name, package, and code."""

    # Required
    name: str
    package: str
    path: Path

    # Generated code of the transformation class
    code: str | None


class TransformationClassGenerator(ABC):
    @abstractmethod
    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: list[EvaluationResult] | None = None,
    ) -> list[Path]:
        """Synthesizes the transformation class based on the provided transformation plan and an optional specific task.

        Args:
            transformation_plan (TransformationPlan): The transformation plan to use for generating the transformation class.
            transformation_class (TransformationClass): The transformation class to generate.
            specific_task (str | None, optional): An optional specific task to focus on when generating the transformation class. Defaults to None.
            evaluation_results (list[EvaluationResult] | None, optional): Optional evaluation results that can be used to inform the generation of the transformation class. Defaults to None.

        Returns:
            list[Path]: The paths of the generated transformation class files."""
