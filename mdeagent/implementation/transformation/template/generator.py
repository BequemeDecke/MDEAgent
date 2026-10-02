"""Template-based transformation class generator.

Uses a piecewise generation approach:
1. Derive source/target/decision types from the transformation plan (requirement 1).
2. Format evaluation results into readable text.
3. Generate fields/constructor and method bodies via LLM calls.
4. Combine pieces and write the transformation class file.

Returns written file paths (requirement 3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain.chat_models import BaseChatModel

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.transformation.template.prompts import (
    create_backward_body_prompt,
    create_fields_and_constructor_prompt,
    create_forward_body_prompt,
    create_synch_body_prompt,
)
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


class TemplateBasedGenerator(TransformationClassGenerator):
    """Template-based transformation class generator.

    The generator uses a piecewise approach: first the type names are derived
    from the transformation plan, evaluation results are formatted, and then
    the LLM is asked to produce fields/constructor and method bodies in
    parallel.  All pieces are combined into a single ``ImplementationTransformationSpec``
    which is rendered against the Jinja template and written to disk.

    It uses the `templates/transformation_class.jinja` template
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the resolver.

        Args:
            model: The LLM model to use for generating code.
        """
        self.model = model
        
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: list[EvaluationResult] | None = None,
    ) -> list[Path]:
        pass
