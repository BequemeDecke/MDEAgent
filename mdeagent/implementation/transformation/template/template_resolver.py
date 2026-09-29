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

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.transformation.template.generator import (
    BackwardMethodBody,
    ForwardMethodBody,
    ImplementationTransformationSpec,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
    ainvoke_and_parse,
)
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


class TemplateResolver(TransformationClassGenerator):
    """Template-based transformation class generator.

    The generator uses a piecewise approach: first the type names are derived
    from the transformation plan, evaluation results are formatted, and then
    the LLM is asked to produce fields/constructor and method bodies in
    parallel.  All pieces are combined into a single ``ImplementationTransformationSpec``
    which is rendered against the Jinja template and written to disk.
    """

    def __init__(self, llm: Any | None = None, workspace: Path | None = None) -> None:
        """Initialize the resolver.

        Args:
            llm: Optional language model for structured generation.
            workspace: Optional workspace path (reserved for future use).
        """
        self.llm = llm
        self._resolver = TransformationClassTemplateResolver()

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
        # 1. Derive metadata from the plan (requirement 1)
        metadata = self._extract_metadata(transformation_plan)

        # 2. Format evaluation results for the prompt
        eval_text = self._format_evaluation_results(evaluation_results)

        # 3. Load the raw template
        raw_template = self._resolver.get_raw_template()

        # 4. Generate fields & constructor first (needed as context for method bodies)
        fields_result = await ainvoke_and_parse(
            self.llm,
            create_fields_and_constructor_prompt(
                task_specification=specific_task or "",
                transformation_plan=str(transformation_plan),
                template=raw_template,
                metadata=metadata,
                evaluation_results_text=eval_text,
            ),
            TransformationFieldsAndConstructor,
        )

        # Build a compact field info string for context in body generation
        fields_info = self._fields_to_info(fields_result.fields)

        # 5. Generate method bodies in parallel
        (
            forward_body,
            backward_body,
            synch_body,
        ) = await asyncio.gather(
            ainvoke_and_parse(
                self.llm,
                create_forward_body_prompt(
                    task_specification=specific_task or "",
                    transformation_plan=str(transformation_plan),
                    template=raw_template,
                    metadata=metadata,
                    fields_info=fields_info,
                    evaluation_results_text=eval_text,
                ),
                ForwardMethodBody,
            ),
            ainvoke_and_parse(
                self.llm,
                create_backward_body_prompt(
                    task_specification=specific_task or "",
                    transformation_plan=str(transformation_plan),
                    template=raw_template,
                    metadata=metadata,
                    fields_info=fields_info,
                    evaluation_results_text=eval_text,
                ),
                BackwardMethodBody,
            ),
            ainvoke_and_parse(
                self.llm,
                create_synch_body_prompt(
                    task_specification=specific_task or "",
                    transformation_plan=str(transformation_plan),
                    template=raw_template,
                    metadata=metadata,
                    fields_info=fields_info,
                    evaluation_results_text=eval_text,
                ),
                SynchMethodBody,
            ),
        )

        # 6. Combine all pieces into the final spec
        combined_spec = ImplementationTransformationSpec(
            package_name=transformation_class["package"],
            source_type=metadata.source_type,
            target_type=metadata.target_type,
            decision_type=metadata.decision_type,
            transformation_package=metadata.transformation_package,
            fields=fields_result.fields or [],
            constructor=fields_result.constructor,
            forward_body=forward_body.forward_body,
            backward_body=backward_body.backward_body,
            synch_body=synch_body.synch_body,
            transform_source_to_target_body=None,  # will default to calling forward
            transform_target_to_source_body=None,  # will default to calling backward
        )

        # 7. Render the template with the generated specification
        class_name = transformation_class["name"]
        rendered_code = self._resolver.render_template(
            combined_spec, class_name=class_name
        )

        # 8. Write the generated code to a file
        target_path = Path(transformation_class["path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(rendered_code, encoding="utf-8")

        # 9. Return written files (requirement 3)
        return [target_path]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(plan: TransformationPlan) -> TransformationClassMetadata:
        """Derive type information from the transformation plan.

        source_type, target_type and decision_type are obtained from the plan
        data rather than asking the LLM to extract them.
        """
        data = plan.data

        source_type = TemplateResolver._extract_class_name(
            data.get("source_model_implementation", "") or ""
        )
        target_type = TemplateResolver._extract_class_name(
            data.get("target_model_implementation", "") or ""
        )
        # Default fallback when nothing could be extracted.
        source_type = source_type or "Object"
        target_type = target_type or "Object"

        transformation_direction = data.get("transformation_direction", "") or ""
        decision_type = (
            "Decision" if "bidirectional" in transformation_direction.lower()
            else "Object"
        )

        # Derive transformation package from source model package.
        source_pkg = data.get("source_model_package", "") or "com.example"
        transformation_package = f"{source_pkg}.transform"

        return TransformationClassMetadata(
            package_name="",
            source_type=source_type,
            target_type=target_type,
            decision_type=decision_type,
            transformation_package=transformation_package,
        )

    @staticmethod
    def _extract_class_name(text: str) -> str | None:
        """Try to extract a Java class/interface name from implementation text."""
        import re

        # public interface ClassName ...
        pattern = r"(?:public\s+)?(?:abstract\s+)?interface\s+(\w+)(?:\s*[<\[]|\s*\{|\s+extends)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # public class ClassName ...
        pattern = r"(?:public\s+)?class\s+(\w+)(?:\s*[<\[]|\s*\{|\s+extends)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _format_evaluation_results(results: list[EvaluationResult] | None) -> str:
        """Convert evaluation results to a readable string for the prompt."""
        if results is None or not results:
            return "No evaluation results available."
        return format_evaluation_results(results)

    @staticmethod
    def _fields_to_info(fields: list[dict[str, str]]) -> str:
        """Convert field list to a human-readable info string."""
        if not fields:
            return "No fields defined."
        return ", ".join(
            f"{f.get('type', 'Object')} {f.get('name', 'field')}" for f in fields
        )
