"""Template-based transformation class generator.

Uses a piecewise generation approach with structured LLM outputs:
1. Derive source/target/decision types from the transformation plan (requirement 1).
2. Format evaluation results into readable text (requirement 2).
3. Generate fields/constructor and method bodies via structured LLM calls.
4. Combine pieces and write the transformation class file (requirement 3).

Returns written file paths.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from jinja2 import Template as Jinja2Template
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel

from mdeagent.comprehension.plan import TransformationPlan

# ─── Template directory resolution ──────────────────────────────────
# Resolve relative to this module so it works regardless of cwd.
# generator.py → template/ → transformation/ → implementation/ → mdeagent/ → project root
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "templates"


# ─── Stub / Placeholder Classes ─────────────────────────────────────
from mdeagent.implementation.types import TransformationClassGenerator
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.transformation.template.prompts import (
    create_backward_body_prompt,
    create_fields_and_constructor_prompt,
    create_forward_body_prompt,
    create_synch_body_prompt,
)


# ─── Stub / Placeholder Classes ─────────────────────────────────────
# These classes are imported by mdeagent/implementation/__init__.py.
# They are kept as stubs for backwards compatibility with existing code
# (e.g. template_resolver.py, bxtool/implement_bx_tool.py) and will be
# replaced as the refactoring progresses.

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Protocol


class StructuredResponseParser(ABC):
    """Abstract base class for structured response parsers."""

    @abstractmethod
    def parse(self, response: str) -> Any:
        pass


class JsonParser(StructuredResponseParser):
    """JSON-based structured response parser."""

    def parse(self, response: str) -> Any:
        import json
        return json.loads(response)


class YamlLikeParser(StructuredResponseParser):
    """YAML-like structured response parser."""

    def parse(self, response: str) -> Any:
        try:
            import yaml
            return yaml.safe_load(response)
        except ImportError:
            return {"raw": response}


class FallbackParser(StructuredResponseParser):
    """Parser that falls back to returning raw text."""

    def parse(self, response: str) -> Any:
        return {"raw": response}


class TransformationClassSpec(BaseModel):
    """Spec for a transformation class (legacy placeholder)."""

    package_name: str = ""
    class_name: str = ""
    source_type: str = "Object"
    target_type: str = "Object"
    decision_type: str = "Object"


class ImplementationTransformationSpec(BaseModel):
    """Full implementation spec combining metadata, fields, and method bodies."""

    package_name: str = ""
    source_type: str = "Object"
    target_type: str = "Object"
    decision_type: str = "Object"
    transformation_package: str = "com.example.transform"
    fields: list[dict[str, str]] | None = None
    constructor: dict[str, Any] | None = None
    forward_body: str | None = None
    backward_body: str | None = None
    synch_body: str | None = None
    transform_source_to_target_body: str | None = None
    transform_target_to_source_body: str | None = None


class TransformationClassTemplateResolver:
    """Template resolver for rendering transformation classes (legacy placeholder)."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self._template_dir = template_dir or _TEMPLATE_DIR

    def get_raw_template(self) -> str:
        template_path = self._template_dir / "transformation_class.jinja"
        return template_path.read_text(encoding="utf-8")

    def render_template(
        self, spec: ImplementationTransformationSpec, class_name: str = "Transformation"
    ) -> str:
        from jinja2 import Template as Jinja2Template
        raw = self.get_raw_template()
        template = Jinja2Template(raw)
        context = {
            "package_name": spec.package_name,
            "transformation_package": spec.transformation_package,
            "class_name": class_name,
            "source_type": spec.source_type,
            "target_type": spec.target_type,
            "decision_type": spec.decision_type,
            "fields": spec.fields,
            "constructor": spec.constructor,
            "forward_body": spec.forward_body,
            "backward_body": spec.backward_body,
            "synch_body": spec.synch_body,
            "transform_source_to_target_body": spec.transform_source_to_target_body,
            "transform_target_to_source_body": spec.transform_target_to_source_body,
        }
        return template.render(**context)


class CodeGenerator:
    """Generic code generator base (legacy placeholder)."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace or Path.cwd()


class LlmClient(Protocol):
    """Protocol for LLM clients."""

    def invoke(self, prompt: str) -> Any:
        ...


# ─── Legacy sync helper (used by bxtool and other callers) ───────────


def invoke_and_parse(prompt: str, model_class: type[BaseModel]) -> BaseModel:
    """Sync wrapper around ainvoke_and_parse (legacy helper).

    This function exists for callers that use the synchronous bxtool path.
    """
    import asyncio

    # Create a new event loop for synchronous calls
    loop = asyncio.new_event_loop()
    try:
        # We need an LLM instance — this is a limitation of the sync wrapper.
        # In practice callers should use ainvoke_and_parse for async code.
        raise RuntimeError(
            "invoke_and_parse requires async context. "
            "Use ainvoke_and_parse(llm, prompt, model_class) instead."
        )
    finally:
        loop.close()


# ─── Pydantic Models for Structured LLM Outputs ─────────────────────


class TransformationClassMetadata(BaseModel):
    """Structured metadata extracted from the transformation plan."""

    package_name: str = ""
    source_type: str = "Object"
    target_type: str = "Object"
    decision_type: str = "Object"
    transformation_package: str = "com.example.transform"


class TransformationFieldsAndConstructor(BaseModel):
    """Fields and constructor for the transformation class."""

    fields: list[dict[str, str]] | None = None
    constructor: dict[str, Any] | None = None


class ForwardMethodBody(BaseModel):
    """Java code for the forward transformation method body."""

    forward_body: str | None = None


class BackwardMethodBody(BaseModel):
    """Java code for the backward transformation method body."""

    backward_body: str | None = None


class SynchMethodBody(BaseModel):
    """Java code for the synchronization method body."""

    synch_body: str | None = None


# ─── Helper Function ────────────────────────────────────────────────


async def ainvoke_and_parse(
    llm: BaseChatModel, prompt: str, model_class: type[BaseModel]
) -> BaseModel:
    """Invoke the LLM with structured output parsing.

    Uses ``BaseChatModel.with_structured_output`` to create a model-bound
    runnable that returns a ``BaseModel`` instance.

    Args:
        llm: The language model to invoke.
        prompt: The prompt string to send to the model.
        model_class: The Pydantic model class for structured output.

    Returns:
        An instance of *model_class* with the parsed content.
    """
    structured_model = llm.with_structured_output(model_class, include_raw=False)
    result = await structured_model.ainvoke(prompt)
    return result


# ─── Generator Class ────────────────────────────────────────────────


class TemplateBasedGenerator(TransformationClassGenerator):
    """Template-based transformation class generator.

    The generator uses a piecewise approach: first the type names are derived
    from the transformation plan, evaluation results are formatted, and then
    the LLM is asked to produce fields/constructor and method bodies in
    parallel. All pieces are combined into a single spec which is rendered
    against the Jinja template and written to disk.

    It uses the ``templates/transformation_class.jinja`` template.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        workspace: Path | None = None,
        *,
        template_dir: Path | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            llm: The LLM model to use for generating code.
            workspace: The workspace path (used for writing output files).
            template_dir: Optional override for the template directory.
                Falls back to ``templates/`` relative to this module.
        """
        self.llm = llm
        self.workspace = workspace
        self._template_dir = template_dir or _TEMPLATE_DIR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metadata(self, plan: TransformationPlan) -> TransformationClassMetadata:
        """Extract metadata (types and packages) from the transformation plan.

        Derives source/target type names from the *source_model_implementation*
        and *target_model_implementation* fields of the plan data.
        Derives the decision type from *transformation_direction*.
        """
        source_impl = plan.data.get("source_model_implementation", "")
        target_impl = plan.data.get("target_model_implementation", "")
        direction = plan.data.get("transformation_direction", "").lower()
        source_pkg = plan.data.get("source_model_package", "")

        source_type = self._extract_class_name(source_impl) or "Object"
        target_type = self._extract_class_name(target_impl) or "Object"
        decision_type = "Decision" if "bidirectional" in direction else "Object"
        transformation_package = (
            f"{source_pkg}.transform" if source_pkg else "com.example.transform"
        )

        return TransformationClassMetadata(
            source_type=source_type,
            target_type=target_type,
            decision_type=decision_type,
            transformation_package=transformation_package,
        )

    @staticmethod
    def _extract_class_name(impl: str) -> str | None:
        """Extract the class or interface name from implementation text."""
        for pattern in (r"public\s+interface\s+(\w+)", r"public\s+class\s+(\w+)"):
            match = re.search(pattern, impl)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _format_evaluation_results(
        results: list[EvaluationResult] | None,
    ) -> str:
        """Format evaluation results into readable text for the prompt."""
        if not results:
            return "No evaluation results available."
        return format_evaluation_results(results)

    @staticmethod
    def _fields_to_info(fields: list[dict[str, str]] | None) -> str:
        """Convert a fields list into a human-readable info string."""
        if not fields:
            return "No fields defined."
        return ", ".join(
            f"{f.get('type', 'Object')} {f.get('name', 'field')}" for f in fields
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: "TransformationClass",  # noqa: F821
        specific_task: str | None = None,
        evaluation_results: list[EvaluationResult] | None = None,
    ) -> list[Path]:
        """Synthesize the transformation class based on the provided plan.

        Args:
            transformation_plan: The transformation plan to use.
            transformation_class: Dict describing the target class (name, package, path).
            specific_task: Optional task focus passed to LLM prompts.
            evaluation_results: Optional evaluation results to inform generation.

        Returns:
            A list containing the single Path of the generated file.
        """
        # 1. Extract metadata from plan
        metadata = self._extract_metadata(transformation_plan)

        # 2. Format evaluation results
        eval_text = self._format_evaluation_results(evaluation_results)

        # 3. Read template once (used both in prompts and rendering)
        template_path = self._template_dir / "transformation_class.jinja"
        raw_template = template_path.read_text(encoding="utf-8")
        jinja_template = Jinja2Template(raw_template)

        # 4. Create prompts for all LLM calls
        fields_prompt = create_fields_and_constructor_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata,
            evaluation_results_text=eval_text,
        )

        fields_info = self._fields_to_info([])  # No fields yet — context placeholder

        forward_prompt = create_forward_body_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata,
            fields_info=fields_info,
            evaluation_results_text=eval_text,
        )

        backward_prompt = create_backward_body_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata,
            fields_info=fields_info,
            evaluation_results_text=eval_text,
        )

        synch_prompt = create_synch_body_prompt(
            task_specification=specific_task,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata,
            fields_info=fields_info,
            evaluation_results_text=eval_text,
        )

        # 5. Call LLM in parallel for independent parts
        fields_result, forward_result, backward_result, synch_result = (
            await asyncio.gather(
                ainvoke_and_parse(
                    self.llm, fields_prompt, TransformationFieldsAndConstructor
                ),
                ainvoke_and_parse(
                    self.llm, forward_prompt, ForwardMethodBody
                ),
                ainvoke_and_parse(
                    self.llm, backward_prompt, BackwardMethodBody
                ),
                ainvoke_and_parse(self.llm, synch_prompt, SynchMethodBody),
            )
        )

        # 6. Build template context
        fields = fields_result.fields or []
        constructor = fields_result.constructor

        context = {
            "package_name": transformation_class["package"],
            "transformation_package": metadata.transformation_package,
            "class_name": transformation_class["name"],
            "source_type": metadata.source_type,
            "target_type": metadata.target_type,
            "decision_type": metadata.decision_type,
            "fields": fields,
            "constructor": constructor,
            "forward_body": forward_result.forward_body,
            "backward_body": backward_result.backward_body,
            "synch_body": synch_result.synch_body,
            "transform_source_to_target_body": None,
            "transform_target_to_source_body": None,
        }

        rendered_code = jinja_template.render(**context)

        # 7. Write the generated file
        tc_path = transformation_class["path"]
        tc_path.parent.mkdir(parents=True, exist_ok=True)
        tc_path.write_text(rendered_code, encoding="utf-8")

        return [tc_path]
