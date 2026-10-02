"""Tests for TemplateBasedGenerator — the template-based TransformationClassGenerator.

These tests verify:
1. Metadata is correctly extracted from the transformation plan.
2. Evaluation results are formatted into readable text.
3. The generator returns a list of Path objects.
4. The transformation class file is written correctly.
5. Missing fields default to sensible values.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.implementation.transformation.template.generator import (
    TemplateBasedGenerator,
)
from mdeagent.implementation.types import TransformationClass, TransformationClassGenerator

# ─── Fixtures ───────────────────────────────────────────────────────

def _make_plan(
    source_impl: str = "",
    target_impl: str = "",
    direction: str = "source to target",
    source_pkg: str = "com.example.source",
    target_pkg: str = "com.example.target",
) -> TransformationPlan:
    """Build a minimal TransformationPlan mock for testing."""
    mock_parser = MagicMock()
    mock_parser.to_dict.return_value = {"type": "FileTransformationPlanParser", "args": {}}

    # Mock template to avoid Jinja file loading in __str__
    mock_template = MagicMock()
    mock_template.render.return_value = "mocked plan"

    plan = TransformationPlan.__new__(TransformationPlan)
    plan.parser = mock_parser
    plan.template = mock_template
    plan._template_path = Path("templates")
    plan.data = {
        "source_model_package": source_pkg,
        "target_model_package": target_pkg,
        "iteration": 1,
        "source_model_implementation": source_impl,
        "target_model_implementation": target_impl,
        "transformation_direction": direction,
        "difficulties": "",
        "implementation_steps": "",
    }
    plan.to_dict = MagicMock(return_value={
        "data": plan.data.copy(),
        "parser": mock_parser.to_dict(),
        "template": Path("templates"),
    })
    return plan


def _make_tc(path: Path, name: str = "TestTransformation") -> TransformationClass:
    return TransformationClass(
        name=name,
        package="com.example.transform",
        path=path,
        code=None,
    )


def _make_ainvoke_and_parse_mock():
    """Create a mock for ainvoke_and_parse that returns correct Pydantic models."""
    from mdeagent.implementation.transformation.template.generator import (
        BackwardMethodBody,
        ForwardMethodBody,
        SynchMethodBody,
        TransformationFieldsAndConstructor,
    )

    async def fake_ainvoke_and_parse(llm, prompt, model_class):
        """Return the model class instance that matches the requested type."""
        if issubclass(model_class, TransformationFieldsAndConstructor):
            return TransformationFieldsAndConstructor(
                fields=[{"type": "String", "name": "name"}],
                constructor=None,
            )
        elif issubclass(model_class, ForwardMethodBody):
            return ForwardMethodBody(forward_body='System.out.println("forward");')
        elif issubclass(model_class, BackwardMethodBody):
            return BackwardMethodBody(backward_body='System.out.println("backward");')
        elif issubclass(model_class, SynchMethodBody):
            return SynchMethodBody(synch_body='System.out.println("synch");')
        # Fallback — shouldn't happen
        return ForwardMethodBody(forward_body="")

    return AsyncMock(side_effect=fake_ainvoke_and_parse)


# ─── Test Mixin for patching ────────────────────────────────────────

class _AsyncGeneratorTest:
    """Mixin providing a _patch_ainvoke helper for async generator tests."""

    def _patch_ainvoke(self, async_mock):
        """Patch ainvoke_and_parse in the generator module."""
        return patch(
            "mdeagent.implementation.transformation.template.generator.ainvoke_and_parse",
            async_mock,
        )


# ─── TestMetadataExtraction ─────────────────────────────────────────

class TestMetadataExtraction(TestCase):
    """Test that metadata is extracted from the plan, not from the LLM."""

    def test_extract_source_type_from_interface(self):
        """Source type should be extracted from 'public interface Foo'."""
        source_impl = "public interface Person {"
        target_impl = "public interface Address {"
        plan = _make_plan(source_impl=source_impl, target_impl=target_impl)

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.source_type, "Person")
        self.assertEqual(metadata.target_type, "Address")

    def test_extract_source_type_from_class(self):
        """Falls back to 'public class' when no interface is found."""
        source_impl = "public class Person {"
        target_impl = "public class Address {"
        plan = _make_plan(source_impl=source_impl, target_impl=target_impl)

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.source_type, "Person")
        self.assertEqual(metadata.target_type, "Address")

    def test_default_to_object_when_no_class_found(self):
        """When no class/interface name can be extracted, defaults to Object."""
        plan = _make_plan(source_impl="", target_impl="")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.source_type, "Object")
        self.assertEqual(metadata.target_type, "Object")

    def test_decision_type_for_bidirectional(self):
        """Bidirectional direction should yield 'Decision' as decision_type."""
        plan = _make_plan(direction="bidirectional")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.decision_type, "Decision")

    def test_decision_type_for_unidirectional(self):
        """Unidirectional direction should yield 'Object' as decision_type."""
        plan = _make_plan(direction="source to target")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.decision_type, "Object")

    def test_transformation_package_from_source(self):
        """Transformation package is derived from source model package."""
        plan = _make_plan(source_pkg="com.myapp.models")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.transformation_package, "com.myapp.models.transform")

    def test_transformation_package_defaults_when_empty(self):
        """Empty source package falls back to 'com.example.transform'."""
        plan = _make_plan(source_pkg="")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.transformation_package, "com.example.transform")


# ─── TestEvaluationResultFormatting ─────────────────────────────────

class TestEvaluationResultFormatting(TestCase):
    """Test that evaluation results are converted to readable text."""

    def test_none_returns_default_message(self):
        self.assertEqual(
            TemplateBasedGenerator._format_evaluation_results(None),
            "No evaluation results available.",
        )

    def test_empty_list(self):
        self.assertEqual(
            TemplateBasedGenerator._format_evaluation_results([]),
            "No evaluation results available.",
        )

    def test_list_with_results(self):
        """List with EvaluationResult objects is formatted correctly."""
        results = [
            EvaluationResult(
                content="File not found: Test.java",
                metadata={"success": False, "include_in_report": True},
            ),
        ]
        text = TemplateBasedGenerator._format_evaluation_results(results)
        self.assertIn("[FAILURE]", text)
        self.assertIn("File not found: Test.java", text)


# ─── TestFieldsToInfo ───────────────────────────────────────────────

class TestFieldsToInfo(TestCase):
    """Test the _fields_to_info helper."""

    def test_empty_fields(self):
        self.assertEqual(TemplateBasedGenerator._fields_to_info([]), "No fields defined.")

    def test_single_field(self):
        fields = [{"type": "String", "name": "name"}]
        self.assertEqual(
            TemplateBasedGenerator._fields_to_info(fields),
            "String name",
        )

    def test_multiple_fields(self):
        fields = [
            {"type": "String", "name": "name"},
            {"type": "int", "name": "age"},
        ]
        result = TemplateBasedGenerator._fields_to_info(fields)
        self.assertIn("String name", result)
        self.assertIn("int age", result)


# ─── TestSynthesizeTransformationClass ──────────────────────────────

class TestSynthesizeTransformationClass(_AsyncGeneratorTest, TestCase):
    """Test the full synthesize_transformation_class async method."""

    def test_returns_list_of_paths(self):
        """synthesize_transformation_class returns a list of Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "TestTransformation.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                result = asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(p, Path) for p in result))
            self.assertEqual(result[0], tc_path)

    def test_writes_java_file(self):
        """The generated file contains the class name and expected methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "TestTransformation.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path, name="TestTransformation")
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            self.assertTrue(tc_path.exists())
            content = tc_path.read_text()
            self.assertIn("TestTransformation", content)
            self.assertIn("forward", content)
            self.assertIn("backward", content)
            self.assertIn("synch", content)

    def test_creates_parent_directory(self):
        """Parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            nested_path = workspace / "src" / "main" / "java" / "TestTransformation.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(nested_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            self.assertTrue(nested_path.exists())

    def test_calls_llm_with_correct_prompts(self):
        """The generator calls ainvoke_and_parse with the right model classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            # Should be called 4 times: fields, forward, backward, synch
            self.assertEqual(mock_parse.call_count, 4)

            # Check that the correct model classes were passed
            # ainvoke_and_parse(llm, prompt, model_class) -> model_class is at index 2
            model_classes = [call[0][2] for call in mock_parse.call_args_list]
            from mdeagent.implementation.transformation.template.generator import (
                BackwardMethodBody,
                ForwardMethodBody,
                SynchMethodBody,
                TransformationFieldsAndConstructor,
            )

            self.assertIn(TransformationFieldsAndConstructor, model_classes)
            self.assertIn(ForwardMethodBody, model_classes)
            self.assertIn(BackwardMethodBody, model_classes)
            self.assertIn(SynchMethodBody, model_classes)

    def test_passes_specific_task_to_llm(self):
        """The specific_task parameter is included in LLM prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                        specific_task="Only implement forward transformation",
                    )
                )

            # ainvoke_and_parse(llm, prompt, model_class) -> prompt is at index 1
            prompts = [call[0][1] for call in mock_parse.call_args_list]
            self.assertTrue(
                any("Only implement forward transformation" in p for p in prompts),
                "specific_task not found in any prompt",
            )

    def test_evaluation_results_passed_to_llm(self):
        """Evaluation results are included in the LLM prompts."""
        eval_results = [
            EvaluationResult(
                content="Compilation error",
                metadata={"success": False, "include_in_report": True},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                        evaluation_results=eval_results,
                    )
                )

            # ainvoke_and_parse(llm, prompt, model_class) -> prompt is at index 1
            prompts = [call[0][1] for call in mock_parse.call_args_list]
            self.assertTrue(
                any("Compilation error" in p for p in prompts),
                "Evaluation result not found in any prompt",
            )


# ─── TestAinvokeAndParse ────────────────────────────────────────────

class TestAinvokeAndParse(TestCase):
    """Test the ainvoke_and_parse helper function."""

    def test_returns_pydantic_model(self):
        """ainvoke_and_parse returns a BaseModel instance."""
        from mdeagent.implementation.transformation.template.generator import (
            ForwardMethodBody,
            ainvoke_and_parse,
        )

        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(
            return_value=ForwardMethodBody(forward_body='System.out.println("test");')
        )
        mock_llm.with_structured_output.return_value = mock_structured

        result = asyncio.run(
            ainvoke_and_parse(mock_llm, "test prompt", ForwardMethodBody)
        )

        self.assertIsInstance(result, ForwardMethodBody)
        self.assertEqual(result.forward_body, 'System.out.println("test");')

    def test_calls_with_structured_output(self):
        """ainvoke_and_parse uses with_structured_output on the model."""
        from mdeagent.implementation.transformation.template.generator import (
            SynchMethodBody,
            ainvoke_and_parse,
        )

        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(
            return_value=SynchMethodBody(synch_body="synch body")
        )
        mock_llm.with_structured_output.return_value = mock_structured

        asyncio.run(
            ainvoke_and_parse(mock_llm, "test", SynchMethodBody)
        )

        # Verify with_structured_output was called with correct model class
        mock_llm.with_structured_output.assert_called_once()
        call_args = mock_llm.with_structured_output.call_args
        self.assertIs(call_args[0][0], SynchMethodBody)


# ─── TestTemplateRendering ──────────────────────────────────────────

class TestTemplateRendering(_AsyncGeneratorTest, TestCase):
    """Test that the Jinja template produces correct Java output."""

    def test_rendered_file_contains_package(self):
        """The rendered Java file contains the correct package declaration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path, name="MyTransform")
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            content = tc_path.read_text()
            self.assertIn("package com.example.transform", content)

    def test_rendered_file_contains_class_name(self):
        """The rendered Java file contains the correct class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "MySpecificTransformation.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path, name="MySpecificTransformation")
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            content = tc_path.read_text()
            self.assertIn("public class MySpecificTransformation", content)

    def test_rendered_file_contains_fields(self):
        """Fields from LLM response are rendered in the Java file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan(
                source_impl="public interface Person {",
                target_impl="public interface Address {",
            )
            tc = _make_tc(tc_path, name="PersonAddressTransform")
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            content = tc_path.read_text()
            # Fields from mock should appear in the file
            self.assertIn("private String name", content)

    def test_rendered_file_contains_method_bodies(self):
        """Method bodies from LLM responses are rendered in the Java file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            content = tc_path.read_text()
            self.assertIn('System.out.println("forward")', content)
            self.assertIn('System.out.println("backward")', content)
            self.assertIn('System.out.println("synch")', content)


# ─── TestBidirectionalTransformation ────────────────────────────────

class TestBidirectionalTransformation(_AsyncGeneratorTest, TestCase):
    """Test bidirectional transformation with Decision type."""

    def test_decision_type_is_decision(self):
        """Bidirectional direction sets decision_type to 'Decision'."""
        plan = _make_plan(direction="bidirectional")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.decision_type, "Decision")

    def test_unidirectional_decision_type_is_object(self):
        """Unidirectional direction sets decision_type to 'Object'."""
        plan = _make_plan(direction="source to target")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.decision_type, "Object")

    def test_bidirectional_rendering(self):
        """Bidirectional transformation renders with Decision type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "BidirectionalTransform.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan(
                source_impl="public interface Source {",
                target_impl="public interface Target {",
                direction="bidirectional",
            )
            tc = _make_tc(tc_path, name="BidirectionalTransform")
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            content = tc_path.read_text()
            self.assertIn("AgentTransformationForEMF<", content)
            self.assertIn("Source,", content)
            self.assertIn("Target,", content)
            self.assertIn("Decision", content)


# ─── TestEmptyAndEdgeCases ──────────────────────────────────────────

class TestEmptyAndEdgeCases(_AsyncGeneratorTest, TestCase):
    """Test edge cases like empty plans and missing data."""

    def test_empty_plan_defaults(self):
        """Empty plan data produces sensible defaults."""
        plan = _make_plan(source_impl="", target_impl="", direction="", source_pkg="")

        resolver = TemplateBasedGenerator()
        metadata = resolver._extract_metadata(plan)

        self.assertEqual(metadata.source_type, "Object")
        self.assertEqual(metadata.target_type, "Object")
        self.assertEqual(metadata.decision_type, "Object")
        self.assertEqual(metadata.transformation_package, "com.example.transform")

    def test_multiple_evaluation_results_formatted(self):
        """Multiple evaluation results are all included in the prompt."""
        results = [
            EvaluationResult(
                content="Compilation error in Test.java",
                metadata={"success": False},
            ),
            EvaluationResult(
                content="File missing: Utils.java",
                metadata={"success": False},
            ),
            EvaluationResult(
                content="All checks passed",
                metadata={"success": True},
            ),
        ]

        text = TemplateBasedGenerator._format_evaluation_results(results)
        self.assertIn("Compilation error in Test.java", text)
        self.assertIn("File missing: Utils.java", text)
        self.assertIn("All checks passed", text)
        self.assertIn("[FAILURE]", text)
        self.assertIn("[SUCCESS]", text)

    def test_writes_single_file_returned_in_list(self):
        """synthesize_transformation_class returns a list with exactly one Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tc_path = workspace / "Test.java"

            resolver = TemplateBasedGenerator()
            plan = _make_plan()
            tc = _make_tc(tc_path)
            mock_parse = _make_ainvoke_and_parse_mock()

            with self._patch_ainvoke(mock_parse):
                result = asyncio.run(
                    resolver.synthesize_transformation_class(
                        transformation_plan=plan,
                        transformation_class=tc,
                    )
                )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], tc_path)


# ─── TestConstructor ────────────────────────────────────────────────

class TestConstructor(TestCase):
    """Tests for the TemplateBasedGenerator constructor."""

    def test_resolver_exists_and_is_async(self):
        """The synthesize method should be async."""
        import inspect

        resolver = TemplateBasedGenerator()
        self.assertTrue(
            inspect.iscoroutinefunction(resolver.synthesize_transformation_class)
        )

    def test_interface_compliance(self):
        """TemplateBasedGenerator should implement TransformationClassGenerator."""
        self.assertIsInstance(
            TemplateBasedGenerator(), TransformationClassGenerator
        )

    def test_llm_is_optional(self):
        """TemplateBasedGenerator can be instantiated without an LLM."""
        resolver = TemplateBasedGenerator()
        self.assertIsNone(resolver.llm)

    def test_workspace_can_be_passed(self):
        """The workspace parameter sets the workspace attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            resolver = TemplateBasedGenerator(workspace=workspace)
            self.assertEqual(resolver.workspace, workspace)

    def test_llm_can_be_set_via_kwarg(self):
        """The llm parameter can be set via constructor."""
        mock_llm = MagicMock()
        resolver = TemplateBasedGenerator(llm=mock_llm)
        self.assertIs(resolver.llm, mock_llm)
