import asyncio
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from mdeagent.comprehension import TransformationPlanData
from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationError, EvaluationResult, EvaluationRun
from mdeagent.preparation.prepare_workspace import (
    StructureFixStrategy,
    create_prepare_workspace_node,
    workspace_structure_is_clean,
)
from mdeagent.preparation.state import ModelImplementation, PreparationState
from mdeagent.util import copy_workspace, log_workspace_structure


# --------------------------------------------------------------------------- #
# Helpers for building evaluation runs/results used by the tests below.
# --------------------------------------------------------------------------- #
def _eval_run(results=None, errors=None) -> EvaluationRun:
    return EvaluationRun(
        started_at=datetime.now(tz=UTC),
        execution_time_ms=0,
        iteration=1,
        results=results or [],
        errors=errors or [],
    )


def _eval_result(success: bool) -> EvaluationResult:
    return EvaluationResult(
        content="some content",
        metadata={"success": success, "include_in_report": False},
    )


def _clean_workspace_structure_run() -> EvaluationRun:
    """A WorkspaceStructureEvaluation run that reported no problems."""
    return _eval_run()


def _dirty_workspace_structure_run() -> EvaluationRun:
    """A WorkspaceStructureEvaluation run that reported a problem (success=False)."""
    return _eval_run(results=[_eval_result(success=False)])


class TestWorkspaceStructureIsClean(TestCase):
    """Unit tests for the eval-result-based ``workspace_structure_is_clean`` helper
    that replaces the former filesystem-based ``is_workspace_structure_correct``.
    """

    def test_clean__no_evaluation_results_returns_false(self):
        # No evaluation results at all -> conservative default: not clean.
        self.assertFalse(workspace_structure_is_clean(PreparationState()))

    def test_clean__workspace_structure_run_missing_returns_false(self):
        # Results present but no workspace_structure run -> not clean.
        state = PreparationState(
            latest_evaluation_runs={"tools_installed": _clean_workspace_structure_run()}
        )
        self.assertFalse(workspace_structure_is_clean(state))

    def test_clean__run_without_results_and_without_errors(self):
        state = PreparationState(
            latest_evaluation_runs={
                "workspace_structure": _clean_workspace_structure_run()
            }
        )
        self.assertTrue(workspace_structure_is_clean(state))

    def test_clean__run_with_success_true_result(self):
        state = PreparationState(
            latest_evaluation_runs={
                "workspace_structure": _eval_run(results=[_eval_result(success=True)])
            }
        )
        self.assertTrue(workspace_structure_is_clean(state))

    def test_clean__run_with_success_false_result(self):
        state = PreparationState(
            latest_evaluation_runs={
                "workspace_structure": _dirty_workspace_structure_run()
            }
        )
        self.assertFalse(workspace_structure_is_clean(state))

    def test_clean__run_with_error_returns_false(self):
        state = PreparationState(
            latest_evaluation_runs={
                "workspace_structure": _eval_run(
                    errors=[EvaluationError(message="boom", type="ValueError")]
                )
            }
        )
        self.assertFalse(workspace_structure_is_clean(state))

    def test_clean__missing_workspace_structure_returns_false(self):
        # If there is no workspace_structure run, the structure is not clean.
        state = PreparationState(
            latest_evaluation_runs={"other_run": _clean_workspace_structure_run()}
        )
        self.assertFalse(workspace_structure_is_clean(state))


class TestPrepareWorkspace(TestCase):
    def setUp(self):
        self.maxDiff = None
        
        def fix_structure_side_effect(state: PreparationState) -> PreparationState:
            """Mock fix strategy that creates the missing pom.xml."""
            workspace = state.get("workspace_path")
            artifact_id = state.get("artifact_id")
            group_id = state.get("group_id")
            if workspace and artifact_id and group_id:
                # Create parent pom.xml
                parent_pom_path = workspace / "pom.xml"
                parent_pom_path.parent.mkdir(parents=True, exist_ok=True)
                parent_pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>parent</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
</project>""")
                # Create child pom.xml
                child_pom_path = workspace / artifact_id / "pom.xml"
                child_pom_path.parent.mkdir(parents=True, exist_ok=True)
                child_pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
            return PreparationState()
        
        self.fix_strategy = Mock(spec=StructureFixStrategy)
        self.fix_strategy.fix_structure.side_effect = fix_structure_side_effect
        self.prepare_workspace_node = create_prepare_workspace_node(
            self.fix_strategy
        )
        self.fake_data = TransformationPlanData(
            iteration=0,
            source_model_package="de.example.mdeagent",
            target_model_package="de.example.mdeagent",
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties="",
            implementation_steps="",
        )
        self.template_path = Path.cwd() / "templates"

    def _mock_subprocess_run(self, args, **kwargs):
        """Helper to mock subprocess.run and create minimal Maven project structure."""
        import subprocess
        cwd = kwargs.get('cwd', Path.cwd())
        
        # Check if this is an archetype:generate call
        if 'archetype:generate' in args:
            # Extract artifactId from args
            artifact_id = None
            group_id = None
            for arg in args:
                if arg.startswith('-DartifactId='):
                    artifact_id = arg.split('=')[1]
                elif arg.startswith('-DgroupId='):
                    group_id = arg.split('=')[1]
            
            if artifact_id and group_id:
                # Create the child project structure
                child_path = Path(cwd) / artifact_id
                child_path.mkdir(parents=True, exist_ok=True)
                
                # Create pom.xml
                pom_path = child_path / "pom.xml"
                pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
                
                # Create src directory structure
                src_path = child_path / "src" / "main" / "java" / group_id.replace('.', '/')
                src_path.mkdir(parents=True, exist_ok=True)
                
                # Create App.java
                app_java = src_path / "App.java"
                app_java.write_text(f"package {group_id};\npublic class App {{}}")
        
        return subprocess.CompletedProcess(args=args, returncode=0)

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,  # Will be set per test
    )
    def test_prepare_workspace__given_folder_does_not_exist(
        self,
        mock_run: Mock,
    ):
        """
        This test checks if the workspace is created successfully if the given workspace folder does not exist
        """
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = (
                Path(temp_dir) / "workspace"
            )  # This folder is not created yet
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                # package_path="de.example.mdeagent",
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            # Check direct output
            self.assertIsInstance(
                output_state.get("transformation_plan"),
                TransformationPlan,
                "The output state should contain a transformation plan.",
            )
            self.assertIsInstance(
                output_state.get("bxtool_path"),
                Path,
                "The output state should contain the bxtool path.",
            )
            self.assertIsInstance(
                output_state.get("transformation_class_path"),
                Path,
                "The output state should contain the transformation class path.",
            )
            self.assertEqual(
                output_state.get("maven_project_path"),
                workspace_path / "mdeagent",
            )
            self.assertEqual(
                output_state.get("transformation_package_path"),
                "de.example.mdeagent",
                "The output state should contain the transformation package path.",
            )

            # Check if subprocess.run was called for creating child project with archetype and for validating
            self.assertEqual(mock_run.call_count, 2)

            # Check indirect output
            self.assertTrue(
                (Path(workspace_path) / "mdeagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(workspace_path) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(workspace_path) / "mdeagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__given_folder_exists(
        self,
        mock_run: Mock,
    ):
        """
        This test checks if the workspace is created successfully if the given workspace folder exists
        """
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                # package_path="de.example.mdeagent",
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            # Check direct output
            self.assertIsInstance(
                output_state.get("transformation_plan"),
                TransformationPlan,
                "The output state should contain a transformation plan.",
            )
            self.assertIsInstance(
                output_state.get("bxtool_path"),
                Path,
                "The output state should contain the bxtool path.",
            )
            self.assertIsInstance(
                output_state.get("transformation_class_path"),
                Path,
                "The output state should contain the transformation class path.",
            )
            self.assertEqual(
                output_state.get("maven_project_path"),
                Path(temp_dir) / "mdeagent",
            )
            self.assertEqual(
                output_state.get("transformation_package_path"),
                "de.example.mdeagent",
                "The output state should contain the transformation package path.",
            )
            # Check if subprocess.run was called for creating child project with archetype and for validating
            self.assertEqual(mock_run.call_count, 2)

            # Check indirect output
            self.assertTrue(
                (Path(temp_dir) / "mdeagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    def test_prepare_workspace__content_exists_structure_incorrect(
        self
    ):
        """
        This test checks if the StructureFixStrategy is invoked if the workspace folder exists but the structure is incorrect.
        Some strategies would be:
        - Delete the existing content and create the structure again
        - Move the existing content to a backup folder and create the structure again
        - Merge the existing content with the new structure (if possible)
        - Abort the operation and ask the user to fix the structure manually
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
            )

            # Create the workspace manually
            # There is no "pom.xml"
            (Path(temp_dir) / "mdeagent").mkdir(parents=True)
            (Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md").touch()
            (
                Path(temp_dir)
                / "mdeagent"
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "mdeagent"
            ).mkdir(parents=True)

            asyncio.run(self.prepare_workspace_node(input_state))
            self.fix_strategy.fix_structure.assert_called_once_with(input_state)

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__transformation_plan_exists(
        self,
        mock_run: Mock,
    ):
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake transformation plan in the workspace
            tp_path = Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md"
            tp_path.parent.mkdir(parents=True, exist_ok=True)
            tp_path.touch()
            tp = TransformationPlan.from_dict(
                {
                    "data": self.fake_data,
                    "parser": {
                        "type": "FileTransformationPlanParser",
                        "args": {"file_path": str(tp_path)},
                    },
                    "template": self.template_path,
                }
            )
            tp.update_iteration(1)

            # Create the rest of the necessary structure for the workspace
            # Parent pom.xml
            (Path(temp_dir) / "pom.xml").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>de.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
</project>""")
            
            # Child project directory with pom.xml
            child_pom_path = Path(temp_dir) / "mdeagent" / "pom.xml"
            child_pom_path.parent.mkdir(parents=True, exist_ok=True)
            child_pom_path.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>de.example</groupId>
    <artifactId>mdeagent</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
            
            # src directory
            (Path(temp_dir) / "mdeagent" / "src" / "main" / "java" / "de" / "example" / "mdeagent").mkdir(parents=True, exist_ok=True)

            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                # package_path="de.example.mdeagent",
                group_id="de.example",
                artifact_id="mdeagent",
            )

            output_state = asyncio.run(
                self.prepare_workspace_node(input_state)
            )
            actual_tp: TransformationPlan = output_state.get("transformation_plan")

            self.assertIsNotNone(
                actual_tp,
                "The output state should contain a transformation plan.",
            )

            self.assertEqual(
                actual_tp.to_dict(),
                tp.to_dict(),
                "The transformation plan in the output state should match the existing transformation plan.",
            )

            # Should call subprocess.run for validating the existing Maven project, but not for creating a new one
            self.assertEqual(mock_run.call_count, 1)

    def test_prepare_workspace__state_properties_missing(self):
        input_state = PreparationState(
            required_tools=[],
            workspace_path=None,  # Missing workspace path
            # package_path="de.example.mdeagent",
            group_id="de.example",
            artifact_id="mdeagent",
        )

        with self.assertRaises(ValueError):
            asyncio.run(self.prepare_workspace_node(input_state))

        input_state = PreparationState(
            required_tools=[],
            workspace_path=Path("/some/path"),
            group_id=None,
            artifact_id="mdeagent",
        )

        with self.assertRaises(ValueError):
            asyncio.run(self.prepare_workspace_node(input_state))

        input_state = PreparationState(
            required_tools=[],
            workspace_path=Path("/some/path"),
            group_id="de.example",
            artifact_id=None,
        )

        with self.assertRaises(ValueError):
            asyncio.run(self.prepare_workspace_node(input_state))

    def test_prepare_workspace__with_benchmarx_path_no_bxtool_adapter(self):
        """Test that no BxTool adapter is created when benchmarx_path is provided."""
        mock_run = Mock()
        mock_run.side_effect = self._mock_subprocess_run
        
        with patch("subprocess.run", mock_run), tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            benchmarx_path = Path(temp_dir) / "benchmarx" / "tool.jar"
            benchmarx_path.parent.mkdir(parents=True, exist_ok=True)
            benchmarx_path.touch()  # Create dummy file
            
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                benchmarx_path=benchmarx_path,  # BenchmarX path is set
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            # Check that bxtool_path is None when benchmarx_path is provided
            self.assertIsNone(
                output_state.get("bxtool_path"),
                "The bxtool_path should be None when benchmarx_path is provided.",
            )

            # Verify that no BxTool adapter file was created (name-agnostic:
            # the adapter name is derived from the model folder names, which are
            # not set in this test, so just assert no ``*BxToolAdapter.java``
            # file exists in the package path).
            package_dir = (
                workspace_path
                / "mdeagent"
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "mdeagent"
            )
            adapter_files = list(package_dir.glob("*BxToolAdapter.java"))
            self.assertEqual(
                adapter_files,
                [],
                "No BxToolAdapter.java file should be created when benchmarx_path is provided.",
            )

            # But transformation_class_path should still be set (user will implement it)
            self.assertIsInstance(
                output_state.get("transformation_class_path"),
                Path,
                "The transformation_class_path should still be set.",
            )

    # ------------------------------------------------------------------ #
    # Requirements: prepare_workspace uses the latest WorkspaceStructureEvaluation
    # results (instead of the former is_workspace_structure_correct) to decide
    # what to do.
    # ------------------------------------------------------------------ #

    def test_prepare_workspace__clean_workspace_does_nothing(self):
        """Anforderung 1: an existing workspace whose latest
        ``WorkspaceStructureEvaluation`` is clean must not be touched.
        ``prepare_workspace`` returns early and only advances the iteration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-empty workspace from a previous iteration. The actual contents
            # are irrelevant because the decision is based on the evaluation
            # results, not on a filesystem check.
            (Path(temp_dir) / "leftover.txt").write_text("leftover")

            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=2,
                latest_evaluation_runs={
                    "workspace_structure": _clean_workspace_structure_run()
                },
            )

            with patch("subprocess.run") as mock_run:
                output_state = asyncio.run(
                    self.prepare_workspace_node(input_state)
                )

            # The fix strategy must not be invoked ...
            self.fix_strategy.fix_structure.assert_not_called()
            # ... and no Maven subprocess must be executed.
            self.assertEqual(
                mock_run.call_count,
                0,
                "No Maven command must be executed when the workspace is clean.",
            )
            # Only the iteration counter is advanced; nothing else is set.
            self.assertEqual(
                output_state.get("iteration"),
                3,
                "The iteration counter must be advanced even when nothing is done.",
            )
            self.assertIsNone(
                output_state.get("transformation_plan"),
                "No transformation plan must be (re)loaded when the workspace is clean.",
            )
            self.assertIsNone(
                output_state.get("maven_project_path"),
                "No maven project path must be set when the workspace is clean.",
            )
            # The workspace is left untouched.
            self.assertTrue(
                (Path(temp_dir) / "leftover.txt").exists(),
                "The workspace must not be modified when it is clean.",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__not_clean_workspace_applies_fix_strategy(
        self, mock_run: Mock
    ):
        """Anforderung 2: an existing workspace whose latest
        ``WorkspaceStructureEvaluation`` reports problems must be repaired via
        the ``StructureFixStrategy``.
        """
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-empty workspace with an incorrect structure (no pom.xml).
            (Path(temp_dir) / "mdeagent").mkdir(parents=True)
            (Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md").touch()

            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=1,
                latest_evaluation_runs={
                    "workspace_structure": _dirty_workspace_structure_run()
                },
            )

            asyncio.run(self.prepare_workspace_node(input_state))

            self.fix_strategy.fix_structure.assert_called_once_with(input_state)

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__empty_workspace_creates_even_when_evaluation_reports_problems(
        self, mock_run: Mock
    ):
        """Anforderung 3: an empty workspace is created as usual. The empty check
        takes precedence over the evaluation results, so a 'not clean' evaluation
        must NOT trigger the fix strategy.
        """
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"

            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=0,
                latest_evaluation_runs={
                    "workspace_structure": _dirty_workspace_structure_run()
                },
            )

            output_state = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            # Empty workspace -> create, NOT fix strategy.
            self.fix_strategy.fix_structure.assert_not_called()
            self.assertEqual(
                output_state.get("maven_project_path"),
                workspace_path / "mdeagent",
            )
            self.assertTrue(
                (workspace_path / "pom.xml").exists(),
                "The parent pom.xml should be created for an empty workspace.",
            )
            self.assertTrue(
                (workspace_path / "mdeagent" / "TRANSFORMATION.md").exists(),
                "The TRANSFORMATION.md file should be created for an empty workspace.",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__empty_workspace_creates_even_when_evaluation_is_clean(
        self, mock_run: Mock
    ):
        """Anforderung 3: an empty workspace is created as usual even when the
        evaluation already reports a clean state (e.g. iteration == 0 where the
        conditional edge routes to prepare_workspace unconditionally).
        """
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"

            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=0,
                latest_evaluation_runs={
                    "workspace_structure": _clean_workspace_structure_run()
                },
            )

            output_state = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            self.fix_strategy.fix_structure.assert_not_called()
            self.assertEqual(
                output_state.get("maven_project_path"),
                workspace_path / "mdeagent",
            )

    # ------------------------------------------------------------------ #
    # The transformation class name (and the BxTool adapter name) are derived
    # deterministically from the source/target model folder names
    # (``<Source>To<Target>Transformation`` / ``<Source>To<Target>BxToolAdapter``).
    # ------------------------------------------------------------------ #

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__derives_transformation_class_name_from_model_paths(
        self, mock_run: Mock
    ):
        """The transformation class name is derived deterministically from the
        source/target model folder names (``<Source>To<Target>Transformation``)."""
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                source_model=ModelImplementation(
                    name="Families",
                    path=Path(temp_dir) / "Families",
                    implementation=None,
                ),
                target_model=ModelImplementation(
                    name="Persons",
                    path=Path(temp_dir) / "Persons",
                    implementation=None,
                ),
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            self.assertEqual(
                output_state.get("transformation_class_path").name,
                "FamiliesToPersonsTransformation.java",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__derives_bxtool_adapter_name_from_model_paths(
        self, mock_run: Mock
    ):
        """The BxTool adapter file name follows ``<Source>To<Target>BxToolAdapter``."""
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                source_model=ModelImplementation(
                    name="Families",
                    path=Path(temp_dir) / "Families",
                    implementation=None,
                ),
                target_model=ModelImplementation(
                    name="Persons",
                    path=Path(temp_dir) / "Persons",
                    implementation=None,
                ),
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            bxtool_path = output_state.get("bxtool_path")
            self.assertIsNotNone(bxtool_path)
            self.assertEqual(bxtool_path.name, "FamiliesToPersonsBxToolAdapter.java")
            self.assertTrue(
                bxtool_path.exists(),
                "The BxTool adapter file should be created with the derived name.",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__naming_defaults_when_models_missing(
        self, mock_run: Mock
    ):
        """When no source/target model is set, the names fall back to
        ``UnknownToUnknown...`` so the naming never fails."""
        mock_run.side_effect = self._mock_subprocess_run

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
            )

            output_state: PreparationState = asyncio.run(
                self.prepare_workspace_node(input_state)
            )

            self.assertEqual(
                output_state.get("transformation_class_path").name,
                "UnknownToUnknownTransformation.java",
            )
            bxtool_path = output_state.get("bxtool_path")
            self.assertIsNotNone(bxtool_path)
            self.assertEqual(
                bxtool_path.name, "UnknownToUnknownBxToolAdapter.java"
            )

    def test_prepare_workspace__clean_workspace_skips_naming(self):
        """Early return: when the workspace is already clean no naming/Maven work
        happens (the transformation class path is left unset)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "leftover.txt").write_text("leftover")

            input_state = PreparationState(
                required_tools=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=1,
                latest_evaluation_runs={
                    "workspace_structure": _clean_workspace_structure_run()
                },
            )

            with patch("subprocess.run") as mock_run:
                output_state = asyncio.run(
                    self.prepare_workspace_node(input_state)
                )

            self.assertEqual(mock_run.call_count, 0)
            self.assertIsNone(output_state.get("transformation_class_path"))
            self.assertIsNone(output_state.get("bxtool_path"))



class TestMavenIntegration(TestCase):
    def setUp(self):
        if not shutil.which("mvn"):
            self.skipTest("Maven is not installed. Skipping Maven integration tests.")

    def test_prepare_workspace__maven_project_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_tools=["mvn"],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                source_model=ModelImplementation(
                    name="Families",
                    path=Path(temp_dir) / "Families",
                    implementation=None,
                ),
                target_model=ModelImplementation(
                    name="Persons",
                    path=Path(temp_dir) / "Persons",
                    implementation=None,
                ),
            )

            try:
                output = asyncio.run(
                    create_prepare_workspace_node(
                        fix_strategy=Mock(spec=StructureFixStrategy),
                    )(input_state)
                )

                self.assertEqual(
                    output.get("maven_project_path"),
                    Path(temp_dir) / "mdeagent",
                    "The output state should contain the maven project path.",
                )
            except Exception as e:
                log_workspace_structure(Path(temp_dir))
                copy_workspace(
                    Path(temp_dir),
                    Path.cwd() / ".mdeagent-workspace/prepare_workspace_test",
                )
                self.fail(
                    f"prepare_workspace_node raised an exception unexpectedly: {e}"
                )

            # Check for the correct structure
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                ).exists(),
                "The 'src/main/java/de/example/mdeagent' folder should be created in the workspace.",
            )

            # Check if the bxtool Java file is created with the deterministically
            # derived name (``<Source>To<Target>BxToolAdapter``).
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                    / "FamiliesToPersonsBxToolAdapter.java"
                ).exists(),
                "The bxtool Java file should be created in the package path.",
            )

            # Check if the AgentTransformationForEMF.java file is created
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                    / "AgentTransformationForEMF.java"
                ).exists(),
                "The AgentTransformationForEMF.java file should be created in the package path.",
            )

            # Check if the App.java file is deleted
            self.assertFalse(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "App.java"
                ).exists(),
                "The App.java file should be deleted.",
            )
