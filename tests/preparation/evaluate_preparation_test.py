"""Tests for the preparation agent loop.

These tests cover the requirements of the agent-loop refactoring:

1. The validation node was renamed to ``evaluate_preparation`` (evaluation).
2. ``PreparationState`` carries an ``iteration`` field that is initialised to
   ``0`` by the preparation node.
3. The graph evaluates *first* and prepares the workspace afterwards.
4. A conditional edge routes either to ``prepare_workspace`` (first iteration or
   evaluation problems) or to ``END`` (everything went smoothly).
"""

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from mdeagent.evaluation import EvaluationExecutor, implementations
from mdeagent.evaluation.types import (
    Evaluation,
    EvaluationError,
    EvaluationResult,
    EvaluationRun,
)
from mdeagent.preparation.graph import build_preparation_graph
from mdeagent.preparation.evaluate_preparation import create_evaluate_preparation
from mdeagent.preparation.prepare_workspace import (
    StructureFixStrategy,
    create_prepare_workspace_node,
)
from mdeagent.preparation.state import PreparationState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(results=None, errors=None) -> EvaluationRun:
    return EvaluationRun(
        started_at=datetime.now(tz=UTC),
        execution_time_ms=0,
        iteration=1,
        results=results or [],
        errors=errors or [],
    )


def _result(success: bool) -> EvaluationResult:
    return EvaluationResult(
        content="some content",
        metadata={"success": success, "include_in_report": False},
    )


def _build_executor(evaluations: dict) -> EvaluationExecutor:
    return EvaluationExecutor(evaluations=evaluations)


def _default_executor() -> EvaluationExecutor:
    """Executor with the same evaluations the preparation graph uses."""
    return _build_executor(
        evaluations={
            "workspace_structure": {
                "evaluation": implementations.WorkspaceStructureEvaluation(),
                "evaluation_schema": implementations.WorkspaceStructureSchema,
            },
            "tools_installed": {
                "evaluation": implementations.ToolInstalledEvaluation(),
                "evaluation_schema": implementations.ToolInstalledSchema,
            },
        }
    )


class _StatefulEvaluation(Evaluation):
    """An evaluation whose ``run`` fails for the first ``fail_until`` calls and
    succeeds (returns no results / no errors) afterwards."""

    def __init__(self, fail_until: int = 0):
        self._fail_until = fail_until
        self._calls = 0

    async def setup(self) -> None:
        pass

    async def run(self, **kwargs) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        self._calls += 1
        if self._calls <= self._fail_until:
            return ([_result(success=False)], [])
        return ([], [])


# --------------------------------------------------------------------------- #
# 1. & 4. Conditional edge decision function
# --------------------------------------------------------------------------- #
class TestEvaluatePreparationDecision(TestCase):
    def setUp(self):
        self.decide = create_evaluate_preparation()

    def test_decision__is_callable(self):
        self.assertTrue(callable(self.decide))

    def test_decision__first_iteration_always_prepare_workspace(self):
        """Requirement 4: the first run must always route to prepare_workspace."""
        # Even when the evaluation already reported problems the first iteration
        # must go to prepare_workspace because the workspace is not prepared yet.
        state = PreparationState(
            iteration=0,
            latest_evaluation_runs={
                "workspace_structure": _run(results=[_result(success=False)]),
                "tools_installed": _run(errors=[EvaluationError(message="boom", type="ValueError")]),
            },
        )
        self.assertEqual(self.decide(state), "structure_incomplete")

    def test_decision__first_iteration_with_empty_results(self):
        state = PreparationState(iteration=0, latest_evaluation_runs={})
        self.assertEqual(self.decide(state), "structure_incomplete")

    def test_decision__subsequent_iteration_clean_results_routes_to_end(self):
        state = PreparationState(
            iteration=1,
            latest_evaluation_runs={
                "workspace_structure": _run(results=[_result(success=True)]),
                "tools_installed": _run(),
            },
        )
        self.assertEqual(self.decide(state), "workspace_prepared")

    def test_decision__subsequent_iteration_no_results_no_errors_routes_to_end(self):
        state = PreparationState(
            iteration=1,
            latest_evaluation_runs={"workspace_structure": _run()},
        )
        self.assertEqual(self.decide(state), "workspace_prepared")

    def test_decision__subsequent_iteration_with_errors_routes_to_prepare_workspace(self):
        state = PreparationState(
            iteration=1,
            latest_evaluation_runs={
                "workspace_structure": _run(errors=[EvaluationError(message="boom", type="ValueError")]),
            },
        )
        self.assertEqual(self.decide(state), "structure_incomplete")

    def test_decision__subsequent_iteration_with_failing_result_routes_to_prepare_workspace(self):
        """Preparation evaluations signal problems via results with success=False."""
        state = PreparationState(
            iteration=1,
            latest_evaluation_runs={
                "workspace_structure": _run(results=[_result(success=False)]),
            },
        )
        self.assertEqual(self.decide(state), "structure_incomplete")

    def test_decision__latest_results_with_failing_results_routes_to_prepare_workspace(self):
        """execution_mode='all' returns a dict; failing results route to prepare_workspace."""
        state = PreparationState(
            iteration=1,
            latest_evaluation_runs={"test_run": _run(results=[_result(success=False)])},
        )
        self.assertEqual(self.decide(state), "structure_incomplete")

        state_clean = PreparationState(
            iteration=1,
            latest_evaluation_runs={"test_run": _run()},
        )
        self.assertEqual(self.decide(state_clean), "workspace_prepared")

    def test_decision__max_iterations_routes_to_end(self):
        """The safety guard must terminate the loop even with failing results."""
        state = PreparationState(
            iteration=5,
            latest_evaluation_runs={
                "workspace_structure": _run(results=[_result(success=False)]),
            },
        )
        self.assertEqual(self.decide(state), "workspace_prepared")

    def test_decision__custom_max_iterations(self):
        # With failing results the decision keeps routing to prepare_workspace
        # until the (custom) max_iterations guard kicks in.
        state = PreparationState(
            iteration=2,
            latest_evaluation_runs={
                "workspace_structure": _run(results=[_result(success=False)]),
            },
        )
        # iteration (2) >= max_iterations (2) -> terminate.
        self.assertEqual(self.decide(state, max_iterations=2), "workspace_prepared")
        # iteration (2) < max_iterations (10) and there are problems -> retry.
        self.assertEqual(self.decide(state, max_iterations=10), "structure_incomplete")


# --------------------------------------------------------------------------- #
# 2. iteration field is initialised by the preparation node
# --------------------------------------------------------------------------- #
class TestPreparationStateIteration(TestCase):
    def test_state__iteration_is_optional_and_defaults_to_zero(self):
        # When not provided the key is simply absent -> .get defaults to 0.
        state = PreparationState()
        self.assertNotIn("iteration", state)
        self.assertEqual(state.get("iteration", 0), 0)


# --------------------------------------------------------------------------- #
# 3. Graph topology: evaluate first, then prepare_workspace (conditional)
# --------------------------------------------------------------------------- #
class TestPreparationGraphStructure(TestCase):
    def _compile(self, download_benchmarx: bool = False) -> CompiledStateGraph:
        return build_preparation_graph(
            evaluation_executor=_default_executor(),
            max_iterations=10,
            download_benchmarx=download_benchmarx,
        ).compile()

    def _edges(self, graph: CompiledStateGraph):
        return [
            (e.source, e.target, e.conditional, e.data)
            for e in graph.get_graph().edges
        ]

    def test_graph__nodes_contain_evaluate_preparation_not_validate(self):
        graph = self._compile()
        nodes = graph.get_graph().nodes
        self.assertIn("evaluate_preparation", nodes)
        self.assertIn("prepare_workspace", nodes)
        self.assertIn("explore_models", nodes)
        self.assertNotIn("validate_preparation", nodes)

    def test_graph__start_edges_to_evaluate_preparation(self):
        """Requirement 3: the evaluation runs first, not prepare_workspace."""
        graph = self._compile()
        edges = self._edges(graph)
        self.assertIn(("__start__", "evaluate_preparation", False, None), edges)
        # There must NOT be a direct START -> prepare_workspace edge anymore.
        self.assertFalse(
            any(s == "__start__" and t == "prepare_workspace" for s, t, _, _ in edges),
            "START must not edge directly into prepare_workspace anymore.",
        )

    def test_graph__conditional_edge_after_evaluate_preparation(self):
        """Requirement 4: conditional edge to prepare_workspace OR END."""
        graph = self._compile()
        edges = self._edges(graph)
        # Two conditional edges leave evaluate_preparation.
        conditional_targets = {
            t for s, t, cond, _ in edges if s == "evaluate_preparation" and cond
        }
        self.assertEqual(
            conditional_targets,
            {"prepare_workspace", "__end__"},
            "evaluate_preparation must conditionally route to prepare_workspace or END.",
        )
        # The 'workspace_prepared' branch must carry the 'end' data label.
        end_branches = [
            data for s, t, cond, data in edges
            if s == "evaluate_preparation" and cond and t == "__end__"
        ]
        self.assertIn("workspace_prepared", end_branches)

    def test_graph__prepare_workspace_edges_to_explore_models(self):
        graph = self._compile()
        edges = self._edges(graph)
        self.assertIn(("prepare_workspace", "explore_models", False, None), edges)
        self.assertIn(("explore_models", "evaluate_preparation", False, None), edges)

    def test_graph__download_benchmarx_inserted_between_prepare_and_explore(self):
        graph = self._compile(download_benchmarx=True)
        nodes = graph.get_graph().nodes
        self.assertIn("download_benchmarx", nodes)
        edges = self._edges(graph)
        self.assertIn(("prepare_workspace", "download_benchmarx", False, None), edges)
        self.assertIn(("download_benchmarx", "explore_models", False, None), edges)


# --------------------------------------------------------------------------- #
# prepare_workspace increments the iteration counter
# --------------------------------------------------------------------------- #
class TestPrepareWorkspaceIncrementsIteration(TestCase):
    def _mock_subprocess_run(self, args, **kwargs):
        import subprocess

        cwd = kwargs.get("cwd", Path.cwd())
        if "archetype:generate" in args:
            artifact_id = None
            group_id = None
            for arg in args:
                if arg.startswith("-DartifactId="):
                    artifact_id = arg.split("=")[1]
                elif arg.startswith("-DgroupId="):
                    group_id = arg.split("=")[1]
            if artifact_id and group_id:
                child_path = Path(cwd) / artifact_id
                child_path.mkdir(parents=True, exist_ok=True)
                (child_path / "pom.xml").write_text(
                    f"<?xml version='1.0'?><project><artifactId>{artifact_id}</artifactId>"
                    f"<groupId>{group_id}</groupId><version>1.0</version></project>"
                )
                src = child_path / "src" / "main" / "java" / group_id.replace(".", "/")
                src.mkdir(parents=True, exist_ok=True)
                (src / "App.java").write_text(f"package {group_id};\npublic class App {{}}")
        return subprocess.CompletedProcess(args=args, returncode=0)

    def _make_node(self):
        fix_strategy = Mock(spec=StructureFixStrategy)
        return create_prepare_workspace_node(fix_strategy)

    @patch("subprocess.run")
    def test_prepare_workspace__increments_iteration(self, mock_run: Mock):
        mock_run.side_effect = self._mock_subprocess_run
        node = self._make_node()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            workspace_path.mkdir()
            (workspace_path / "leftover.txt").write_text("leftover")  # ensure iterdir() is not empty
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                iteration=3,
                latest_evaluation_runs={
                    "workspace_structure": _run()
                },
            )
            output_state = asyncio.run(node(input_state))
            self.assertEqual(
                output_state.get("iteration"),
                4,
                "prepare_workspace must increment the iteration counter.",
            )

    @patch("subprocess.run")
    def test_prepare_workspace__defaults_iteration_to_one_when_missing(self, mock_run: Mock):
        mock_run.side_effect = self._mock_subprocess_run
        node = self._make_node()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            workspace_path.mkdir()
            (workspace_path / "leftover.txt").write_text("leftover")  # ensure iterdir() is not empty
            input_state = PreparationState(
                required_tools=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                latest_evaluation_runs={
                    "workspace_structure": _run()
                },
            )
            output_state = asyncio.run(node(input_state))
            self.assertEqual(
                output_state.get("iteration"),
                1,
                "When iteration is missing it should default to 0 and become 1.",
            )


# --------------------------------------------------------------------------- #
# End-to-end loop: evaluation fails -> prepare_workspace retries -> success
# --------------------------------------------------------------------------- #
class TestPreparationLoop(TestCase):
    """Verifies that the conditional edge drives a real retry loop.

    The heavy ``prepare_workspace`` (Maven) and ``explore_models`` nodes are
    replaced by lightweight mocks so the loop logic can be tested without mvn.
    The evaluation is a stateful mock that reports problems for the first cycle
    (ignored because iteration == 0) and the second cycle (-> retry) and reports
    success on the third cycle (-> END).
    """

    def _build_mocks(self):
        calls = {"prepare_workspace": 0, "explore_models": 0}

        def mock_prepare_workspace_node(*args, **kwargs):
            async def prepare_workspace(state: PreparationState) -> PreparationState:
                calls["prepare_workspace"] += 1
                # The graph wrapper adds the iteration increment, so we just
                # return the state unchanged.
                return {}
            return prepare_workspace

        def mock_explore_models_node(*args, **kwargs):
            async def explore_models(state: PreparationState) -> PreparationState:
                calls["explore_models"] += 1
                return {}
            return explore_models

        # Use the real evaluation node creation so the loop works
        def mock_create_evaluation_node(*args, **kwargs):
            from mdeagent.preparation.evaluate_preparation import create_evaluate_preparation
            from mdeagent.evaluation.node import create_evaluation_node as real_create
            return real_create(*args, **kwargs)

        return calls, mock_prepare_workspace_node, mock_explore_models_node, mock_create_evaluation_node

    def test_loop__retries_until_evaluation_succeeds(self):
        calls, mock_prepare, mock_explore, mock_eval = self._build_mocks()

        # Fail for the first 2 calls of each evaluation (covers the iter==0 and
        # iter==1 cycles), succeed from call 3 onwards (iter==2 cycle -> END).
        executor = _build_executor(
            evaluations={
                "workspace_structure": {
                    "evaluation": _StatefulEvaluation(fail_until=2),
                    "evaluation_schema": implementations.WorkspaceStructureSchema,
                },
                "tools_installed": {
                    "evaluation": _StatefulEvaluation(fail_until=2),
                    "evaluation_schema": implementations.ToolInstalledSchema,
                },
            }
        )

        with patch(
            "mdeagent.preparation.graph.create_prepare_workspace_node",
            side_effect=mock_prepare,
        ), patch(
            "mdeagent.preparation.graph.create_explore_models_node",
            side_effect=mock_explore,
        ), patch(
            "mdeagent.preparation.graph.create_evaluation_node",
            side_effect=mock_eval,
        ):
            graph = build_preparation_graph(
                evaluation_executor=executor,
                max_iterations=10,
            ).compile()

            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir) / "workspace"
                workspace.mkdir()
                initial_state = PreparationState(
                    workspace_path=workspace,
                    group_id="de.example",
                    artifact_id="mdeagent",
                    required_tools=["mvn"],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: PreparationState = output.value

        # The workspace should have been (re-)prepared exactly twice: once for
        # the first iteration and once for the retry triggered by the failing
        # evaluation on the second cycle.
        self.assertEqual(
            calls["prepare_workspace"],
            2,
            "prepare_workspace should be called twice (initial + one retry).",
        )
        # The loop must terminate (not run forever) and end with iteration == 2.
        self.assertEqual(output_state.get("iteration"), 2)
        # The final evaluation (cycle 3) reported no problems.
        latest = output_state.get("latest_evaluation_runs", {})
        for run in latest.values():
            self.assertEqual(run.errors, [])
            self.assertEqual(run.results, [])

    def test_loop__terminates_immediately_when_first_evaluation_is_clean(self):
        """If iteration == 0 the decision always routes to prepare_workspace, so
        even a 'clean' first evaluation still triggers exactly one preparation
        followed by a clean second evaluation -> END (no retry)."""
        calls, mock_prepare, mock_explore, mock_eval = self._build_mocks()

        executor = _build_executor(
            evaluations={
                "workspace_structure": {
                    "evaluation": _StatefulEvaluation(fail_until=0),
                    "evaluation_schema": implementations.WorkspaceStructureSchema,
                },
                "tools_installed": {
                    "evaluation": _StatefulEvaluation(fail_until=0),
                    "evaluation_schema": implementations.ToolInstalledSchema,
                },
            }
        )

        with patch(
            "mdeagent.preparation.graph.create_prepare_workspace_node",
            side_effect=mock_prepare,
        ), patch(
            "mdeagent.preparation.graph.create_explore_models_node",
            side_effect=mock_explore,
        ), patch(
            "mdeagent.preparation.graph.create_evaluation_node",
            side_effect=mock_eval,
        ):
            graph = build_preparation_graph(
                evaluation_executor=executor,
                max_iterations=10,
            ).compile()

            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir) / "workspace"
                workspace.mkdir()
                initial_state = PreparationState(
                    workspace_path=workspace,
                    group_id="de.example",
                    artifact_id="mdeagent",
                    required_tools=["mvn"],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: PreparationState = output.value
        self.assertEqual(calls["prepare_workspace"], 1)
        self.assertEqual(output_state.get("iteration"), 1)
