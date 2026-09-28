import asyncio
import tempfile
from pathlib import Path
from typing import TypedDict
from unittest import TestCase

from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph

from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    SerializedTransformationPlan,
    TransformationPlan,
)
from mdeagent.tracking import control_iteration
from mdeagent.util import cancel_if_iteration_exceeded, with_transformation


class TestState(TypedDict):
    iteration: int
    transformation_plan: SerializedTransformationPlan | None
    codes: int


def create_dummy_node(return_code: int):
    async def dummy_node(state: TestState) -> TestState:
        new_codes = state.get("codes", 0) + return_code
        return TestState(codes=new_codes)

    return dummy_node


def build_graph(max_iteration: int, successfull_end: int) -> CompiledStateGraph:
    graph = StateGraph(state_schema=TestState)
    graph.add_node(
        "one",
        with_transformation(
            with_transformation(
                create_dummy_node(return_code=1), cancel_if_iteration_exceeded(max_iteration=max_iteration) # It can run for 4 iterations, but we set the limit to 2 for testing
            ),
            control_iteration,
        ),
    )
    graph.add_node("two", create_dummy_node(return_code=0))

    graph.add_edge(START, "one")
    graph.add_edge("one", "two")
    graph.add_conditional_edges(
        "two",
        lambda state: str(state.get("codes", 0) >= successfull_end),
        {"True": END, "False": "one"},
    )
    return graph.compile()


class TestIterationControl(TestCase):
    def test_update_iteration_and_transformation_plan(self):
        graph = build_graph(max_iteration=5, successfull_end=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(workspace / "TRANSFORMATION.md")
            )
            input_state = TestState(
                iteration=0, transformation_plan=transformation_plan.to_dict(), codes=0
            )
            output = asyncio.run(graph.ainvoke(input_state, version="v2"))

            self.assertEqual(output.value.get("iteration"), 3)
            self.assertEqual(
                output.value.get("transformation_plan").get("data").get("iteration"), 3
            )

    def test_update_iteration_without_transformation_plan(self):
        input_state = TestState(iteration=0, transformation_plan=None, codes=0)
        graph = build_graph(max_iteration=5, successfull_end=3)
        output = asyncio.run(graph.ainvoke(input_state, version="v2"))

        self.assertEqual(output.value.get("iteration"), 3)
        self.assertIsNone(output.value.get("transformation_plan"))


class TestIterationExceeded(TestCase):
    def test_cancel_iteration_exceeded(self):
        graph = build_graph(max_iteration=2, successfull_end=3)
        input_state = TestState(iteration=0, transformation_plan=None, codes=0)

        with self.assertRaises(RuntimeError):
            asyncio.run(graph.ainvoke(input_state, version="v2"))

    def test_cancel_iteration_does_not_exceed(self):
        graph = build_graph(max_iteration=5, successfull_end=3)
        input_state = TestState(iteration=0, transformation_plan=None, codes=0)

        output = asyncio.run(graph.ainvoke(input_state, version="v2"))
        self.assertEqual(output.value.get("iteration"), 3)
