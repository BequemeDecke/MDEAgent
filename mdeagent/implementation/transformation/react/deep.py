import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import BaseChatModel

from mdeagent.comprehension.tools import read_transformation_plan
from mdeagent.implementation.transformation.react.middleware import (
    TrackWrittenFilesMiddleware,
)
from mdeagent.implementation.transformation.react.wrapper import (
    TransformationClassAgentState,
)
from mdeagent.models import build_coding_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a coding agent in a model driven development environment that helps writing java code for model to model transformations.
You are given the interfaces of the source and target models, and the transformation rules, and you are expected to write the code for the transformation between them.

A transformation plan is also provided, which describes the steps to be taken in order to perform the transformation. You should follow the plan and write the code accordingly.
"""


def build_deep_agent(workspace: Path, model: BaseChatModel | None = None):
    """Creates a coding agent that can write code for model to model transformations based on the provided transformation plan.
    The agent is created with a custom set of tools that allow it to read and write the transformation class files, as well as to read the transformation plan.

    Args:
        workspace (Path): The root path of the workspace where the agent will operate.
        model (BaseChatModel | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if model is None:
        model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        state_schema=TransformationClassAgentState,
        middleware=[
            TrackWrittenFilesMiddleware(
                workspace_path=workspace, file_extension_filter=".java"
            )
        ],
        tools=[read_transformation_plan],
    )
