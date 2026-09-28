from pathlib import Path
from typing import Literal

from mdeagent.implementation.types import TransformationClassGenerator


def create_transformation_class_generator(
    strategy: Literal["deep_agent", "hybrid_agent", "template_based", "pi"], workspace: Path, **kwargs
) -> TransformationClassGenerator:
    """
    Factory function to create a TransformationClassGenerator based on the configuration.
    """
    if strategy not in ["deep_agent", "hybrid_agent", "template_based", "pi"]:
        raise NotImplementedError(
            f"Unknown transformation implementation strategy: {strategy}"
        )

    if strategy == "template_based":
        raise NotImplementedError(
            "The 'template_based' strategy is not yet implemented. Please use 'deep_agent' or 'hybrid_agent'."
        )

    if strategy == "pi":
        from mdeagent.implementation.transformation.external.pi import (
            PITransformationClassGenerator,
        )

        return PITransformationClassGenerator(workspace)

    from mdeagent.implementation.transformation.react.wrapper import (
        TransformationClassAgentWrapper,
    )

    if strategy == "hybrid_agent":
        raise NotImplementedError(
            "The 'hybrid_agent' strategy is not yet implemented. Please use 'deep_agent' or 'template_based'."
        )
    else:
        from mdeagent.implementation.transformation.react.deep import (
            build_deep_agent,
        )

        graph = build_deep_agent(workspace, model=kwargs.get("model"))

    agent_wrapper = TransformationClassAgentWrapper(graph)
    return agent_wrapper
