from typing import Any

from mdeagent.state import MDEAgentState


def mde_to_transformation_path(state: MDEAgentState) -> dict[str, Any]:
    """Maps the transformation class path from MDEAgentState to evaluation parameters."""
    path = state.get("transformation_class_path")
    if path is None:
        return {"transformation_class_path": ""}

    return {"transformation_class_path": str(path)}
