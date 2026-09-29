from typing import Any

from mdeagent.implementation.state import ImplementationState


def implementation_to_transformation_path(
    state: ImplementationState,
) -> dict[str, Any]:
    """Maps the transformation class path from ImplementationState to evaluation parameters."""
    path = state.get("transformation_class")
    if path is None:
        return {"transformation_class_path": ""}

    # TransformationClass is a TypedDict with a 'path' field of type Path
    return {"transformation_class_path": str(path["path"])}
