import re

from pydantic import BaseModel, field_validator

from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult


class TransformationCodeSchema(BaseModel):
    transformation_class_path: str

    @field_validator("transformation_class_path")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("transformation_class_path must not be empty")
        return v


class TransformationCodeEvaluation(Evaluation):
    """Evaluates whether the transformation class file at a given path contains
    actual Java code (is not empty / only whitespace / only comments)."""

    # Minimal set of Java keywords that signal "there is real code here".
    _JAVA_KEYWORDS = re.compile(
        r"\b(class|interface|enum|public|private|protected|void|int|String|return|if|for|while|new|extends|implements)\b",
        re.IGNORECASE,
    )

    # Pattern that matches a Java class declaration with at least one method.
    _CLASS_DECLARATION = re.compile(
        r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+\w+\s*\{.*\}",
        re.DOTALL,
    )

    async def setup(self) -> None:
        pass

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        path_str: str = kwargs.get("transformation_class_path", "")

        if not path_str:
            return (
                [
                    EvaluationResult(
                        content="No transformation class path provided.",
                        metadata={"success": False, "include_in_report": True},
                    )
                ],
                [],
            )

        from pathlib import Path

        file_path = Path(path_str)

        # 1. File must exist.
        if not file_path.exists():
            return (
                [
                    EvaluationResult(
                        content=f"Transformation class file does not exist: {file_path}",
                        metadata={"success": False, "include_in_report": True},
                    )
                ],
                [],
            )

        if not file_path.is_file():
            return (
                [
                    EvaluationResult(
                        content=f"Transformation class path is not a file: {file_path}",
                        metadata={"success": False, "include_in_report": True},
                    )
                ],
                [],
            )

        # 2. Read file content.
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return (
                [],
                [
                    EvaluationError(
                        message=f"Failed to read file '{file_path}': {e}",
                        type=type(e).__name__,
                        details={"file": str(file_path)},
                    )
                ],
            )

        results: list[EvaluationResult] = []

        # 2a. Non-empty / non-whitespace check.
        if not content.strip():
            results.append(
                EvaluationResult(
                    content=f"Transformation class file is empty: {file_path}",
                    metadata={"success": False, "include_in_report": True},
                )
            )
            return results, []

        # 2b. Contains at least some Java-like code (keyword or class declaration).
        if self._JAVA_KEYWORDS.search(content) or self._CLASS_DECLARATION.search(
            content
        ):
            results.append(
                EvaluationResult(
                    content=f"Transformation class file contains code: {file_path}",
                    metadata={"success": True, "include_in_report": False},
                )
            )
        else:
            results.append(
                EvaluationResult(
                    content=f"Transformation class file does not contain code (only comments / non-code): {file_path}",
                    metadata={"success": False, "include_in_report": True},
                )
            )

        return results, []
