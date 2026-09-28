import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, TypedDict

from jinja2 import Environment, FileSystemLoader, Template

from mdeagent.util import limit_text

logger = logging.getLogger(__name__)


class TransformationPlanData(TypedDict):
    source_model_package: str
    target_model_package: str
    iteration: int
    source_model_implementation: str
    target_model_implementation: str
    transformation_direction: str
    difficulties: str
    implementation_steps: str

    def __repr__(self) -> str:
        return (
            f"TransformationPlanData(source_model_package={limit_text(self['source_model_package'], 10)}, "
            f"\ttarget_model_package={limit_text(self['target_model_package'], 10) if len(self['target_model_package']) > 0 else 'N/A'}, "
            f"\titeration={self['iteration']}, "
            f"\tsource_model_implementation={limit_text(self['source_model_implementation'], 10)}, "
            f"\ttarget_model_implementation={limit_text(self['target_model_implementation'], 10)}, "
            f"\ttransformation_direction={limit_text(self['transformation_direction'], 10)}, "
            f"\tdifficulties={limit_text(self['difficulties'], 10)}, "
            f"\timplementation_steps={limit_text(self['implementation_steps'], 10)})"
        )


class SerializedTransformationPlanParser(TypedDict):
    type: str
    args: dict[str, Any]

    def __repr__(self) -> str:
        return f"SerializedTransformationPlanParser(type={self['type']}, args={self['args']})"


class SerializedTransformationPlan(TypedDict):
    data: TransformationPlanData
    parser: SerializedTransformationPlanParser
    template: Path

    def __repr__(self) -> str:
        return f"SerializedTransformationPlan(data={self['data']}, parser={self['parser']}, template={self['template']})"


class TransformationPlanParser(ABC):
    @abstractmethod
    def parse(self) -> TransformationPlanData:
        """
        Parsed the transformation plan.

        Returns:
            TransformationPlanData: The parsed transformation plan data.
        """
        pass

    @abstractmethod
    def save(self, data: str) -> None:
        pass

    @abstractmethod
    def to_dict(self) -> SerializedTransformationPlanParser:
        pass

    @classmethod
    @abstractmethod
    def from_dict(
        cls, data: SerializedTransformationPlanParser
    ) -> "TransformationPlanParser":
        pass


class FileTransformationPlanParser(TransformationPlanParser):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def _parse_header(self, content: str) -> dict:
        """
        Parses the header of the transformation plan file to extract source_model_package, target_model_package, and iteration.

        ---
        source_model_package: {{source_model_package}}
        target_model_package: {{target_model_package}}
        iteration: {{iteration}}
        ---

        Args:
            content (str): The content of the transformation plan file.
        Returns:
            dict: A dictionary containing source_model_package, target_model_package, and iteration.
        """
        match = re.search(
            r"---\s*source_model_package:\s*(?P<source_model_package>.*?)\s*target_model_package:\s*(?P<target_model_package>.*?)\s*iteration:\s*(?P<iteration>\d+)\s*---",
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse transformation plan header.")

        return {
            "source_model_package": match.group("source_model_package"),
            "target_model_package": match.group("target_model_package"),
            "iteration": int(match.group("iteration")),
        }

    def _parse_model_implementation(
        self, content: str, model_type: Literal["source", "target"]
    ) -> str:
        """
        Parses the source or target model implementation from the transformation plan file.
        With following regex:
        ```md
        --- BEGIN SOURCE MODEL ---
        {{source_model_implementation}}
        --- END SOURCE MODEL ---
        ```
        """
        if model_type == "source":
            pattern = r"---\s*BEGIN SOURCE MODEL\s*---\s*(?P<source_model_implementation>.*?)\s*---\s*END SOURCE MODEL\s*---"
        else:
            pattern = r"---\s*BEGIN TARGET MODEL\s*---\s*(?P<target_model_implementation>.*?)\s*---\s*END TARGET MODEL\s*---"

        match = re.search(
            pattern,
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse model implementation.")

        return match.group(f"{model_type}_model_implementation")

    def _parse_transformation_direction(self, content: str) -> str:
        """
        Parses the transformation direction from the transformation plan file.
        With following regex:
        ```md
        --- BEGIN TRANSFORMATION DIRECTION ---
        {{transformation_direction}}
        --- END TRANSFORMATION DIRECTION ---
        ```
        """
        match = re.search(
            r"---\s*BEGIN TRANSFORMATION DIRECTION\s*---\s*(?P<transformation_direction>.*?)\s*---\s*END TRANSFORMATION DIRECTION\s*---",
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse transformation direction.")

        return match.group("transformation_direction")

    def _parse_difficulties(self, content: str) -> str:
        """
        Parses the transformation difficulties from the transformation plan file.
        With following regex:
        ```md
        --- BEGIN DIFFICULTIES ---
        {{difficulties}}
        --- END DIFFICULTIES ---
        ```
        """
        match = re.search(
            r"---\s*BEGIN DIFFICULTIES\s*---\s*(?P<difficulties>.*?)\s*---\s*END DIFFICULTIES\s*---",
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse transformation difficulties.")

        return match.group("difficulties")

    def _parse_implementation_steps(self, content: str) -> str:
        """
        Parses the implementation steps from the transformation plan file.
        With following regex:
        ```md
        --- BEGIN IMPLEMENTATION STEPS ---
        {{implementation_steps}}
        --- END IMPLEMENTATION STEPS ---
        ```
        """
        match = re.search(
            r"---\s*BEGIN IMPLEMENTATION STEPS\s*---\s*(?P<implementation_steps>.*?)\s*---\s*END IMPLEMENTATION STEPS\s*---",
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse implementation steps.")

        return match.group("implementation_steps")

    def parse(self) -> TransformationPlanData:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Transformation plan file not found at {self.file_path}"
            )
        content = self.file_path.read_text()
        if content.strip() == "":
            raise ValueError("The transformation plan file is empty.")

        header_data = self._parse_header(content)
        source_model_implementation = self._parse_model_implementation(
            content, "source"
        )
        target_model_implementation = self._parse_model_implementation(
            content, "target"
        )
        transformation_direction = self._parse_transformation_direction(content)
        difficulties = self._parse_difficulties(content)
        implementation_steps = self._parse_implementation_steps(content)

        return {
            "source_model_package": header_data["source_model_package"],
            "target_model_package": header_data["target_model_package"],
            "iteration": header_data["iteration"],
            "source_model_implementation": source_model_implementation,
            "target_model_implementation": target_model_implementation,
            "transformation_direction": transformation_direction,
            "difficulties": difficulties,
            "implementation_steps": implementation_steps,
        }

    def save(self, data: str) -> None:
        self.file_path.write_text(data)

    def to_dict(self) -> SerializedTransformationPlanParser:
        return {
            "type": self.__class__.__name__,
            "args": {
                "file_path": str(self.file_path),
            },
        }

    @classmethod
    def from_dict(
        cls, data: SerializedTransformationPlanParser
    ) -> "FileTransformationPlanParser":
        if data["type"] != cls.__name__:
            raise ValueError(f"Invalid parser type: {data['type']}")
        file_path = Path(data["args"]["file_path"])
        return cls(file_path)


class TransformationPlan:
    data: TransformationPlanData
    parser: TransformationPlanParser
    template: Template
    _template_path: Path  # only for serialization

    def __init__(
        self,
        parser: TransformationPlanParser,
        template_path: Path,
    ):
        self.parser = parser
        self.template = Environment(
            loader=FileSystemLoader(template_path)
        ).get_template("transformation_plan.jinja")
        self._template_path = template_path

    @classmethod
    def from_dict(cls, plan_dict: SerializedTransformationPlan) -> "TransformationPlan":
        """
        Creates a TransformationPlan instance from a dictionary representation without parsing the transformation plan again.
        """
        template_path = plan_dict["template"]
        data = plan_dict["data"]
        type_of_parser = plan_dict["parser"]["type"]
        if type_of_parser == "FileTransformationPlanParser":
            parser = FileTransformationPlanParser.from_dict(plan_dict["parser"])
        else:
            raise ValueError(f"Unknown parser type: {type_of_parser}")

        tp = cls(parser=parser, template_path=template_path)
        tp.data = data  # Instead of parsing use the deserialized data directly
        return tp

    @classmethod
    def parse(
        cls,
        parser: TransformationPlanParser,
        template_path: Path = Path.cwd() / "templates",
    ) -> "TransformationPlan":
        """
        Creates a TransformationPlan instance by parsing the transformation plan using the provided parser.

        This method is used to create an initial transformation plan.
        """
        tp = cls(parser=parser, template_path=template_path)
        try:
            tp.data = parser.parse()
        except Exception as e:
            logger.warning(f"Error occurred while parsing transformation plan: {e}")
            tp.data = {
                "source_model_package": "",
                "target_model_package": "",
                "iteration": 0,
                "source_model_implementation": "",
                "target_model_implementation": "",
                "transformation_direction": "",
                "difficulties": "",
                "implementation_steps": "",
            }
            logger.info("Creating initial transformation plan with default values.")
            parser.save(str(tp))
        return tp

    def __str__(self) -> str:
        rendered_content = self.template.render(
            source_model_package=self.data["source_model_package"],
            target_model_package=self.data["target_model_package"],
            iteration=self.data["iteration"],
            source_model_implementation=self.data["source_model_implementation"],
            target_model_implementation=self.data["target_model_implementation"],
            transformation_direction=self.data["transformation_direction"],
            difficulties=self.data["difficulties"],
            implementation_steps=self.data["implementation_steps"],
        )
        return rendered_content

    def to_dict(self) -> SerializedTransformationPlan:
        return {
            "data": self.data.copy(),
            "parser": self.parser.to_dict(),
            "template": self._template_path,
        }

    def update_package_information(
        self, source_model_package: str, target_model_package: str
    ):
        self.data["source_model_package"] = source_model_package
        self.data["target_model_package"] = target_model_package
        self.parser.save(str(self))

    def update_iteration(self, iteration: int):
        self.data["iteration"] = iteration
        self.parser.save(str(self))

    def update_model_implementation(
        self, source_model_implementation: str, target_model_implementation: str
    ):
        self.data["source_model_implementation"] = source_model_implementation
        self.data["target_model_implementation"] = target_model_implementation
        self.parser.save(str(self))

    def update_transformation_direction(self, transformation_direction: str):
        self.data["transformation_direction"] = transformation_direction
        self.parser.save(str(self))

    def update_transformation_difficulties(self, difficulties: str):
        self.data["difficulties"] = difficulties
        self.parser.save(str(self))

    def update_implementation_steps(self, implementation_steps: str):
        self.data["implementation_steps"] = implementation_steps
        self.parser.save(str(self))
