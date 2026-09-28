import logging
import subprocess
from pathlib import Path

from mdeagent.preparation.pom import Module, Parent, Pom

logger = logging.getLogger(__name__)


class MavenProject:
    pom: Pom
    workspace: Path

    def __init__(self, pom: Pom, workspace: Path):
        self.pom = pom
        self.workspace = workspace

    def validate(self) -> bool:
        """
        Execute `mvn validate` to check if the Maven project is valid.

        :return: True if the project is valid, False otherwise.
        """
        validate_process = subprocess.run(
            ["mvn", "validate"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate_process.returncode != 0:
            logger.error(
                f"Maven validation failed for project at {self.workspace}:\n{validate_process.stdout}\n{validate_process.stderr}"
            )
        return validate_process.returncode == 0

    def format(self) -> bool:
        """
        Execute `mvn spotless:apply` to format the code in the Maven project.

        :return: True if the formatting was successful, False otherwise.
        """
        format_process = subprocess.run(
            ["mvn", "spotless:apply"],
            capture_output=True,
            cwd=self.workspace,
            check=False,
            text=True,
        )
        return format_process.returncode == 0

    def compile(self) -> tuple[bool, str]:
        """
        Execute `mvn compile` to compile the Maven project.

        :return: A tuple containing a boolean indicating success and the output of the compilation process.
        """
        compile_process = subprocess.run(
            ["mvn", "compile"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        return (
            compile_process.returncode == 0,
            compile_process.stderr
            if compile_process.returncode != 0
            else compile_process.stdout,
        )

    def build(self) -> bool:
        """
        Execute `mvn package` to build the Maven project.

        :return: True if the build was successful, False otherwise.
        """
        build_process = subprocess.run(
            ["mvn", "package"], 
            cwd=self.workspace, 
            capture_output=True,
            text=True,
            check=False, 
        )
        return build_process.returncode == 0

    def add_file(self, relative_path: Path, content: str) -> Path:
        """
        Add a new file to the Maven project.

        :param relative_path: The relative path of the file to be added.
        :param content: The content to write into the file.
        """
        full_path = self.workspace / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    def get_package_path(self, package: str) -> Path:
        """
        Get the path to the specified package in the Maven project.

        :param package: The package name (e.g., "com.example").
        :return: The Path object representing the package directory.
        """
        package_path = Path(*package.split("."))
        return self.workspace / "src" / "main" / "java" / package_path

    def add_java_class(self, package: str, class_name: str, content: str) -> Path:
        """
        Add a new Java class to the Maven project.

        :param package: The package name for the Java class.
        :param class_name: The name of the Java class.
        :param content: The content of the Java class.
        """
        # Get absolute package path, then convert to relative path for add_file()
        package_path = self.get_package_path(package)
        relative_package_path = package_path.relative_to(self.workspace)
        java_file_path = relative_package_path / f"{class_name}.java"
        return self.add_file(java_file_path, content)

    @classmethod
    def load(cls, workspace: Path) -> "MavenProject":
        """
        Load an existing Maven project from the specified workspace.

        :param workspace: The path to the workspace where the Maven project is located.
        :return: An instance of MavenProject representing the loaded project.
        :raises FileNotFoundError: If pom.xml is not found in the workspace.
        :raises ValueError: If the pom.xml is invalid or missing required elements.
        """
        pom_path = workspace / "pom.xml"
        if not pom_path.exists():
            raise FileNotFoundError(f"pom.xml not found in {workspace}")

        # Validate that the POM file is not empty before parsing
        pom_content = pom_path.read_text()
        if not pom_content.strip():
            raise ValueError(f"pom.xml at {pom_path} is empty")

        try:
            pom = Pom(pom_path)
        except Exception as e:
            raise ValueError(f"Failed to parse pom.xml: {e}") from e

        return cls(pom, workspace)

    @classmethod
    def create(
        cls,
        workspace: Path,
        group_id: str,
        artifact_id: str,
        parent: "MavenProject | None" = None,
    ) -> "MavenProject":
        """
        Create a new Maven project in the specified workspace.

        :param workspace: The path to the workspace where the project will be created.
        :param group_id: The group ID for the Maven project.
        :param artifact_id: The artifact ID for the Maven project.
        :param parent: The parent Maven project.
        :return: An instance of MavenProject representing the created project.
        """
        if parent is None:
            pom = Pom.new(workspace, group_id=group_id, artifact_id=artifact_id)
            return cls(pom, workspace)

        cp_process = subprocess.run(
            [
                "mvn",
                "archetype:generate",
                "-DgroupId=" + group_id,
                "-DartifactId=" + artifact_id,
                "-DarchetypeArtifactId=maven-archetype-simple",
                "-DarchetypeVersion=1.5",
                "-DinteractiveMode=false",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if cp_process.returncode != 0:
            raise RuntimeError(
                f"Failed to create Maven project. Return code: {cp_process.returncode}"
            )
        parent.pom.add_module(Module(artifact_id))
        pom_path = workspace / artifact_id / "pom.xml"
        pom = Pom(pom_path)
        return cls(pom, workspace / artifact_id)

    @classmethod
    def register_submodule(
        cls, parent: "MavenProject", submodule: "MavenProject"
    ) -> None:
        """
        Register a submodule in the parent Maven project.

        :param parent: The parent Maven project.
        :param submodule: The submodule Maven project to be registered.
        """
        parent_pom = parent.pom
        submodule_pom = submodule.pom

        parent_pom.add_module(Module(submodule_pom.artifact_id))
        parent_pom.save()
        submodule_pom.set_parent(
            Parent(
                group_id=parent_pom.group_id,
                artifact_id=parent_pom.artifact_id,
                version=parent_pom.version,
            )
        )
        submodule_pom.save()
