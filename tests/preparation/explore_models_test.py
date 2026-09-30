import asyncio
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase

from mdeagent.preparation.explore_models import (
    create_explore_models_node,
)
from mdeagent.preparation.pom import Pom
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.state import ModelImplementation, PreparationState

METAMODEL_PATH = Path.cwd() / ".mdeagent-tests" / "setup" / "metamodels"


# --- HELPER FUNCTIONS FOR TESTS ---
def check_metamodel_submodule_installed():
    """
    Checks whether the git submodule for the metamodel is initialized without changing
    its checked-out revision.
    """
    git_check = subprocess.run(
        ["git", "submodule", "status", "--", str(METAMODEL_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        git_check.returncode == 0
        and bool(git_check.stdout.strip())
        and not git_check.stdout.startswith("-")
    )


def check_metamodel_submodule_exists():
    """
    Checks if the git submodule for the metamodel exists for Families and Persons. If not, it skips the test and raises an error.
    """
    return (METAMODEL_PATH / "Families").exists() and (
        METAMODEL_PATH / "Persons"
    ).exists()


def get_model_package_files(model_path: Path, artifact_id: str) -> list[Path]:
    """
    Returns the paths of the Java files in the model package, including the gen subfolder if it exists.
    """
    gen_path = model_path / "gen" / artifact_id
    search_path = gen_path if gen_path.exists() and gen_path.is_dir() else model_path
    return list(search_path.glob("*.java"))


class TestExploreModels(TestCase):
    def setUp(self):
        if (
            not check_metamodel_submodule_installed()
            or not check_metamodel_submodule_exists()
        ):
            self.skipTest(
                "Git submodule for metamodel is not installed. Skipping test."
            )

        self.families_model_path = METAMODEL_PATH / "Families"
        self.persons_model_path = METAMODEL_PATH / "Persons"
        self.explore_models = create_explore_models_node()

    def test_explore_models__models_exist(self):
        """
        Test, which checks if the explore_models function correctly follows the path and reads the content of the implementation of the model.
        Module names are derived from the path stem (e.g., /path/Families → Families).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workspace = temp_path / "workspace"
            workspace.mkdir()

            artifact_id = "Transformation"
            group_id = "com.example"
            parent_project = MavenProject.create(workspace, group_id, workspace.stem)
            transformation_project = MavenProject.create(workspace, group_id, artifact_id, parent_project)

            result = asyncio.run(self.explore_models(
                PreparationState(
                    workspace_path=workspace,
                    artifact_id=artifact_id,
                    maven_project_path=transformation_project.workspace,
                    group_id=group_id,
                    source_model=ModelImplementation(
                        name="Families",
                        path=self.families_model_path,
                    ),
                    target_model=ModelImplementation(
                        name="Persons",
                        path=self.persons_model_path,
                    ),
                )
            ))

            # Check if the parent/pom.xml has been updated with the new modules
            parent_pom = Pom(workspace / "pom.xml")
            module_ids = [m.artifact_id for m in parent_pom.modules]
            self.assertIn("Families", module_ids)
            self.assertIn("Persons", module_ids)

            # Check if the source and target models have been copied to the workspace with correct names
            self.assertTrue((workspace / "Families").exists())
            self.assertTrue((workspace / "Persons").exists())

            # Check if the Maven projects for source and target models have been created and contain the expected Java files
            source_java_files = get_model_package_files(
                workspace / "Families", "Families"
            )
            target_java_files = get_model_package_files(
                workspace / "Persons", "Persons"
            )
            self.assertTrue(any(f.name == "Family.java" for f in source_java_files))
            self.assertTrue(any(f.name == "Person.java" for f in target_java_files))

            # Check if the result contains the implementation content for both models
            self.assertIn("implementation", result["source_model"])
            self.assertIn("implementation", result["target_model"])

            # Check if copied metamodel packages have parent set in pom.xml
            source_pom_content = (workspace / "Families" / "pom.xml").read_text()
            target_pom_content = (workspace / "Persons" / "pom.xml").read_text()
            self.assertIn("<parent>", source_pom_content)
            self.assertIn("<parent>", target_pom_content)
