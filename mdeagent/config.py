import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration class for used models in the agent."""

    API_KEY: SecretStr = Field()
    BASE_URL: str = Field()
    BASE_MODEL: str = Field()
    CODING_MODEL: str = Field()
    REQUEST_TIMEOUT: int = Field(default=60)
    MAX_RETRIES: int = Field(default=0)


class LangFuseConfig(BaseModel):
    SECRET_KEY: SecretStr = Field()
    PUBLIC_KEY: SecretStr = Field()
    BASE_URL: str = Field()


class VariablesConfig(BaseModel):
    UPDATED_FILE_INDEX: int = Field(
        default=13,
        description="The index at which the file path starts in the tool message content for write_file tool messages.",
    )
    TRANSFORMATION_CLASS_NAME: str = Field(
        default="MDEAgentTransformation",
        description="The name of the transformation class that will be generated and used in the transformation process.",
    )


class AgentControlConfig(BaseModel):
    """Configuration class for the agent control."""

    WORKFLOW_MAX_ITERATIONS: int = Field(
        default=5,
        description="Maximum number of iterations for the workflow transformation process.",
    )
    SUBGRAPH_MAX_ITERATIONS: int = Field(
        default=3,
        description="Maximum number of iterations for a subgraph process.",
    )
    TRANSFORMATION_IMPLEMENTATION_STRATEGY: Literal[
        "deep_agent", "hybrid_agent", "template_based", "pi"
    ] = Field(
        default="deep_agent",
        description="The strategy to be used for the transformation implementation. Options are 'deep_agent', 'hybrid_agent', 'template_based' or 'pi'.",
    )


class Config(BaseModel):
    """Main configuration class that holds all configurations for the application."""

    MODEL: ModelConfig
    LANGFUSE: LangFuseConfig
    VARIABLES: VariablesConfig
    AGENT_CONTROL: AgentControlConfig

    @classmethod
    def get_instance(cls, env_path: Path = Path.cwd() / ".env") -> "Config":
        """Singleton pattern to get a single instance of the configuration."""
        if not hasattr(cls, "_instance"):
            cls._instance = load_config(env_path=env_path)
        return cls._instance


def load_config(env_path: Path) -> BaseModel:
    # Load environment variables from the .env file
    has_env_loaded = load_dotenv(dotenv_path=env_path)
    assert has_env_loaded, f"Failed to load environment variables from {env_path}"

    # Save the loaded environment variables to a config class for easy access
    agent_config = ModelConfig(
        API_KEY=os.getenv("API_KEY"),
        BASE_URL=os.getenv("BASE_URL"),
        BASE_MODEL=os.getenv("BASE_MODEL"),
        CODING_MODEL=os.getenv("CODING_MODEL"),
        REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT", "60")),
        MAX_RETRIES=int(os.getenv("MAX_RETRIES", "0")),
    )

    langfuse_config = LangFuseConfig(
        SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY"),
        PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY"),
        BASE_URL=os.getenv("LANGFUSE_BASE_URL"),
    )

    variables_config = VariablesConfig(
        UPDATED_FILE_INDEX=int(os.getenv("UPDATED_FILE_INDEX", "13")),
        TRANSFORMATION_CLASS_NAME=os.getenv(
            "TRANSFORMATION_CLASS_NAME", "MDEAgentTransformation"
        ),
    )

    workflow_approach_config = AgentControlConfig(
        WORKFLOW_MAX_ITERATIONS=int(os.getenv("WORKFLOW_MAX_ITERATIONS", "5")),
        TRANSFORMATION_IMPLEMENTATION_STRATEGY=os.getenv(
            "TRANSFORMATION_IMPLEMENTATION_STRATEGY", "deep_agent"
        ),
        SUBGRAPH_MAX_ITERATIONS=int(os.getenv("SUBGRAPH_MAX_ITERATIONS", "3")),
    )

    # Log the loaded configurations
    logger.debug("--- Loaded Configurations ---")
    logger.debug(f"Loaded ModelConfig: {agent_config}")
    logger.debug(f"Loaded LangFuseConfig: {langfuse_config}")
    logger.debug(f"Loaded VariablesConfig: {variables_config}")
    logger.debug(f"Loaded AgentControlConfig: {workflow_approach_config}")
    logger.debug("-----------------------------")

    return Config(
        MODEL=agent_config,
        LANGFUSE=langfuse_config,
        VARIABLES=variables_config,
        AGENT_CONTROL=workflow_approach_config,
    )
