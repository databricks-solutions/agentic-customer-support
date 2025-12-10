"""Core configuration schemas for the telco support agent."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Model for LLM configuration from YAML."""

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)


class UCConfig(BaseModel):
    """Unity Catalog configuration with separation of data reading vs agent artifacts."""

    # For reading data (customer info, billing, etc.) - always prod
    data_catalog: str = "telco_customer_support_prod"
    data_schema: str = "gold"

    # For agent artifacts (functions, models) - environment specific
    agent_catalog: str = "telco_customer_support_prod"  # Default to prod for testing
    agent_schema: str = "agent"
    model_name: str = "telco_customer_support_agent"

    def get_uc_function_name(self, function_name: str) -> str:
        """Returns full UC function name (uses agent catalog)."""
        return f"{self.agent_catalog}.{self.agent_schema}.{function_name}"

    def get_uc_table_name(self, table_name: str) -> str:
        """Returns full UC table name for data reading (uses data catalog)."""
        return f"{self.data_catalog}.{self.data_schema}.{table_name}"

    def get_uc_index_name(self, index_name: str) -> str:
        """Returns full UC vector search index name (uses data catalog)."""
        return f"{self.data_catalog}.{self.data_schema}.{index_name}"

    def get_uc_model_name(self) -> str:
        """Returns full UC model name (uses agent catalog)."""
        return f"{self.agent_catalog}.{self.agent_schema}.{self.model_name}"

    def get_uc_vs_endpoint_name(self) -> str:
        """Returns full UC vector search endpoint name (uses data catalog)."""
        return f"{self.data_catalog}.{self.data_schema}.{self.vs_endpoint_name}"

    @classmethod
    def load_from_file(cls) -> Optional["UCConfig"]:
        """Load UCConfig from yaml file."""
        import logging

        logger = logging.getLogger(__name__)
        # Try file-based paths first (model serving)
        search_paths = [
            # Model Serving: artifacts under /model/artifacts/
            Path("/model/artifacts") / "uc_config.yaml",
            # Development/notebook setting: configs in project structure
            Path(__file__).parent.parent.parent
            / "configs"
            / "agents"
            / "uc_config.yaml",
            # Current working directory (for local development)
            Path.cwd() / "configs" / "agents" / "uc_config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found UC config artifact at: {path}")
                with open(path) as f:
                    config_dict = yaml.safe_load(f)
                    return cls(**config_dict)
        return None


class MCPServer(BaseModel):
    """MCP Server configuration."""

    server_url: Optional[str] = None
    app_name: Optional[str] = None


class CacheConfig(BaseModel):
    """Configuration for agent caching."""

    enabled: bool = Field(default=False, description="Enable caching for this agent")
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for cache hits (0.0-1.0)",
    )


class AgentConfig(BaseModel):
    """Model for agent configuration from YAML."""

    name: str
    description: str
    llm: LLMConfig
    system_prompt: str
    uc_functions: list[str] = Field(default_factory=list)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    cache: Optional[CacheConfig] = Field(
        default=None, description="Optional cache configuration"
    )
    uc_config: UCConfig

    @classmethod
    def load_from_file(cls, agent_type: str, uc_config: UCConfig) -> "AgentConfig":
        """Load agent config from YAML file."""
        import logging

        logger = logging.getLogger(__name__)

        # path checking - different between model serving and development
        config_paths = [
            # Model Serving: artifacts under /model/artifacts/
            Path("/model/artifacts") / f"{agent_type}.yaml",
            # Development/notebook setting: configs in project structure
            Path(__file__).parent.parent.parent
            / "configs"
            / "agents"
            / f"{agent_type}.yaml",
            # Current working directory (for local development)
            Path.cwd() / "configs" / "agents" / f"{agent_type}.yaml",
        ]

        for path in config_paths:
            if path.exists():
                logger.info(f"Found {agent_type} config at: {path}")
                with open(path) as f:
                    config_dict = yaml.safe_load(f)
                    # Replace {env} placeholders with actual environment
                    config_dict = cls._interpolate_environment(config_dict)
                    return cls(**config_dict, uc_config=uc_config)

        logger.error(f"Config file not found for {agent_type}. Searched paths:")
        for path in config_paths:
            logger.error(f"  - {path} (exists: {path.exists()})")

        raise FileNotFoundError(f"Agent config file not found for {agent_type}")

    @classmethod
    def _interpolate_environment(cls, config_dict: dict) -> dict:
        """Replace {env} placeholders in config with actual environment value."""
        import json
        import os

        env = os.getenv("ENV", "dev")
        # Convert to JSON string and back to handle nested dictionaries
        config_str = json.dumps(config_dict)
        interpolated_str = config_str.replace("{env}", env)
        return json.loads(interpolated_str)
