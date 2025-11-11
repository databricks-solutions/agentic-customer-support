"""Functions for deploying Telco Support Agent models."""

import time
from typing import Any, Optional, Union

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointStateConfigUpdate,
    EndpointStateReady,
)
from mlflow.entities.model_registry import ModelVersion

from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentDeploymentError(Exception):
    """Raised when agent deployment fails."""

    pass


def deploy_agent(
    uc_model_name: str,
    model_version: Union[str, int, ModelVersion],
    deployment_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    scale_to_zero_enabled: bool = False,
    environment_vars: Optional[dict[str, str]] = None,
    workload_size: str = "Small",
    wait_for_ready: bool = True,
    permissions: Optional[dict] = None,
    instructions: Optional[str] = None,
    budget_policy_id: Optional[str] = None,
) -> Any:
    """Deploy a registered agent model to a Model Serving endpoint.

    Args:
        uc_model_name: Fully qualified name in Unity Catalog (catalog.schema.model)
        model_version: Version of the model to deploy
        deployment_name: Optional name for the deployment
        tags: Optional tags to attach to the deployment
        scale_to_zero_enabled: Whether to enable scale-to-zero for idle endpoints
        environment_vars: Optional environment variables for the deployment
        workload_size: Size of the workload ("Small", "Medium", "Large")
        wait_for_ready: Whether to wait for deployment to be ready
        permissions: Optional permissions configuration dict with 'users' and 'permission_level'
        instructions: Optional review instructions for the Review App
        budget_policy_id: Optional budget policy ID

    Returns:
        Deployment object with information about the deployed agent

    Raises:
        AgentDeploymentError: If deployment fails
    """
    # get version number if ModelVersion object
    if isinstance(model_version, ModelVersion):
        model_version = model_version.version

    logger.info(f"Deploying agent model: {uc_model_name} version {model_version}")
    logger.info(f"Workload size: {workload_size}")
    logger.info(f"Scale to zero: {scale_to_zero_enabled}")

    # set default tags if none provided
    if tags is None:
        tags = {"source": "telco_support_agent"}

    # validate workload size
    valid_sizes = ["Small", "Medium", "Large"]
    if workload_size not in valid_sizes:
        raise ValueError(
            f"Invalid workload size: {workload_size}. Must be one of: {valid_sizes}"
        )

    try:
        # deploy agent
        logger.info("Starting deployment...")

        # Check if endpoint exists to determine if we should pass tags
        w = WorkspaceClient()
        try:
            w.serving_endpoints.get(deployment_name)
            logger.info(f"Endpoint {deployment_name} exists, updating...")
            # Don't pass tags on update to avoid duplicate key error
            deployment = agents.deploy(
                model_name=uc_model_name,
                model_version=model_version,
                endpoint_name=deployment_name,
                scale_to_zero=scale_to_zero_enabled,
                environment_vars=environment_vars,
                workload_size=workload_size,
                budget_policy_id=budget_policy_id,
            )
        except Exception:
            # Endpoint doesn't exist, create new with tags
            logger.info(f"Creating new endpoint {deployment_name}...")
            deployment = agents.deploy(
                model_name=uc_model_name,
                model_version=model_version,
                endpoint_name=deployment_name,
                tags=tags,
                scale_to_zero=scale_to_zero_enabled,
                environment_vars=environment_vars,
                workload_size=workload_size,
                budget_policy_id=budget_policy_id,
            )

        logger.info(
            f"Agent deployment started. Endpoint name: {deployment.endpoint_name}"
        )

        # set review instructions if provided
        if instructions:
            logger.info("Setting review instructions...")
            agents.set_review_instructions(uc_model_name, instructions)

        # wait for endpoint to be ready if requested
        if wait_for_ready:
            logger.info("Waiting for endpoint to be ready...")
            _wait_for_endpoint_ready(deployment.endpoint_name)
            logger.info("Endpoint is ready")

        # set permissions if provided
        if permissions:
            logger.info("Setting agent permissions...")
            _set_permissions(uc_model_name, permissions)
            logger.info("Permissions set successfully")

        logger.info(f"Successfully deployed agent at: {deployment.endpoint_name}")
        logger.info(f"Query endpoint: {deployment.query_endpoint}")

        return deployment

    except Exception as e:
        error_msg = f"Failed to deploy agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise AgentDeploymentError(error_msg) from e


def cleanup_old_deployments(
    model_name: str,
    current_version: str,
    endpoint_name: str,
    keep_previous_count: int = 1,
    max_deletion_attempts: int = 3,
    wait_between_attempts: int = 30,
    wait_after_deletion: int = 10,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Clean up older deployed agent versions.

    Retrieves all deployed versions for a given model and deletes older
    versions while keeping the current version and a configurable number of previous versions.

    Args:
        model_name: The full UC model name
        current_version: The current version number (string) that should be kept
        endpoint_name: The name of the endpoint
        keep_previous_count: Number of previous versions to keep (default: 1)
        max_deletion_attempts: Maximum number of attempts to delete a version (default: 3)
        wait_between_attempts: Seconds to wait between deletion attempts (default: 30)
        wait_after_deletion: Seconds to wait after successful deletion (default: 10)
        raise_on_error: Whether to raise exceptions on cleanup errors (default: False)

    Returns:
        Dictionary containing:
            - versions_deleted: List of successfully deleted versions
            - versions_failed: List of versions that couldn't be deleted
            - versions_kept: List of versions that were kept

    Raises:
        AgentDeploymentError: If raise_on_error is True and cleanup fails
    """
    result = {
        "versions_deleted": [],
        "versions_failed": [],
        "versions_kept": [int(current_version)],
    }

    try:
        # get all deployed versions for this model
        all_deployments = agents.get_deployments(model_name=model_name)

        if not all_deployments:
            logger.info("No existing deployments found to clean up")
            return result

        endpoint_deployments = [
            dep for dep in all_deployments if dep.endpoint_name == endpoint_name
        ]

        logger.info(
            f"Found {len(endpoint_deployments)} existing deployments for {model_name} in endpoint {endpoint_name}"
        )

        current_version_int = int(current_version)
        deployed_versions = [int(dep.model_version) for dep in endpoint_deployments]
        deployed_versions.sort(reverse=True)

        previous_versions = [v for v in deployed_versions if v < current_version_int]
        versions_to_keep = [current_version_int]
        if previous_versions and keep_previous_count > 0:
            versions_to_keep.extend(previous_versions[:keep_previous_count])
        versions_to_delete = [v for v in deployed_versions if v not in versions_to_keep]

        result["versions_kept"] = versions_to_keep

        if not versions_to_delete:
            logger.info("No older versions to delete")
            return result

        logger.info(f"Current version: {current_version_int}")
        logger.info(f"Versions to keep: {versions_to_keep}")
        logger.info(f"Versions to delete: {versions_to_delete}")

        w = WorkspaceClient()

        # can only delete one at a time, need to wait for endpoint to update
        for version in versions_to_delete:
            attempt = 0
            deleted = False

            while attempt < max_deletion_attempts and not deleted:
                attempt += 1
                try:
                    # check if endpoint is ready for updates
                    endpoint_status = w.serving_endpoints.get(endpoint_name)
                    if (
                        endpoint_status.state.config_update
                        == EndpointStateConfigUpdate.IN_PROGRESS
                    ):
                        logger.info(
                            f"Endpoint update in progress, waiting before deleting version {version} "
                            f"(attempt {attempt}/{max_deletion_attempts})"
                        )
                        time.sleep(wait_between_attempts)
                        continue

                    # delete the deployment
                    agents.delete_deployment(
                        model_name=model_name, model_version=version
                    )
                    logger.info(
                        f"Successfully deleted deployment for version {version}"
                    )
                    deleted = True
                    result["versions_deleted"].append(version)

                    # wait for endpoint update to complete
                    logger.info(
                        f"Waiting for endpoint update to complete after deleting version {version}"
                    )
                    wait_start = time.time()
                    while time.time() - wait_start < wait_after_deletion:
                        endpoint_status = w.serving_endpoints.get(endpoint_name)
                        if (
                            endpoint_status.state.config_update
                            != EndpointStateConfigUpdate.IN_PROGRESS
                        ):
                            logger.info(
                                f"Endpoint update completed for version {version}"
                            )
                            break
                        time.sleep(10)

                except Exception as e:
                    error_str = str(e)
                    logger.warning(
                        f"Failed to delete deployment for version {version} "
                        f"(attempt {attempt}/{max_deletion_attempts}): {error_str}"
                    )

                    # if deployment doesn't exist, no need to retry
                    if "does not exist" in error_str.lower():
                        logger.info(
                            f"Deployment for version {version} does not exist, skipping further attempts"
                        )
                        break

                    if attempt == max_deletion_attempts:
                        result["versions_failed"].append(version)
                    else:
                        time.sleep(
                            wait_between_attempts * 2
                        )  # wait longer between retry attempts

            if not deleted:
                logger.warning(
                    f"Gave up deleting version {version} after {max_deletion_attempts} attempts"
                )

        return result

    except Exception as e:
        error_msg = f"Error while cleaning up older deployments: {str(e)}"
        logger.warning(error_msg)

        if raise_on_error:
            raise AgentDeploymentError(error_msg) from e

        return result


def _wait_for_endpoint_ready(endpoint_name: str) -> None:
    """Wait for endpoint to be ready."""
    logger.info(f"Starting to wait for endpoint {endpoint_name}")
    w = WorkspaceClient()
    iteration = 0

    while True:
        endpoint_state = w.serving_endpoints.get(endpoint_name).state
        ready_state = endpoint_state.ready
        config_state = endpoint_state.config_update

        if (
            ready_state == EndpointStateReady.NOT_READY
            or config_state == EndpointStateConfigUpdate.IN_PROGRESS
        ):
            iteration += 1
            if iteration % 4 == 0:
                logger.info(
                    f"Still waiting for endpoint... (Ready state: {ready_state}, Config state: {config_state})"
                )
            time.sleep(30)
        else:
            break

    logger.info(f"Endpoint {endpoint_name} is ready")


def _set_permissions(model_name: str, permissions: Union[dict, list[dict]]) -> None:
    """Set agent permissions for one or more permission configurations.

    Args:
        model_name: Full UC model name
        permissions: Permission configuration dictionary or list of dictionaries with structure:
            Single permission:
            {
                "users": List[str],  # List of user/group names
                "permission_level": str  # One of CAN_VIEW, CAN_QUERY, CAN_MANAGE
            }

            Multiple permissions:
            [
                {
                    "users": List[str],
                    "permission_level": str
                },
                ...
            ]
    """
    logger.info(f"Setting permissions for {model_name}")
    logger.debug(f"Permission config: {permissions}")

    try:
        permission_configs = (
            permissions if isinstance(permissions, list) else [permissions]
        )

        for permission_config in permission_configs:
            users = permission_config.get("users")
            permission_level = permission_config.get("permission_level")

            if not users or not permission_level:
                raise ValueError(
                    "Both 'users' and 'permission_level' must be specified in each permission config"
                )

            logger.info(
                f"Setting {permission_level} permission for users: {', '.join(users)}"
            )

            agents.set_permissions(
                model_name=model_name,
                users=users,
                permission_level=getattr(agents.PermissionLevel, permission_level),
            )

        logger.info("All permissions set successfully")

    except ValueError as e:
        error_msg = f"Invalid permission configuration: {str(e)}"
        logger.error(error_msg)
        raise AgentDeploymentError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to set permissions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise AgentDeploymentError(error_msg) from e
