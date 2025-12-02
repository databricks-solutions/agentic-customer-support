"""Production monitoring setup for deployed agents."""

from mlflow.genai.scorers import ScorerSamplingConfig, delete_scorer, list_scorers

from telco_support_agent.evaluation import (
    REGISTERABLE_SCORERS,
    SCORERS,
    SCORER_CONFIGS,
)
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentMonitoringError(Exception):
    """Raised when agent monitoring creation operations fail."""

    pass


def setup_production_monitoring(
    experiment_id: str,
    replace_existing: bool = True,
) -> list:
    """Set up production monitoring by registering and starting scorers.

    Registers all make_judge and code-based scorers. Predefined scorers (Safety,
    RelevanceToQuery) are used directly without registration.

    Args:
        experiment_id: MLflow experiment ID for the deployed agent.
        replace_existing: Whether to delete and recreate existing scorers.

    Returns:
        List of registered and started scorer instances.

    Raises:
        AgentMonitoringError: If scorer setup fails.
    """
    try:
        # Delete existing scorers if replacing
        if replace_existing:
            try:
                existing_scorers = list_scorers(experiment_id=experiment_id)
                logger.info(
                    f"Found {len(existing_scorers)} existing scorers for experiment: {experiment_id}"
                )
                logger.info("Deleting existing scorers for replacement...")
                for scorer in existing_scorers:
                    delete_scorer(name=scorer.name)
                    logger.info(f"Deleted scorer: {scorer.name}")
            except ValueError:
                logger.info(f"No existing scorers found for experiment: {experiment_id}")

        # Register and start scorers
        registered_scorers = []

        logger.info(
            f"Total scorers: {len(SCORERS)} (registering {len(REGISTERABLE_SCORERS)})"
        )

        for scorer in REGISTERABLE_SCORERS:
            scorer_name = scorer.name
            config = SCORER_CONFIGS.get(scorer_name, {"sample_rate": 1.0})
            sample_rate = config["sample_rate"]

            logger.info(f"Registering scorer: {scorer_name}")

            # Register scorer
            registered = scorer.register(name=scorer_name)

            # Start monitoring with sampling config
            started = registered.start(
                sampling_config=ScorerSamplingConfig(sample_rate=sample_rate)
            )

            registered_scorers.append(started)
            logger.info(
                f"Scorer {scorer_name} monitoring active (sample_rate={sample_rate})"
            )

        logger.info(
            f"Production monitoring setup complete: {len(registered_scorers)} scorers registered"
        )
        return registered_scorers

    except Exception as e:
        error_msg = f"Failed to setup production monitoring: {str(e)}"
        logger.error(error_msg)
        raise AgentMonitoringError(error_msg) from e
