"""Monitoring utils for deployed agents."""

from typing import Optional

from mlflow.genai.scorers import ScorerSamplingConfig, delete_scorer, list_scorers

from telco_support_agent.evaluation.scorers.base_scorer import (
    BaseScorer,
    BuiltInScorerWrapper,
)
from telco_support_agent.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentMonitoringError(Exception):
    """Raised when agent monitoring creation operations fail."""

    pass


def setup_agent_scorers(
    experiment_id: str,
    replace_existing: bool = True,
    builtin_scorers: Optional[list[BuiltInScorerWrapper]] = None,
    custom_scorers: Optional[list[BaseScorer]] = None,
) -> any:
    """Set up a set of scorers for the deployed agent with custom scorers and built-in scorers.

    Args:
        experiment_id: MLflow experiment ID.
        replace_existing: Whether to replace the existing scorers.
        builtin_scorers: List of built-in scorers to use.
        custom_scorers: List of custom scorer functions to use

    Returns:
        Created custom scorers.

    Raises:
        AgentMonitoringError: If the scorer's creation fails
    """
    try:
        if replace_existing:
            try:
                actual_scorers = list_scorers(experiment_id=experiment_id)
                logger.info(
                    f"Found existing scorers: {actual_scorers} for experiment: {experiment_id}"
                )
                logger.info("Deleting existing scorers for replacement...")
                for scorer in actual_scorers:
                    delete_scorer(name=scorer.name)
            except ValueError:
                logger.info(
                    f"No existing scorers found for experiment: {experiment_id}"
                )

        logger.info(f"Creating scorers for experiment: {experiment_id}")

        scorers_result = []

        all_scores = builtin_scorers + custom_scorers

        if all_scores:
            scorers_mapping = {
                scorer.name: scorer
                for scorer in list_scorers(experiment_id=experiment_id)
            }
            logger.info("Adding custom and built-in scorers to experiment.")
            for scorer in all_scores:
                created_scorer = create_scorer(
                    scorer=scorer, scorers_mapping=scorers_mapping
                )
                scorers_result.append(created_scorer)

        logger.info(f"Experiment configured with {len(all_scores)} scorers.")

        return scorers_result

    except Exception as e:
        error_msg = f"Failed to setup experiment scorers: {str(e)}"
        logger.error(error_msg)
        raise AgentMonitoringError(error_msg) from e


def create_scorer(scorer, scorers_mapping):
    """Function to create a new scorer.

    Args:
        scorer: BaseScorer or BuiltInScorerWrapper object with scorer function and metadata.
        scorers_mapping: Dict with pre-existing scorers on the experiment.

    Returns:
        Created scorer.
    """
    if isinstance(scorer, BaseScorer):
        scorer_fn = scorer.get_online_scorer()
    else:
        scorer_fn = scorer.scorer

    scorer_name = scorer.name
    sample_rate = scorer.sample_rate

    if scorer_name in scorers_mapping:
        # Scorer already exists. Remove and create a scorer.
        # Doing this in case the scorer code changed.
        delete_scorer(name=scorer_name)
    created_scorer = scorer_fn.register(name=scorer_name)
    created_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=sample_rate))
    logger.info(
        f"Adding scorer: {scorer_name} to experiment with sample rate: {sample_rate}."
    )
    return created_scorer
