"""Telco support agent evaluation scorers."""

# TODO: at the moment, the custom scorers cannot be used in monitoring
# as such the current workaround is to use @metrics to define judges for monitoring
# future API release will allow for custom scorers to be used in monitoring
# when this is the case, we can remove the @metrics and use the custom scorers directly

from telco_support_agent.evaluation.scorers.base_scorer import BuiltInScorerWrapper
from telco_support_agent.evaluation.scorers.brand_compliance import (
    BrandComplianceScorer,
)
from telco_support_agent.evaluation.scorers.data_privacy import DataPrivacyScorer
from telco_support_agent.evaluation.scorers.query_resolution import (
    QueryResolutionScorer,
)
from telco_support_agent.evaluation.scorers.response_clarity import (
    ResponseClarityScorer,
)
from telco_support_agent.evaluation.scorers.routing_accuracy import (
    RoutingAccuracyScorer,
)
from telco_support_agent.evaluation.scorers.tool_accuracy import ToolAccuracyScorer

SCORERS = [
    BrandComplianceScorer(),
    DataPrivacyScorer(),
    QueryResolutionScorer(),
    ResponseClarityScorer(),
    RoutingAccuracyScorer(),
    ToolAccuracyScorer(),
]
__all__ = ["SCORERS", "BuiltInScorerWrapper"]
