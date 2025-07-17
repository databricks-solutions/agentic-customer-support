"""Routing accuracy evaluation scorers for telco support agent."""

from typing import Any, Optional

from databricks.agents.evals import metric

from telco_support_agent.evaluation.scorers.base_scorer import GuidelinesScorer


class RoutingAccuracyScorer(GuidelinesScorer):
    guidelines = [
        "If the query is about customer information of the account, plans, or devices, it should be routed to the 'account' agent.",
        "If the query is about bills, payments, charges, refunds, payment methods, or billing disputes, it should be routed to the 'billing' agent.",
        "If the query is about technical issues, internet problems, equipment troubleshooting, service outages, or connectivity problems, it should be routed to the 'tech_support' agent.",
        "If the query is about plan information, upgrades, downgrades, new services, or feature questions, it should be routed to the 'product' agent.",
        "The routed_agent must be the most appropriate specialist for the type of request in the query.",
    ]

    def __init__(self):
        super().__init__("routing_accuracy", self.guidelines)

    def get_context(
        self,
        inputs: dict[str, Any],
        outputs: Optional[Any],
        trace: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Function to create the context used by the guidelines judge.

        Args:
            inputs: Model input represented as dict.
            outputs: Model output represented as dict.
            trace: Model execution trace as MLflow entity.
        """
        query = str(inputs["input"][0]["content"])
        context = {
            "query": query,
            "routed_agent": outputs["custom_outputs"]["routing"]["agent_type"],
        }
        return context

    def get_custom_metric(self):
        """Implementation of custom metric for offline evaluation."""

        @metric
        def routing_accuracy(
            request: dict[str, Any], response: Optional[dict[str, Any]]
        ):
            from mlflow.genai.judges import meets_guidelines

            query = str(request["request"]["input"][0]["content"])

            context = {
                "query": query,
                "routed_agent": response["custom_outputs"]["routing"]["agent_type"],
            }
            guidelines = [
                "If the query is about account management, profile updates, personal information, or login issues, it should be routed to the 'account' agent",
                "If the query is about bills, payments, charges, refunds, payment methods, or billing disputes, it should be routed to the 'billing' agent",
                "If the query is about technical issues, internet problems, equipment troubleshooting, service outages, or connectivity problems, it should be routed to the 'tech_support' agent",
                "If the query is about plan information, upgrades, downgrades, new services, or feature questions, it should be routed to the 'product' agent",
                "The routed_agent must be the most appropriate specialist for the type of request in the query",
            ]
            feedback = meets_guidelines(
                guidelines=guidelines, context=context, name="routing_accuracy"
            )
            return feedback

        return routing_accuracy
