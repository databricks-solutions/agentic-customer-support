"""Query resolution evaluation scorers for telco support agent."""

from mlflow.genai.scorers import scorer


from telco_support_agent.evaluation.scorers.base_scorer import GuidelinesScorer


class QueryResolutionScorer(GuidelinesScorer):
    guidelines = [
        "The response must directly address the specific question or concern raised in the request",
        "If the customer has a problem, the response must provide a clear solution, workaround, or next steps to resolve it",
        "The response must not leave important questions unanswered or ignore key parts of the customer's request",
        "If complete resolution is not possible immediately, the response must explain why and provide a clear path forward",
        "For telco-specific queries (billing, service issues, account changes), the response must provide actionable information relevant to telecommunications services",
    ]

    def __init__(self):
        super().__init__("query_resolution", 1.0, self.guidelines)

    def get_online_scorer(self):
        """Implementation of custom metric for offline evaluation."""

        @scorer
        def query_resolution(inputs, outputs):
            from mlflow.genai.judges import meets_guidelines

            request = str(inputs["request"]["input"])
            response = str(outputs["output"][-1])

            context = {"request": request, "response": response}
            feedback = meets_guidelines(
                guidelines=[
                    "The response must directly address the specific question or concern raised in the request",
                    "If the customer has a problem, the response must provide a clear solution, workaround, or next steps to resolve it",
                    "The response must not leave important questions unanswered or ignore key parts of the customer's request",
                    "If complete resolution is not possible immediately, the response must explain why and provide a clear path forward",
                    "For telco-specific queries (billing, service issues, account changes), the response must provide actionable information relevant to telecommunications services",
                ],
                context=context,
                name="query_resolution",
            )
            return feedback

        return query_resolution
