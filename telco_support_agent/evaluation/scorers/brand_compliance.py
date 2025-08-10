"""Brand compliance evaluation scorers for telco support agent."""

from mlflow.genai.scorers import scorer

from telco_support_agent.evaluation.scorers.base_scorer import GuidelinesScorer


class BrandComplianceScorer(GuidelinesScorer):
    guidelines = [
        "The response must maintain a professional yet friendly tone appropriate for telecommunications customer service.",
        "The response must not make promises about specific delivery dates, service restoration times, or technical capabilities without verification.",
        "The response must use clear, non-technical language that customers can easily understand, explaining technical terms when necessary.",
        "The response must show empathy and understanding for customer concerns, especially when addressing service issues or billing problems.",
        "The response must provide actionable next steps or clear escalation paths when unable to fully resolve an issue.",
        "The response must not include any discriminatory language or make assumptions about customer demographics or technical knowledge.",
    ]

    def __init__(self):
        super().__init__("brand_compliance", 1.0, self.guidelines)

    def get_online_scorer(self):
        """Implementation of custom metric for offline evaluation."""

        @scorer
        def brand_compliance(inputs, outputs):
            from mlflow.genai.judges import meets_guidelines

            request = str(inputs["request"]["input"])
            response = str(outputs["output"][-1])

            context = {"request": request, "response": response}

            feedback = meets_guidelines(
                guidelines=[
                    "The response must maintain a professional yet friendly tone appropriate for telecommunications customer service.",
                    "The response must not make promises about specific delivery dates, service restoration times, or technical capabilities without verification.",
                    "The response must use clear, non-technical language that customers can easily understand, explaining technical terms when necessary.",
                    "The response must show empathy and understanding for customer concerns, especially when addressing service issues or billing problems.",
                    "The response must provide actionable next steps or clear escalation paths when unable to fully resolve an issue.",
                    "The response must not include any discriminatory language or make assumptions about customer demographics or technical knowledge.",
                ],
                context=context,
                name="brand_compliance",
            )
            return feedback

        return brand_compliance
