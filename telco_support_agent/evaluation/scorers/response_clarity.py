"""Response clarity evaluation scorers for telco support agent."""

from mlflow.genai.scorers import scorer

from telco_support_agent.evaluation.scorers.base_scorer import PromptScorer


class ResponseClarityScorer(PromptScorer):
    prompt = """Evaluate the clarity and understandability of this telco customer service response:

            Customer Request: {{request}}
            Agent Response: {{response}}

            Rate the clarity of the response based on these criteria:
            - Language is clear and easy to understand
            - Technical terms are explained when used
            - Information is well-organized and logical
            - Instructions or next steps are specific and actionable
            - Response length is appropriate (not too verbose or too brief)

            Choose the most appropriate clarity rating:

            [[excellent]]: Crystal clear, perfectly organized, easy to follow for any customer
            [[good]]: Clear and understandable with minor room for improvement
            [[adequate]]: Generally clear but some parts could be clearer or better organized
            [[poor]]: Confusing, unclear language, or difficult to understand"""

    numeric_values = {"excellent": 1.0, "good": 0.75, "adequate": 0.5, "poor": 0.0}

    def __init__(self):
        super().__init__(
            name="response_clarity",
            sample_rate=1.0,
            prompt_template=self.prompt,
            numeric_values=self.numeric_values,
        )

    def get_online_scorer(self):
        """Implementation of custom metric for offline evaluation."""

        @scorer
        def response_clarity(inputs, outputs):
            from databricks.agents.evals.judges import custom_prompt_judge

            prompt = """Evaluate the clarity and understandability of this telco customer service response:

            Customer Request: {{request}}
            Agent Response: {{response}}

            Rate the clarity of the response based on these criteria:
            - Language is clear and easy to understand
            - Technical terms are explained when used
            - Information is well-organized and logical
            - Instructions or next steps are specific and actionable
            - Response length is appropriate (not too verbose or too brief)

            Choose the most appropriate clarity rating:

            [[excellent]]: Crystal clear, perfectly organized, easy to follow for any customer
            [[good]]: Clear and understandable with minor room for improvement
            [[adequate]]: Generally clear but some parts could be clearer or better organized
            [[poor]]: Confusing, unclear language, or difficult to understand"""
            numeric_values = {
                "excellent": 1.0,
                "good": 0.75,
                "adequate": 0.5,
                "poor": 0.0,
            }
            judge = custom_prompt_judge(
                name="response_clarity",
                prompt_template=prompt,
                numeric_values=numeric_values,
            )
            request = str(inputs["request"]["input"])
            response = str(outputs["output"][-1])

            feedback = judge(request=request, response=response)
            return feedback

        return response_clarity
