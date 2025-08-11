from abc import ABC, abstractmethod
from typing import Any, Optional

from mlflow.entities import Feedback, Trace
from mlflow.genai.judges import custom_prompt_judge, meets_guidelines
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer


class BaseScorer(ABC):
    def __init__(self, name, sample_rate):
        self.name = name
        self.sample_rate = sample_rate

    @abstractmethod
    def get_offline_scorer(self):
        """Implementation of custom scorer for online evaluation."""
        pass

    @abstractmethod
    def get_online_scorer(self):
        """Implementation of custom metric for offline evaluation."""
        pass


class BuiltInScorerWrapper:
    def __init__(self, name: str, sample_rate: float, scorer: BuiltInScorer):
        self.name = name
        self.sample_rate = sample_rate
        self.scorer = scorer


class PromptScorer(BaseScorer):
    def __init__(
        self,
        name,
        sample_rate,
        prompt_template: str,
        numeric_values: dict[str, int | float],
    ):
        super().__init__(name, sample_rate)
        self.prompt_template = prompt_template
        self.numeric_values = numeric_values

    def get_feedback_from_prompt(
        self, inputs: dict[str, Any], outputs: Optional[Any]
    ) -> Feedback:
        """Logic to generate feedback from model input and output using custom prompt judge.

        Args:
            inputs: Model input represented as dict.
            outputs: Model output represented as dict.
        """
        try:
            request = str(inputs["input"])
            response = str(outputs["output"][-1])

            assert self.prompt_template is not None, (
                "Getting feedback from prompt requires a prompt template definition."
            )
            judge = custom_prompt_judge(
                name=self.name,
                prompt_template=self.prompt_template,
                numeric_values=self.numeric_values,
            )
            feedback = judge(request=request, response=response)
            return feedback
        except Exception as e:
            return Feedback(
                value="no", rationale=f"Error evaluating {self.name}: {str(e)}"
            )

    def get_offline_scorer(self):
        """Implementation of custom scorer for online evaluation."""

        @scorer(name=self.name)
        def internal_scorer(inputs: dict[str, Any], outputs: Optional[Any]):
            return self.get_feedback_from_prompt(inputs, outputs)

        return internal_scorer

    def get_online_scorer(self):
        """Implementation of custom metric for offline evaluation."""
        pass


class GuidelinesScorer(BaseScorer):
    def __init__(self, name, sample_rate, guidelines: list[str]):
        super().__init__(name, sample_rate)
        self.guidelines = guidelines

    def get_context(
        self,
        inputs: dict[str, Any],
        outputs: Optional[Any],
        trace: Optional[Trace] = None,
    ) -> dict[str, Any]:
        """Function to create the context used by the guidelines judge.

        Args:
            inputs: Model input represented as dict.
            outputs: Model output represented as dict.
            trace: Model execution trace as MLflow entity.
        """
        request_text = str(inputs["input"])
        response_text = str(outputs["output"][-1])
        context = {"request": request_text, "response": response_text}
        return context

    def get_feedback_from_guidelines(self, context: dict[str, Any]) -> Feedback:
        """Logic to generate feedback from model input and output using guidelines judge.

        Args:
            context: Context used by the guidelines judge.
        """
        try:
            # Guidelines for the judge to evaluate brand compliance.
            feedback = meets_guidelines(
                name=self.name, guidelines=self.guidelines, context=context
            )

            return feedback

        except Exception as e:
            return Feedback(
                value="no", rationale=f"Error evaluating {self.name}: {str(e)}"
            )

    def get_offline_scorer(self):
        """Implementation of custom scorer for online evaluation."""

        @scorer(name=self.name)
        def internal_scorer(
            inputs: dict[str, Any],
            outputs: Optional[Any],
            trace: Optional[Trace] = None,
        ):
            context = self.get_context(inputs, outputs, trace)
            return self.get_feedback_from_guidelines(context)

        return internal_scorer

    def get_online_scorer(self):
        """Implementation of custom scorer for online evaluation."""
        pass
