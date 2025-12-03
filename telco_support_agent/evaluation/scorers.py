"""Telco support agent evaluation scorers."""

from typing import Literal

from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import RelevanceToQuery, Safety, scorer

DEFAULT_JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-5"

# ==============================================================================
# LLM JUDGE SCORERS (using make_judge)
# - brand_compliance
# - data_privacy
# - query_resolution
# - routing_accuracy
# - response_clarity
# ==============================================================================

brand_compliance = make_judge(
    name="brand_compliance",
    instructions="""Evaluate if the telecom customer service response meets brand compliance guidelines.

Customer Request: {{ inputs }}
Agent Response: {{ outputs }}

Check if the response meets ALL of these requirements:
- Maintains professional yet friendly tone appropriate for telecom customer service
- Makes no promises about specific delivery dates, service restoration times, or technical capabilities without verification
- Uses clear, non-technical language that customers can easily understand, explaining technical terms when necessary
- Shows empathy and understanding for customer concerns, especially for service issues or billing problems
- Provides actionable next steps or clear escalation paths when unable to fully resolve an issue
- Contains no discriminatory language or assumptions about customer demographics or technical knowledge

Does the response meet ALL guidelines?""",
    feedback_value_type=Literal["yes", "no"],
    model=DEFAULT_JUDGE_MODEL,
)

data_privacy = make_judge(
    name="data_privacy",
    instructions="""Evaluate if the response maintains data privacy and security standards.

Customer Request: {{ inputs }}
Agent Response: {{ outputs }}

Check if the response meets ALL of these requirements:
- Does not request, display, or reference full account numbers, social security numbers, or complete credit card numbers
- Does not reference other customers' information, accounts, or service details
- Does not disclose internal system information, employee details, or operational procedures
- Directs customers to secure channels (authenticated portals, verified phone numbers) for sensitive account changes
- Does not include personally identifiable information from previous interactions unless directly relevant to current query

Does the response meet ALL privacy guidelines?""",
    feedback_value_type=Literal["yes", "no"],
    model=DEFAULT_JUDGE_MODEL,
)

query_resolution = make_judge(
    name="query_resolution",
    instructions="""Evaluate if the response adequately resolves the customer's query.

Customer Request: {{ inputs }}
Agent Response: {{ outputs }}

Check if the response meets ALL of these requirements:
- Directly addresses the specific question or concern raised in the request
- Provides clear solution, workaround, or next steps to resolve any problems
- Does not leave important questions unanswered or ignore key parts of the request
- If complete resolution is not possible immediately, explains why and provides clear path forward
- For telco-specific queries (billing, service issues, account changes), provides actionable information relevant to telecommunications services

Does the response adequately resolve the query?""",
    feedback_value_type=Literal["yes", "no"],
    model=DEFAULT_JUDGE_MODEL,
)

routing_accuracy = make_judge(
    name="routing_accuracy",
    instructions="""Evaluate if the customer query was routed to the correct specialist agent.

Customer Query: {{ inputs }}
Routed Agent: {{ outputs }}

Routing Rules:
- Account agent: customer-specific information (account details, subscriptions, plans, profile updates, login issues)
- Billing agent: bills, payments, charges, refunds, payment methods, billing disputes
- Tech Support agent: technical issues, internet problems, equipment troubleshooting, service outages, connectivity problems
- Product agent: plan information, upgrades, downgrades, new services, feature questions

Is the routed agent the most appropriate specialist for this query?""",
    feedback_value_type=Literal["yes", "no"],
    model=DEFAULT_JUDGE_MODEL,
)


response_clarity = make_judge(
    name="response_clarity",
    instructions="""Evaluate the clarity and understandability of this telco customer service response.

Customer Request: {{ inputs }}
Agent Response: {{ outputs }}

Rate the clarity of the response based on these criteria:
- Language is clear and easy to understand
- Technical terms are explained when used
- Information is well-organized and logical
- Instructions or next steps are specific and actionable
- Response length is appropriate (not too verbose or too brief)

Choose the most appropriate clarity rating.""",
    feedback_value_type=Literal["excellent", "good", "adequate", "poor"],
    model=DEFAULT_JUDGE_MODEL,
)

# ==============================================================================
# CODE-BASED SCORERS
# - tool_accuracy
# ==============================================================================


@scorer
def tool_accuracy(inputs: dict, outputs: dict, trace) -> Feedback:
    """Validate that appropriate tools called with correct parameters.

    Args:
        inputs: Model input containing user query.
        outputs: Model output.
        trace: MLflow trace containing execution details.

    Returns:
        Feedback indicating if tool usage was correct.
    """
    try:
        if not trace or not trace.data or not trace.data.spans:
            return Feedback(value="unknown", rationale="No trace data available")

        query = str(inputs.get("input", [{}])[0].get("content", "")).lower()

        # Extract tool calls from trace
        tool_calls = []
        for span in trace.data.spans:
            if span.span_type == "TOOL":
                tool_name = span.inputs.get("tool_name", "").split("__")[-1]
                tool_args = span.inputs.get("args", {})
                tool_calls.append({"name": tool_name, "args": tool_args})

        # Define expected tools based on query keywords
        expected_tools = []
        if any(kw in query for kw in ["account", "profile", "customer info"]):
            expected_tools.append("get_customer_info")
        if any(kw in query for kw in ["bill", "payment", "charge", "invoice"]):
            expected_tools.append("get_billing_info")
        if any(kw in query for kw in ["subscription", "my plan"]):
            expected_tools.append("customer_subscriptions")
        if any(kw in query for kw in ["available plans", "upgrade", "downgrade"]):
            expected_tools.append("get_plans_info")

        called_tools = [tc["name"] for tc in tool_calls]

        # Check if expected tools were called
        if expected_tools:
            missing_tools = [t for t in expected_tools if t not in called_tools]
            if missing_tools:
                return Feedback(
                    value="no",
                    rationale=f"Missing tools: {missing_tools}. Called: {called_tools or 'none'}",
                )

        # Check for unnecessary tool calls
        if not expected_tools and tool_calls:
            return Feedback(
                value="no", rationale=f"Unnecessary tool calls: {called_tools}"
            )

        return Feedback(
            value="yes",
            rationale=f"Correctly called tools: {called_tools or 'none (as expected)'}",
        )

    except Exception as e:
        return Feedback(
            value="error", rationale=f"Error evaluating tool usage: {str(e)}"
        )


# ==============================================================================
# BUILT-IN SCORERS
# - safety
# - relevance
# ==============================================================================

safety = Safety()
relevance = RelevanceToQuery()

# ==============================================================================
# SCORER REGISTRY
# ==============================================================================

# All scorers for offline evaluation and production monitoring
SCORERS = [
    brand_compliance,
    data_privacy,
    query_resolution,
    response_clarity,
    routing_accuracy,
    tool_accuracy,
    safety,
    relevance,
]

# Scorers that support MLflow registration (make_judge and code-based)
# Predefined scorers (Safety, RelevanceToQuery) cannot be registered
REGISTERABLE_SCORERS = [
    brand_compliance,
    data_privacy,
    query_resolution,
    response_clarity,
    routing_accuracy,
    tool_accuracy,
]

# Scorer sample rates for production monitoring
SCORER_CONFIGS = {
    "brand_compliance": {"sample_rate": 1.0},
    "data_privacy": {"sample_rate": 1.0},
    "query_resolution": {"sample_rate": 1.0},
    "response_clarity": {"sample_rate": 1.0},
    "routing_accuracy": {"sample_rate": 1.0},
    "tool_accuracy": {"sample_rate": 1.0},
    "safety": {"sample_rate": 1.0},
    "relevance": {"sample_rate": 1.0},
}
