"""Supervisor agent to orchestrate specialized sub-agents."""

from collections.abc import Generator
from typing import NamedTuple, Optional, Union
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from mlflow.models import set_model
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from telco_support_agent.agents.account import AccountAgent
from telco_support_agent.agents.agent_types import AgentType
from telco_support_agent.agents.base_agent import BaseAgent
from telco_support_agent.agents.billing import BillingAgent
from telco_support_agent.agents.product import ProductAgent
from telco_support_agent.agents.tech_support import TechSupportAgent
from telco_support_agent.agents.utils.message_formatting import (
    extract_user_query,
)
from telco_support_agent.agents.utils.topic_utils import (
    load_topics_from_yaml,
    topic_classification,
)
from telco_support_agent.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class AgentExecutionResult(NamedTuple):
    """Result of agent execution preparation."""

    sub_agent: Optional[BaseAgent]
    agent_type: AgentType
    query: str
    custom_outputs: dict
    error_response: Optional[dict] = None


class SupervisorAgent(BaseAgent):
    """Supervisor agent to orchestrate specialized sub-agents.

    This agent analyzes customer queries and routes them to the appropriate
    sub-agent based on query content and intent.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        config_dir: Optional[str] = None,
        disable_tools: Optional[list[str]] = None,
        uc_config: Optional[UCConfig] = None,
    ):
        """Initialize supervisor agent.

        Args:
            llm_endpoint: Optional override for LLM endpoint
            config_dir: Optional directory for config files
            disable_tools: Optional list of tool names to disable.
                Can be either simple names (e.g., 'get_usage_info') or full UC function
                names (e.g., 'telco_customer_support_dev.agent.get_usage_info').
            uc_config: Optional UC configuration for Unity Catalog resources
        """
        # NOTE: don't need UC function tools for supervisor
        # the routing logic will be implemented directly in this class
        super().__init__(
            agent_type="supervisor",
            llm_endpoint=llm_endpoint,
            config_dir=config_dir,
            tools=[],  # no tools needed for routing
            uc_config=uc_config,
        )

        self._sub_agents = {}
        if disable_tools is None:
            disable_tools = self._load_disable_tools_from_artifact()
        self.disable_tools = disable_tools or []
        self._topic_categories = load_topics_from_yaml()
        if self.disable_tools:
            logger.info(
                f"Supervisor configured with disabled tools: {self.disable_tools}"
            )

    def get_description(self) -> str:
        """Return a description of this agent."""
        return "Supervisor agent that routes customer queries to specialized sub-agents"

    def _get_sub_agent(self, agent_type: Union[AgentType, str]) -> Optional[BaseAgent]:
        """Get or initialize a sub-agent if implemented.

        Args:
            agent_type: Type of sub-agent to get

        Returns:
            Initialized sub-agent or None if not implemented
        """
        agent_type_str = (
            agent_type.value if isinstance(agent_type, AgentType) else agent_type
        )

        if agent_type_str in self._sub_agents:
            return self._sub_agents[agent_type_str]

        agent_type_enum = (
            agent_type
            if isinstance(agent_type, AgentType)
            else AgentType.from_string(agent_type)
        )

        agents_classes = {
            AgentType.ACCOUNT: AccountAgent,
            AgentType.BILLING: BillingAgent,
            AgentType.TECH_SUPPORT: TechSupportAgent,
            AgentType.PRODUCT: ProductAgent,
        }

        if agent_type_enum in agents_classes:
            try:
                # Prepare agent initialization kwargs
                agent_kwargs = {
                    "llm_endpoint": self.llm_endpoint,
                    "disable_tools": self.disable_tools,
                    "uc_config": self.config.uc_config,
                }

                agent = agents_classes[agent_type_enum](**agent_kwargs)
                self._sub_agents[agent_type_str] = agent
                logger.info(f"Initialized {agent_type_str} agent")
                return agent
            except Exception as e:
                logger.error(f"Error initializing {agent_type_str} agent: {str(e)}")
                raise
        else:
            logger.warning(f"{agent_type_str.capitalize()} agent not implemented yet.")
            return None

    @mlflow.trace(span_type=SpanType.AGENT)
    def route_query(self, query: str) -> AgentType:
        """Determine which sub-agent should handle the query.

        Args:
            query: User query to classify

        Returns:
            The agent type that should handle this query
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = self.call_llm(messages)
            agent_type_str = response.get("content", "").strip().lower()

            try:
                agent_type = AgentType.from_string(agent_type_str)
                logger.info(f"Routing query to {agent_type.value} agent")
                return agent_type
            except ValueError:
                logger.warning(
                    f"LLM returned invalid agent type: {agent_type_str}. Falling back to account agent."
                )
                return AgentType.ACCOUNT

        except Exception as e:
            logger.error(
                f"Error in routing query: {str(e)}. Falling back to account agent."
            )
            return AgentType.ACCOUNT

    @mlflow.trace(span_type=SpanType.LLM)
    def _classify_query(self, query: str) -> dict[str, str]:
        classification_result = topic_classification(query, self._topic_categories)
        detected_topic = classification_result.get("topic")
        if detected_topic is not None:
            mlflow.update_current_trace(tags={"topic": detected_topic})
        return classification_result

    def _prepare_agent_execution(
        self, request: ResponsesAgentRequest
    ) -> AgentExecutionResult:
        """Prepare for agent execution by handling routing / validation.

        Consolidates common logic between predict and predict_stream.

        Args:
            request: The request containing user messages

        Returns:
            AgentExecutionResult containing all necessary information for execution
        """
        # extract the user query from the input
        user_query = extract_user_query(request.input)
        if not user_query:
            # no user messages found, return error response
            error_response = {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "No user query found in the request.",
                    }
                ],
                "id": str(uuid4()),
            }
            return AgentExecutionResult(
                sub_agent=None,
                agent_type=AgentType.ACCOUNT,  # placeholder
                query="",
                custom_outputs={},
                error_response=error_response,
            )

        # determine which agent should handle query
        agent_type = self.route_query(user_query)

        # prepare custom outputs with routing decision
        custom_outputs = request.custom_inputs.copy() if request.custom_inputs else {}
        custom_outputs["routing"] = {
            "agent_type": agent_type.value,
        }

        # add disabled tools info to custom outputs
        if self.disable_tools:
            custom_outputs["routing"]["disable_tools"] = self.disable_tools

        # get sub-agent
        sub_agent = self._get_sub_agent(agent_type)

        # classify query based on topic categories
        classification_result = self._classify_query(user_query)
        custom_outputs["topic"] = classification_result.get("topic")

        # if sub-agent not implemented, prepare non-response
        if sub_agent is None:
            error_response = self.generate_non_response(agent_type, user_query)
            return AgentExecutionResult(
                sub_agent=None,
                agent_type=agent_type,
                query=user_query,
                custom_outputs=custom_outputs,
                error_response=error_response,
            )

        return AgentExecutionResult(
            sub_agent=sub_agent,
            agent_type=agent_type,
            query=user_query,
            custom_outputs=custom_outputs,
            error_response=None,
        )

    def _handle_direct_response(
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        """Handle request directly without routing when intelligence is disabled.

        Args:
            request: The request containing user messages

        Returns:
            Direct response from supervisor as a conversational agent
        """
        logger.debug("Handling request directly without sub-agent routing")

        # Extract user query
        user_query = extract_user_query(request.input)
        if not user_query:
            error_response = {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'm sorry, I didn't receive a clear question. Could you please rephrase your request?",
                    }
                ],
                "id": str(uuid4()),
            }
            return ResponsesAgentResponse(
                output=[error_response],
                custom_outputs=request.custom_inputs.copy()
                if request.custom_inputs
                else {},
            )

        # Use the supervisor's LLM to generate a direct response
        # Convert MLflow Message objects to dictionaries
        messages = [i.model_dump() for i in request.input]

        # Add a system message to make it conversational but limited
        system_message = {
            "role": "system",
            "content": "You are a helpful customer service representative. You can provide general information and assistance, but you cannot access specific customer data or perform account operations. If a customer asks for specific account information, politely explain that you'll need them to contact customer service for detailed account access.",
        }

        # Insert system message at the beginning
        messages_with_system = [system_message] + messages

        try:
            # Call LLM directly without tools (intelligence disabled)
            # Follow the same pattern as route_query - don't pass request to avoid databricks_options processing
            llm_response = self.call_llm(messages_with_system)

            response_content = llm_response.get(
                "content",
                "I apologize, but I'm unable to provide a response at the moment. Please try again.",
            )

            response_message = {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_content,
                    }
                ],
                "id": str(uuid4()),
            }

            # Prepare custom outputs without routing info since no routing occurred
            custom_outputs = (
                request.custom_inputs.copy() if request.custom_inputs else {}
            )
            custom_outputs["routing"] = {
                "agent_type": "supervisor_direct",
                "intelligence_enabled": False,
            }

            self.update_trace_preview(
                user_query=user_query,
                response_data={"output": [response_message]},
            )

            return ResponsesAgentResponse(
                output=[response_message],
                custom_outputs=custom_outputs,
            )

        except Exception as e:
            logger.error(f"Error generating direct response: {e}")
            error_response = {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                    }
                ],
                "id": str(uuid4()),
            }
            return ResponsesAgentResponse(
                output=[error_response],
                custom_outputs=request.custom_inputs.copy()
                if request.custom_inputs
                else {},
            )

    def _handle_direct_response_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle streaming request directly without routing when intelligence is disabled.

        Args:
            request: The request containing user messages

        Yields:
            ResponsesAgentStreamEvent objects for direct response
        """
        logger.debug("Handling streaming request directly without sub-agent routing")

        # Get the direct response
        direct_response = self._handle_direct_response(request)

        # Convert the response to streaming events
        for output_item in direct_response.output:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", item=output_item
            )

    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process user query, route to, and yield response from sub-agent.

        Args:
            request: The request containing user messages

        Returns:
            The response from the sub-agent
        """
        # Extract intelligence_enabled from custom_inputs
        intelligence_enabled = (
            request.custom_inputs.get("intelligence_enabled", True)
            if request.custom_inputs
            else True
        )

        if not intelligence_enabled:
            logger.info("Intelligence disabled - using direct response")
            return self._handle_direct_response(request)

        execution_result = self._prepare_agent_execution(request)

        if execution_result.error_response:
            self.update_trace_preview(
                request_data=request.model_dump(),
                response_data={"output": [execution_result.error_response]},
            )

            return ResponsesAgentResponse(
                output=[execution_result.error_response],
                custom_outputs=execution_result.custom_outputs,
            )

        with mlflow.start_span(
            span_type=SpanType.AGENT, name=f"{execution_result.agent_type.value}_agent"
        ) as span:
            span.set_attributes(
                {
                    "agent_type": execution_result.agent_type.value,
                    "query": execution_result.query,
                    "customer_id": request.custom_inputs.get("customer")
                    if request.custom_inputs
                    else None,
                    "disable_tools": self.disable_tools,
                }
            )
            span.set_inputs(
                {
                    "request": request.model_dump(),
                    "custom_inputs": request.custom_inputs,
                }
            )

            sub_response = execution_result.sub_agent.predict(request)

            # combine custom outputs
            final_custom_outputs = execution_result.custom_outputs.copy()
            if sub_response.custom_outputs:
                final_custom_outputs.update(sub_response.custom_outputs)

            span.set_outputs(
                {
                    "response": sub_response.output,
                    "custom_outputs": final_custom_outputs,
                }
            )

            self.update_trace_preview(
                user_query=f"{execution_result.query}",
                response_data={"output": sub_response.output},
            )

        return ResponsesAgentResponse(
            output=sub_response.output, custom_outputs=final_custom_outputs
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor")
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream response from the selected sub-agent.

        Args:
            request: request containing user messages

        Yields:
            ResponsesAgentStreamEvent objects from the sub-agent
        """
        # Extract intelligence_enabled from custom_inputs
        intelligence_enabled = (
            request.custom_inputs.get("intelligence_enabled", True)
            if request.custom_inputs
            else True
        )

        if not intelligence_enabled:
            logger.info("Intelligence disabled - using direct response (streaming)")
            yield from self._handle_direct_response_stream(request)
            return

        execution_result = self._prepare_agent_execution(request)

        if execution_result.error_response:
            self.update_trace_preview(
                request_data=request.model_dump(),
                response_data={"output": [execution_result.error_response]},
            )

            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", item=execution_result.error_response
            )
            return

        yield ResponsesAgentStreamEvent(
            type="response.debug",
            item={
                "id": str(uuid4()),
                "routing_decision": f"Query routed to {execution_result.agent_type.value} agent",
            },
        )

        try:
            with mlflow.start_span(
                name=f"{execution_result.agent_type.value}_agent"
            ) as span:
                span.set_attributes(
                    {
                        "agent_type": execution_result.agent_type.value,
                        "query": execution_result.query,
                        "streaming": True,
                        "disable_tools": self.disable_tools,
                    }
                )
                span.set_inputs(
                    {
                        "request": request.model_dump(),
                        "customer_id": request.custom_inputs.get("customer")
                        if request.custom_inputs
                        else None,
                    }
                )

                # collect all events to reconstruct the final response
                all_events = []
                response_count = 0

                for event in execution_result.sub_agent.predict_stream(request):
                    response_count += 1
                    all_events.append(event)
                    yield event

                span.set_outputs({"events_streamed": response_count})

                # reconstruct complete response from all events for trace preview
                self._update_stream_trace_preview(
                    execution_result.query, all_events, execution_result.custom_outputs
                )

        except Exception as e:
            logger.error(
                f"Error processing with {execution_result.agent_type.value} agent: {str(e)}"
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": str(uuid4()),
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Error processing your request: {str(e)}",
                        }
                    ],
                    "role": "assistant",
                    "type": "message",
                },
            )

    def _update_stream_trace_preview(
        self,
        user_query: str,
        events: list[ResponsesAgentStreamEvent],
        custom_outputs: dict,
    ) -> None:
        """Update trace preview by reconstructing response from streaming events.

        Args:
            user_query: The original user query
            events: List of all streaming events
            custom_outputs: Custom outputs to include
        """
        try:
            reconstructed_output = []

            for event in events:
                if (
                    hasattr(event, "type")
                    and event.type == "response.output_item.done"
                    and hasattr(event, "item")
                    and isinstance(event.item, dict)
                ):
                    reconstructed_output.append(event.item)

            if reconstructed_output:
                self.update_trace_preview(
                    user_query=user_query,
                    response_data={
                        "output": reconstructed_output,
                        "custom_outputs": custom_outputs,
                    },
                )
            else:
                self.update_trace_preview(user_query=user_query)

        except Exception as e:
            logger.debug(f"Could not update trace preview from streaming events: {e}")
            try:
                self.update_trace_preview(user_query=user_query)
            except Exception as fallback_e:
                logger.debug(f"Fallback trace preview update also failed: {fallback_e}")

    def generate_non_response(
        self, agent_type: Union[AgentType, str], query: str
    ) -> dict:
        """Generate a graceful response when an agent type is not implemented.

        Args:
            agent_type: The agent type that isn't implemented
            query: The original user query

        Returns:
            A response item dictionary
        """
        agent_type_str = (
            agent_type.value if isinstance(agent_type, AgentType) else agent_type
        )

        return {
            "role": "assistant",
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": f"I apologize, but our {agent_type_str} support system is currently being upgraded and isn't available yet. "
                    f"We expect this feature to be available in the next few weeks. "
                    f"In the meantime, I can help with account information, profiles, and technical support. "
                    f"Would you like me to help you with any other questions?",
                }
            ],
            "id": str(uuid4()),
        }


# Create an instance and set the model
# This will be executed when the module is imported by MLflow
agent = SupervisorAgent()
set_model(agent)
