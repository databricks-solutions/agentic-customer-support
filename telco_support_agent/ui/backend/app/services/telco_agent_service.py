"""Service for interacting with the Databricks Telco Support Agent."""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ..config import Settings

logger = logging.getLogger(__name__)


def decode_unicode_escapes(text: str) -> str:
    r"""Decode Unicode escape sequences in text (e.g., \\u2019 -> ')."""
    try:
        # Use codecs to decode Unicode escapes
        return text.encode().decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If decoding fails, return original text
        return text


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str
    content: str


class AgentResponse(BaseModel):
    """Agent response model."""

    response: str
    agent_type: Optional[str] = None
    custom_outputs: Optional[dict] = None
    tools_used: Optional[list[dict]] = None
    trace_id: Optional[str] = None


class TelcoAgentService:
    """Service for interacting with Databricks Telco Support Agent."""

    def __init__(self, settings: Settings):
        """Initialize the service."""
        self.settings = settings

        if not settings.has_auth:
            logger.warning(
                "No Databricks authentication configured. "
                "Set DATABRICKS_TOKEN for local development or ensure app has OAuth credentials."
            )
            self.client = None
            self.access_token = None
        else:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.request_timeout),
            )
            self.access_token = None
            logger.info(
                f"Initialized agent service with {settings.auth_method} authentication"
            )

    async def _get_oauth_token(self) -> str:
        """Get OAuth access token using client credentials flow."""
        if (
            not self.settings.databricks_client_id
            or not self.settings.databricks_client_secret
        ):
            raise ValueError("OAuth credentials not available")

        workspace_host = self.settings.databricks_host
        token_url = f"{workspace_host}/oidc/v1/token"

        logger.info(f"Getting OAuth token from: {token_url}")

        data = {"grant_type": "client_credentials", "scope": "all-apis"}

        auth = (
            self.settings.databricks_client_id,
            self.settings.databricks_client_secret,
        )

        response = await self.client.post(
            token_url,
            data=data,
            auth=auth,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            logger.error(
                f"OAuth token request failed: {response.status_code} - {response.text}"
            )
            raise ValueError(f"Failed to get OAuth token: {response.status_code}")

        token_data = response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            logger.error(f"No access token in response: {token_data}")
            raise ValueError("No access token received from OAuth endpoint")

        logger.info("Successfully obtained OAuth access token")
        return access_token

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with proper authentication."""
        headers = {"Content-Type": "application/json"}

        if self.settings.auth_method == "oauth":
            if not self.access_token:
                try:
                    self.access_token = await self._get_oauth_token()
                except Exception as e:
                    logger.error(f"Failed to get OAuth token: {e}")
                    logger.info("Falling back to direct client secret usage")
                    headers["Authorization"] = (
                        f"Bearer {self.settings.databricks_client_secret}"
                    )
                    return headers

            headers["Authorization"] = f"Bearer {self.access_token}"

        elif self.settings.auth_method == "token":
            headers["Authorization"] = f"Bearer {self.settings.databricks_token}"

        return headers

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    def _build_databricks_payload(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage],
        stream: bool = False,
        intelligence_enabled: bool = True,
    ) -> dict[str, Any]:
        """Build the payload for Databricks API."""
        input_messages = []
        for msg in conversation_history:
            input_messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        input_messages.append({"role": "user", "content": message})

        payload = {
            "input": input_messages,
            "custom_inputs": {
                "customer": customer_id,
                "intelligence_enabled": intelligence_enabled,
            },
            "databricks_options": {"return_trace": True},
        }

        if stream:
            payload["stream"] = True

        # Debug logging
        logger.error(
            f"DEBUG: TelcoAgentService payload - intelligence_enabled: {intelligence_enabled}"
        )
        logger.error(
            f"DEBUG: TelcoAgentService payload - custom_inputs: {payload['custom_inputs']}"
        )

        return payload

    def _parse_agent_response(
        self, databricks_response: dict[str, Any]
    ) -> AgentResponse:
        """Parse the response from Databricks into our format."""
        try:
            # extract main response text
            response_text = ""
            agent_type = None
            tools_used = []
            execution_steps = []
            trace_id = None

            # Extract trace_id from the response
            databricks_output = databricks_response.get("databricks_output", {})
            if isinstance(databricks_output, dict):
                trace_info = databricks_output.get("trace", {})
                if isinstance(trace_info, dict):
                    info = trace_info.get("info", {})
                    if isinstance(info, dict):
                        trace_id = info.get("trace_id")

                if not trace_id and isinstance(databricks_output, dict):
                    trace_id = databricks_output.get(
                        "trace_id"
                    ) or databricks_output.get("request_id")

            if not trace_id:
                trace_id = databricks_response.get("trace_id")

            # parse output array to get execution details
            output = databricks_response.get("output", [])

            for item in output:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    # extract text content from message
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            response_text += content_item.get("text", "")

                elif item.get("type") == "function_call":
                    # capture function call details
                    function_name = item.get("name", "unknown_function")
                    function_args = item.get("arguments", "{}")
                    call_id = item.get("call_id", "")

                    try:
                        if isinstance(function_args, str):
                            parsed_args = json.loads(function_args)
                        else:
                            parsed_args = function_args
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"raw_arguments": function_args}

                    tool_call = {
                        "name": function_name,
                        "arguments": parsed_args,
                        "call_id": call_id,
                        "type": "function_call",
                    }

                    tools_used.append(tool_call)

                    # add to execution steps for detailed view
                    execution_steps.append(
                        {
                            "step_type": "tool_call",
                            "tool_name": function_name,
                            "description": f"Calling {function_name}",
                            "arguments": parsed_args,
                            "reasoning": f"Using {function_name} to retrieve relevant information",
                        }
                    )

                elif item.get("type") == "function_call_output":
                    # capture function results
                    call_id = item.get("call_id", "")
                    output_data = item.get("output", "")

                    try:
                        if isinstance(output_data, str) and output_data.startswith("{"):
                            parsed_output = json.loads(output_data)
                        else:
                            parsed_output = output_data
                    except (json.JSONDecodeError, TypeError):
                        parsed_output = output_data

                    # add result to execution steps
                    execution_steps.append(
                        {
                            "step_type": "tool_result",
                            "call_id": call_id,
                            "description": "Tool execution completed",
                            "result": parsed_output,
                            "reasoning": "Retrieved relevant information to answer the query",
                        }
                    )

            # extract custom outputs for agent routing info
            custom_outputs = databricks_response.get("custom_outputs", {})
            routing_info = custom_outputs.get("routing", {})
            if routing_info:
                agent_type = routing_info.get("agent_type")

            # add routing to execution steps
            if agent_type:
                execution_steps.insert(
                    0,
                    {
                        "step_type": "routing",
                        "description": f"Query routed to {agent_type} agent",
                        "reasoning": f"This query is best handled by the {agent_type} specialist",
                    },
                )

            return AgentResponse(
                response=response_text
                or "I apologize, but I couldn't generate a proper response. Please try again.",
                agent_type=agent_type,
                custom_outputs={**custom_outputs, "execution_steps": execution_steps},
                tools_used=tools_used,
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            return AgentResponse(
                response="I encountered an error processing your request. Please try again.",
                agent_type=None,
                custom_outputs=None,
                tools_used=None,
                trace_id=None,
            )

    def _parse_sse_line(self, line: str) -> Optional[dict]:
        """Parse a single Server-Sent Event line."""
        line = line.strip()

        # skip empty lines and comments
        if not line or line.startswith(":"):
            return None

        # handle data lines
        if line.startswith("data: "):
            data_content = line[6:]  # Remove 'data: ' prefix

            if data_content == "[DONE]":
                return {"type": "done"}

            try:
                return json.loads(data_content)
            except json.JSONDecodeError as e:
                # For large chunks that likely contain trace data, try harder to parse
                if "databricks_output" in data_content and "trace" in data_content:
                    logger.debug(
                        f"Found large SSE chunk with databricks_output: {len(data_content)} chars"
                    )

                    # This is likely the final event with complete response
                    # Extract just the content we need
                    if (
                        '"type":"response.output_item.done"' in data_content
                        and '"role":"assistant"' in data_content
                    ):
                        # Try to extract the item object which contains the full response
                        item_match = re.search(
                            r'"item"\s*:\s*({[^}]*"role"\s*:\s*"assistant"[^}]*})',
                            data_content,
                        )
                        if item_match:
                            try:
                                item_str = item_match.group(1)
                                content_start = item_str.find('"content":')
                                if content_start != -1:
                                    # Extract content array carefully
                                    content_start = item_str.find("[", content_start)
                                    if content_start != -1:
                                        bracket_count = 0
                                        content_end = content_start
                                        for i in range(content_start, len(item_str)):
                                            if item_str[i] == "[":
                                                bracket_count += 1
                                            elif item_str[i] == "]":
                                                bracket_count -= 1
                                                if bracket_count == 0:
                                                    content_end = i + 1
                                                    break

                                        content_str = item_str[
                                            content_start:content_end
                                        ]
                                        # Now parse the content array
                                        content_array = json.loads(content_str)

                                        for content_item in content_array:
                                            if (
                                                content_item.get("type")
                                                == "output_text"
                                            ):
                                                text = content_item.get("text", "")
                                                if text:
                                                    logger.info(
                                                        f"Extracted full text from databricks_output event: {len(text)} chars"
                                                    )
                                                    return {
                                                        "type": "response.output_item.done",
                                                        "item": {
                                                            "type": "message",
                                                            "role": "assistant",
                                                            "content": [
                                                                {
                                                                    "type": "output_text",
                                                                    "text": text,
                                                                }
                                                            ],
                                                        },
                                                    }
                            except Exception as extract_error:
                                logger.debug(
                                    f"Failed to extract from item match: {extract_error}"
                                )

                    # Fallback: Try to extract any text field in the response
                    # Look for the longest text field (likely the complete response)
                    all_text_matches = re.findall(
                        r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', data_content
                    )
                    if all_text_matches:
                        longest_text = ""
                        for raw_text in all_text_matches:
                            try:
                                decoded_text = json.loads('"' + raw_text + '"')
                                if len(decoded_text) > len(longest_text):
                                    longest_text = decoded_text
                            except json.JSONDecodeError:
                                # skip malformed JSON strings
                                continue

                        if longest_text and len(longest_text) > 100:  # Sanity check
                            logger.info(
                                f"Extracted longest text from unparseable chunk: {len(longest_text)} chars"
                            )
                            return {
                                "type": "response.output_item.done",
                                "item": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": longest_text}
                                    ],
                                },
                            }

                    # If we can't extract anything useful, skip silently
                    return None
                else:
                    if len(data_content) > 100:
                        logger.warning(
                            f"Failed to parse SSE data: {data_content[:100]}..., error: {e}"
                        )
                    return None

        return None

    async def send_message(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage] = None,
        intelligence_enabled: bool = True,
    ) -> AgentResponse:
        """Send a message to the telco support agent."""
        if conversation_history is None:
            conversation_history = []

        if not self.client:
            return AgentResponse(
                response=(
                    "Chat functionality is currently unavailable. "
                    "Please configure Databricks authentication to enable agent responses. "
                    "The demo customer list and UI are still available for testing."
                ),
                agent_type="error",
                custom_outputs={"error": "no_authentication"},
                tools_used=None,
                trace_id=None,
            )

        try:
            payload = self._build_databricks_payload(
                message,
                customer_id,
                conversation_history,
                stream=False,
                intelligence_enabled=intelligence_enabled,
            )

            logger.info(f"Sending request to Databricks for customer {customer_id}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            headers = await self._get_headers()

            response = await self.client.post(
                self.settings.databricks_endpoint, json=payload, headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Databricks response: {json.dumps(response_data, indent=2)}")

            return self._parse_agent_response(response_data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Databricks: {e.response.status_code} - {e.response.text}"
            )

            if e.response.status_code == 403 and self.settings.auth_method == "oauth":
                logger.info("Got 403, attempting to refresh OAuth token and retry...")
                try:
                    self.access_token = await self._get_oauth_token()
                    headers = await self._get_headers()

                    response = await self.client.post(
                        self.settings.databricks_endpoint, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    return self._parse_agent_response(response_data)

                except Exception as retry_error:
                    logger.error(f"Retry after token refresh failed: {retry_error}")

            return AgentResponse(
                response=f"Service temporarily unavailable (HTTP {e.response.status_code}). Please try again.",
                agent_type="error",
                custom_outputs={
                    "error": "http_error",
                    "status_code": e.response.status_code,
                },
                tools_used=None,
                trace_id=None,
            )

        except httpx.RequestError as e:
            logger.error(f"Request error to Databricks: {e}")
            return AgentResponse(
                response="Unable to connect to agent service. Please check your connection and try again.",
                agent_type="error",
                custom_outputs={"error": "connection_error"},
                tools_used=None,
                trace_id=None,
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return AgentResponse(
                response=f"An unexpected error occurred: {str(e)}",
                agent_type="error",
                custom_outputs={"error": "unexpected_error"},
                tools_used=None,
                trace_id=None,
            )

    async def send_message_stream(
        self,
        message: str,
        customer_id: str,
        conversation_history: list[ChatMessage] = None,
        intelligence_enabled: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Send a message with real streaming response."""
        if conversation_history is None:
            conversation_history = []

        if not self.client:
            error_response = {
                "type": "error",
                "error": "Chat functionality is currently unavailable. Please configure Databricks authentication.",
                "done": True,
            }
            yield f"data: {json.dumps(error_response)}\n\n"
            return

        try:
            payload = self._build_databricks_payload(
                message,
                customer_id,
                conversation_history,
                stream=True,
                intelligence_enabled=intelligence_enabled,
            )

            logger.info(f"Starting streaming request for customer {customer_id}")
            logger.error(
                f"DEBUG: Request endpoint: {self.settings.databricks_endpoint}"
            )
            logger.error(f"DEBUG: Request payload: {payload}")

            headers = await self._get_headers()
            logger.error(f"DEBUG: Request headers: {headers}")

            # make streaming request
            async with self.client.stream(
                "POST", self.settings.databricks_endpoint, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.reason_phrase}"
                    logger.error(f"Streaming request failed: {error_msg}")

                    error_response = {
                        "type": "error",
                        "error": f"Service temporarily unavailable ({error_msg})",
                        "done": True,
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return

                collected_tools = []
                current_response_text = ""
                agent_type = None
                routing_info = None
                trace_id = None

                # Note: trace_id is not available in streaming responses
                # from Databricks - it's only available in the final non-streaming response
                # that includes databricks_output.trace.info.trace_id

                async for chunk in response.aiter_text():
                    # Check for trace_id in raw chunks and extract it
                    if "trace_id" in chunk and not trace_id:
                        # Look for trace_id in the format: "trace_id": "tr-xxxxx"
                        trace_id_match = re.search(r'"trace_id":\s*"(tr-[^"]+)"', chunk)
                        if trace_id_match:
                            extracted_trace_id = trace_id_match.group(1)
                            logger.info(
                                f"Extracted trace_id from raw chunk: {extracted_trace_id}"
                            )
                            trace_id = extracted_trace_id

                    # Skip text extraction from raw chunks - rely on parsed SSE events instead
                    # This avoids issues with regex-based extraction truncating text

                    lines = chunk.split("\n")

                    for line in lines:
                        event_data = self._parse_sse_line(line)

                        if not event_data:
                            continue

                        event_type = event_data.get("type")
                        logger.debug(f"Processing streaming event: {event_type}")

                        if event_type == "response.debug":
                            # Routing decision
                            item = event_data.get("item", {})
                            routing_decision = item.get("routing_decision", "")

                            if "routed to" in routing_decision.lower():
                                # Extract agent type from routing decision
                                if "account agent" in routing_decision.lower():
                                    agent_type = "account"
                                elif "billing agent" in routing_decision.lower():
                                    agent_type = "billing"
                                elif "tech support agent" in routing_decision.lower():
                                    agent_type = "tech_support"
                                elif "product agent" in routing_decision.lower():
                                    agent_type = "product"

                                routing_info = routing_decision

                                # Send routing event to frontend
                                routing_event = {
                                    "type": "routing",
                                    "agent_type": agent_type,
                                    "routing_decision": routing_decision,
                                }
                                yield f"data: {json.dumps(routing_event)}\n\n"

                        elif event_type == "response.output_item.done":
                            item = event_data.get("item", {})
                            item_type = item.get("type")
                            logger.debug(
                                f"Processing output_item.done with item_type: {item_type}"
                            )

                            if item_type == "function_call":
                                tool_info = {
                                    "type": "tool_call",
                                    "tool_name": item.get("name", "unknown"),
                                    "call_id": item.get("call_id"),
                                    "arguments": item.get("arguments", "{}"),
                                }
                                collected_tools.append(tool_info)
                                yield f"data: {json.dumps(tool_info)}\n\n"

                            elif item_type == "function_call_output":
                                call_id = item.get("call_id")
                                output = item.get("output", "")

                                for tool in collected_tools:
                                    if tool.get("call_id") == call_id:
                                        tool["output"] = output
                                        break

                                tool_result = {
                                    "type": "tool_result",
                                    "call_id": call_id,
                                    "output": output,
                                }
                                yield f"data: {json.dumps(tool_result)}\n\n"

                            elif (
                                item_type == "message"
                                and item.get("role") == "assistant"
                            ):
                                content = item.get("content", [])
                                for content_item in content:
                                    if content_item.get("type") == "output_text":
                                        # Update with the text from the message event
                                        extracted_text = content_item.get("text", "")
                                        if extracted_text:
                                            # Always use the complete text from the message event
                                            current_response_text = extracted_text
                                            logger.info(
                                                f"Set response text from message event: {len(extracted_text)} chars"
                                            )

                                            # Log first 200 chars for debugging
                                            logger.debug(
                                                f"Response text preview: {extracted_text[:200]}..."
                                            )

                                            # Emit the complete response text
                                            response_event = {
                                                "type": "response_text",
                                                "text": current_response_text,
                                            }
                                            yield f"data: {json.dumps(response_event)}\n\n"

                                # Extract trace_id from databricks_output if present in the item
                                databricks_output = item.get("databricks_output", {})
                                if isinstance(databricks_output, dict):
                                    trace_info = databricks_output.get("trace", {})
                                    if isinstance(trace_info, dict):
                                        info = trace_info.get("info", {})
                                        if isinstance(info, dict):
                                            extracted_trace_id = info.get("trace_id")
                                            if extracted_trace_id:
                                                trace_id = extracted_trace_id

                        elif event_type == "done":
                            # stream completed
                            break

                # Log final response length for debugging
                logger.info(
                    f"Sending completion event with response text of {len(current_response_text)} chars"
                )
                if len(current_response_text) < 500:
                    logger.warning(
                        f"Response text seems truncated: {current_response_text}"
                    )

                completion_event = {
                    "type": "completion",
                    "agent_type": agent_type,
                    "routing_decision": routing_info,
                    "tools_used": collected_tools,
                    "final_response": current_response_text,
                    "trace_id": trace_id,
                    "done": True,
                }
                yield f"data: {json.dumps(completion_event)}\n\n"

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Databricks: {e.response.status_code} - {e.response.text}"
            )

            # try to refresh token if 403 and using OAuth
            if e.response.status_code == 403 and self.settings.auth_method == "oauth":
                logger.info("Got 403, attempting to refresh OAuth token and retry...")
                try:
                    self.access_token = await self._get_oauth_token()
                    # retry the streaming request
                    async for event in self.send_message_stream(
                        message, customer_id, conversation_history
                    ):
                        yield event
                    return
                except Exception as retry_error:
                    logger.error(f"Retry after token refresh failed: {retry_error}")

            error_response = {
                "type": "error",
                "error": f"Service temporarily unavailable (HTTP {e.response.status_code})",
                "done": True,
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        except httpx.RequestError as e:
            logger.error(f"Request error to Databricks: {e}")

            error_response = {
                "type": "error",
                "error": "Unable to connect to agent service. Please check your connection and try again.",
                "done": True,
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")

            error_response = {
                "type": "error",
                "error": f"An unexpected error occurred: {str(e)}",
                "done": True,
            }
            yield f"data: {json.dumps(error_response)}\n\n"

    async def health_check(self) -> bool:
        """Check if the Databricks endpoint is healthy."""
        if not self.client:
            return False

        try:
            test_payload = self._build_databricks_payload("Hello", "CUS-10001", [])
            headers = await self._get_headers()

            response = await self.client.post(
                self.settings.databricks_endpoint, json=test_payload, headers=headers
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
