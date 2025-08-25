import { ref } from 'vue';
import type { AgentResponse, ToolCall } from '@/types/AgentResponse';
import type { ApiMessage } from '@/types/ChatMessage';

// Customer info interface
export interface CustomerInfo {
  customer_id: string;
  display_name: string;
}

// Track the current agent response being shown
export const currentAgentResponse = ref<AgentResponse | null>(null);

// Create event emitter to stream agent results
export const agentResultsEmitter = {
  listeners: new Map<string, Function>(),

  addListener(id: string, callback: Function) {
    this.listeners.set(id, callback);
  },

  removeListener(id: string) {
    this.listeners.delete(id);
  },

  emit(id: string, data: any) {
    const callback = this.listeners.get(id);
    if (callback) {
      callback(data);
    }
  }
};

// =============================================================================
// STREAMING EVENT INTERFACES
// =============================================================================

/**
 * Enhanced streaming interface to handle different event types
 */
interface StreamingEvent {
  type: 'routing' | 'tool_call' | 'tool_result' | 'response_text' | 'completion' | 'error';
  agent_type?: string;
  routing_decision?: string;
  tool_name?: string;
  call_id?: string;
  arguments?: string;
  output?: string;
  text?: string;
  tools_used?: any[];
  final_response?: string;
  error?: string;
  done?: boolean;
  trace_id?: string;
}

// =============================================================================
// HUMANIZATION HELPER FUNCTIONS
// =============================================================================

/**
 * Maps technical tool names to user-friendly names with icons
 */
const humanizeToolName = (technicalName: string): string => {
  // Handle Unity Catalog function names with double underscores
  let cleanTechnicalName = technicalName;
  if (technicalName.includes('__')) {
    // Split on __ and take the last part
    const parts = technicalName.split('__');
    cleanTechnicalName = parts[parts.length - 1];
  }

  const toolNameMap: Record<string, string> = {
    'knowledge_base_vector_search': '📚 Knowledge Base Vector Search',
    'support_tickets_vector_search': '🎫 Support Tickets Vector Search',
    'get_customer_info': '👤 Get Customer Info Tool',
    'get_customer_subscriptions': '📋 Get Customer Subscriptions Tool',
    'get_billing_info': '💳 Get Billing Info Tool',
    'get_usage_info': '📊 Get Usage Info Tools',
    'get_plans_info': '📝 Get Plans Info Tools',
    'get_devices_info': '📱 Get Devices Info Tools',
    'get_promotions_info': '🎯 Get Promotions Tool',
    'get_customer_devices': '📲 Get Customer Devices Tool',
    'python_exec': '🐍 Python Executor Tool'
  };

  // If we have a mapping for the clean name, use it
  if (toolNameMap[cleanTechnicalName]) {
    return toolNameMap[cleanTechnicalName];
  }

  // Otherwise, create a fallback by cleaning up the technical name
  const cleanName = cleanTechnicalName.replace(/_/g, ' ');
  const titleCase = cleanName.replace(/\b\w/g, letter => letter.toUpperCase());
  return `🔧 ${titleCase}`;
};

/**
 * Creates human-friendly descriptions based on tool name and context
 */
const createToolDescription = (toolName: string, toolArgs?: any): string => {
  // Handle Unity Catalog function names with double underscores
  let cleanToolName = toolName;
  if (toolName.includes('__')) {
    const parts = toolName.split('__');
    cleanToolName = parts[parts.length - 1];
  }

  const descriptions: Record<string, string> = {
    'knowledge_base_vector_search': 'Searching help articles and guides',
    'support_tickets_vector_search': 'Looking through support ticket history',
    'get_customer_info': 'Retrieving customer account details',
    'get_customer_subscriptions': 'Checking active subscriptions',
    'get_billing_info': 'Accessing billing records',
    'get_usage_info': 'Analyzing usage patterns',
    'get_plans_info': 'Comparing available plans',
    'get_devices_info': 'Looking up device specifications',
    'get_promotions_info': 'Finding current promotions',
    'get_customer_devices': 'Checking registered devices',
    'python_exec': 'Running python code'
  };

  let baseDescription = descriptions[cleanToolName];
  if (!baseDescription) {
    const cleanName = cleanToolName.replace(/_/g, ' ');
    baseDescription = `Processing ${cleanName}`;
  }

  // Special handling for python_exec - try to infer what it's doing from the code
  if (cleanToolName === 'python_exec' && toolArgs?.code) {
    const code = toolArgs.code.toLowerCase();
    if (code.includes('datetime')) {
      baseDescription = 'Calculating date information';
    } else if (code.includes('math') || code.includes('calculate')) {
      baseDescription = 'Performing calculations';
    } else {
      baseDescription = 'Running python code';
    }
  }

  // Add context from arguments if available (for other tools)
  if (toolArgs && cleanToolName !== 'python_exec') {
    if (toolArgs.query) {
      baseDescription += ` for "${toolArgs.query}"`;
    } else if (toolArgs.customer_id) {
      baseDescription += ` for this customer`;
    } else if (toolArgs.start_date && toolArgs.end_date) {
      baseDescription += ` from ${toolArgs.start_date} to ${toolArgs.end_date}`;
    }
  }

  return baseDescription;
};

/**
 * Creates natural, conversational reasoning text
 */
const createNaturalReasoning = (toolName: string, toolArgs?: any): string => {
  // Handle Unity Catalog function names with double underscores
  let cleanToolName = toolName;
  if (toolName.includes('__')) {
    const parts = toolName.split('__');
    cleanToolName = parts[parts.length - 1];
  }

  const reasoningTemplates: Record<string, string> = {
    'knowledge_base_vector_search': 'I need to search through our help documentation to find the most relevant guides and instructions',
    'support_tickets_vector_search': 'Let me check our support history to see if similar issues have been resolved before',
    'get_customer_info': 'I\'ll look up your account details to provide personalized assistance',
    'get_customer_subscriptions': 'Let me check your current subscriptions to understand your services',
    'get_billing_info': 'I\'ll retrieve your billing information to answer your payment-related question',
    'get_usage_info': 'Let me analyze your usage data to provide accurate information about your consumption',
    'get_plans_info': 'I\'ll look up our current plan offerings to help you compare options',
    'get_devices_info': 'Let me check our device catalog to provide detailed specifications',
    'get_promotions_info': 'I\'ll search for current promotions and offers that might benefit you',
    'get_customer_devices': 'Let me review the devices registered to your account',
    'python_exec': 'I need to perform some calculations to get the exact information you requested'
  };

  let reasoning = reasoningTemplates[cleanToolName];
  if (!reasoning) {
    const cleanName = cleanToolName.replace(/_/g, ' ');
    reasoning = `Let me process your request using ${cleanName}`;
  }

  // Special handling for python_exec - try to be more specific
  if (cleanToolName === 'python_exec' && toolArgs?.code) {
    const code = toolArgs.code.toLowerCase();
    if (code.includes('datetime')) {
      reasoning = 'I need to calculate date information to answer your question accurately';
    } else if (code.includes('math') || code.includes('calculate')) {
      reasoning = 'I need to perform some math calculations to answer your question accurately';
    } else {
      reasoning = 'I need to run some calculations to get the exact information you requested';
    }
  }

  // Add specific context for search queries (for other tools)
  if (toolArgs?.query && cleanToolName !== 'python_exec') {
    reasoning += ` related to "${toolArgs.query}"`;
  }

  return reasoning;
};

/**
 * Summarizes tool results in a user-friendly way
 */
const summarizeToolResults = (toolName: string, result: any, toolArgs?: any): string[] => {
  // Handle Unity Catalog function names with double underscores
  let cleanToolName = toolName;
  if (toolName.includes('__')) {
    const parts = toolName.split('__');
    cleanToolName = parts[parts.length - 1];
  }

  const summaries: string[] = [];

  // Add the humanized tool name as first item
  summaries.push(humanizeToolName(toolName));

  if (typeof result === 'string') {
    // Special handling for python_exec results
    if (cleanToolName === 'python_exec') {
      const output = result.trim();

      // Check for date range pattern (YYYY-MM-DD YYYY-MM-DD)
      const dateRangeMatch = output.match(/(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})/);
      if (dateRangeMatch) {
        const [, startDate, endDate] = dateRangeMatch;
        const startFormatted = new Date(startDate).toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric'
        });
        const endFormatted = new Date(endDate).toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric'
        });
        summaries.push(`✅ Calculated date range: ${startFormatted} - ${endFormatted}`);
      }
      // Check for single date
      else if (output.match(/^\d{4}-\d{2}-\d{2}$/)) {
        const formatted = new Date(output).toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric'
        });
        summaries.push(`✅ Calculated date: ${formatted}`);
      }
      // Check for numeric results
      else if (output.match(/^[\d.]+$/)) {
        summaries.push(`✅ Calculated result: ${output}`);
      }
      // Generic calculation result
      else {
        summaries.push(`✅ Calculation completed successfully`);
      }
    }
    // For text results from other tools
    else if (result.includes('page_content')) {
      const matches = result.match(/ID: ([\w-]+)/g);
      const count = matches ? matches.length : 1;
      summaries.push(`✅ Found ${count} relevant ${cleanToolName.includes('knowledge') ? 'help article' + (count > 1 ? 's' : '') : 'support ticket' + (count > 1 ? 's' : '')}`);
    } else if (result.length > 100) {
      summaries.push(`✅ Retrieved detailed information (${Math.ceil(result.length / 100)} sections)`);
    } else {
      summaries.push(`✅ Information retrieved successfully`);
    }
  } else if (Array.isArray(result)) {
    summaries.push(`✅ Found ${result.length} record${result.length !== 1 ? 's' : ''}`);
  } else if (typeof result === 'object' && result !== null) {
    // Handle specific data structures based on tool type
    if (cleanToolName === 'get_customer_subscriptions' && result.subscriptions) {
      const subscriptions = result.subscriptions;
      const activeCount = subscriptions.filter((sub: any) => sub.status === 'Active').length;
      summaries.push(`✅ Found ${activeCount} active subscription${activeCount !== 1 ? 's' : ''}`);

      // Add plan names if available
      const planNames = subscriptions
        .filter((sub: any) => sub.status === 'Active')
        .map((sub: any) => sub.plan?.plan_name)
        .filter(Boolean);
      if (planNames.length > 0) {
        summaries.push(`📋 Plans: ${planNames.join(', ')}`);
      }
    } else if (cleanToolName === 'get_billing_info' && result.billing_records) {
      summaries.push(`✅ Retrieved ${result.billing_records.length} billing record${result.billing_records.length !== 1 ? 's' : ''}`);
    } else if (cleanToolName === 'get_customer_info' && result.customer_id) {
      summaries.push(`✅ Retrieved account information for ${result.customer_id}`);
    } else {
      // Generic object handling
      const keys = Object.keys(result);
      if (keys.length > 0) {
        summaries.push(`✅ Retrieved ${keys.length} data field${keys.length !== 1 ? 's' : ''}`);
      }
    }
  }

  // Add query context if available (but not for python_exec)
  if (toolArgs?.query && cleanToolName !== 'python_exec') {
    summaries.push(`🔍 Search term: "${toolArgs.query}"`);
  }

  return summaries;
};

// =============================================================================
// CORE API FUNCTIONS
// =============================================================================

/**
 * Get list of demo customers from the backend
 */
export const getDemoCustomers = async (): Promise<CustomerInfo[]> => {
  try {
    const response = await fetch('/api/customers');
    if (!response.ok) {
      throw new Error('Failed to fetch customers');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching demo customers:', error);
    // Return default customers if API fails
    return [
      { customer_id: 'CUS-10001', display_name: 'Customer 10001' },
      { customer_id: 'CUS-10002', display_name: 'Customer 10002' },
      { customer_id: 'CUS-10006', display_name: 'Customer 10006' },
    ];
  }
};

/**
 * Convert API messages to conversation history format expected by backend
 */
const convertToConversationHistory = (messages: ApiMessage[]) => {
  // Convert to backend format (ApiMessage only has 'user' | 'assistant')
  return messages.map(msg => ({
    role: msg.role,
    content: msg.content
  }));
};

/**
 * Convert backend response to frontend AgentResponse format (for non-streaming)
 */
const convertBackendToAgentResponse = (databricksResponse: any): AgentResponse => {
  try {
    // Extract main response text
    const response_text = databricksResponse.response || "";
    const agent_type = databricksResponse.agent_type || null;
    const tools: ToolCall[] = [];

    // Get execution steps from the actual response structure
    const execution_steps: any[] = databricksResponse.custom_outputs?.execution_steps || [];

    // Process tools_used array if it exists
    if (databricksResponse.tools_used && Array.isArray(databricksResponse.tools_used)) {
      databricksResponse.tools_used.forEach((tool: any, index: number) => {
        const tool_name = tool.name || "unknown_function";
        const tool_args = tool.arguments || {};

        // Create humanized tool call
        const tool_call: ToolCall = {
          tool_name: humanizeToolName(tool_name),
          description: createToolDescription(tool_name, tool_args),
          reasoning: createNaturalReasoning(tool_name, tool_args),
          type: 'EXTERNAL_API',
          informations: [
            humanizeToolName(tool_name),
            createToolDescription(tool_name, tool_args)
          ]
        };

        // Look for corresponding result in execution_steps
        const tool_result_step = execution_steps.find((step: any) =>
          step.step_type === "tool_result" && step.call_id === tool.call_id
        );

        if (tool_result_step) {
          // Update tool with humanized result summary
          tool_call.informations = summarizeToolResults(
            tool_name,
            tool_result_step.result,
            tool_args
          );
        }

        tools.push(tool_call);
      });
    }

    // If no tools_used but we have execution_steps with tool_call, process those
    if (tools.length === 0 && execution_steps.length > 0) {
      const tool_call_steps = execution_steps.filter((step: any) => step.step_type === "tool_call");

      tool_call_steps.forEach((step: any, index: number) => {
        const tool_name = step.tool_name || "unknown_function";
        const tool_args = step.arguments || {};

        // Create humanized tool call
        const tool_call: ToolCall = {
          tool_name: humanizeToolName(tool_name),
          description: createToolDescription(tool_name, tool_args),
          reasoning: createNaturalReasoning(tool_name, tool_args),
          type: 'EXTERNAL_API',
          informations: [
            humanizeToolName(tool_name),
            createToolDescription(tool_name, tool_args)
          ]
        };

        // Look for corresponding result
        const tool_result_step = execution_steps.find((resultStep: any) =>
          resultStep.step_type === "tool_result" && resultStep.call_id === step.call_id
        );

        if (tool_result_step) {
          // Update tool with humanized result summary
          tool_call.informations = summarizeToolResults(
            tool_name,
            tool_result_step.result,
            tool_args
          );
        }

        tools.push(tool_call);
      });
    }

    // Build final informations in chronological order
    const final_informations: string[] = [];

    // 1. First show routing information (what happened first)
    const routing_steps = execution_steps.filter((step: any) => step.step_type === 'routing');
    routing_steps.forEach((step: any) => {
      final_informations.push(`📍 Query routed to ${agent_type?.replace('_', ' ') || 'agent'} agent`);
    });

    // 2. Then show tools used (what happened next)
    if (tools.length > 0) {
      const toolNames = tools.map(tool => {
        // Extract just the core name from humanized tool names (remove emoji and extra text)
        let coreName = tool.tool_name
          .replace(/^[^\w\s]+\s*/, '') // Remove emoji and leading non-word chars
          .replace(/\s+(Search|Lookup|Details|Information|Analytics|Calculator)$/, ''); // Simplify endings

        // Special cases for better readability
        if (coreName === 'Python') coreName = 'Python Calculator';
        if (coreName === 'Knowledge Base') coreName = 'Knowledge Base';
        if (coreName === 'Support History') coreName = 'Support History';
        if (coreName === 'Get Billing Info Tool') coreName = 'Billing Info';

        return coreName;
      });

      final_informations.push(`🔧 Used ${tools.length} tool${tools.length !== 1 ? 's' : ''} to gather information (${toolNames.join(', ')})`);
    }

    // 3. Finally show who handled it (summary)
    final_informations.push(
      agent_type ? `🤖 Response handled by ${agent_type.replace('_', ' ')} agent` : '🤖 Response handled by AI assistant'
    );

    return {
      question: '', // Will be set by caller
      tools: tools,
      final_answer: response_text || "I apologize, but I couldn't generate a proper response. Please try again.",
      final_informations: final_informations,
      non_intelligent_answer: response_text
    };

  } catch (error) {
    console.error('Error parsing agent response:', error);
    return {
      question: '',
      tools: [],
      final_answer: "I encountered an error processing your request. Please try again.",
      final_informations: ['❌ Error occurred during processing'],
      non_intelligent_answer: "I encountered an error processing your request. Please try again."
    };
  }
};

/**
 * Send a message to the telco support agent backend with streaming
 * @param messages The conversation history
 * @param messageId The ID of the message being responded to
 * @param intelligenceEnabled Whether to show the full intelligence process
 * @param customerID The customer ID for the request
 * @returns A promise that resolves when the agent response is emitted
 */
export const sendMessageToAgent = async (
  messages: ApiMessage[],
  messageId: string,
  intelligenceEnabled: boolean = true,
  customerID: string = 'CUS-10001'
): Promise<void> => {
  try {
    // Get the last user message
    const userMessages = messages.filter(msg => msg.role === 'user');
    const lastUserMessage = userMessages[userMessages.length - 1];

    if (!lastUserMessage) {
      throw new Error('No user message found');
    }

    // Convert conversation history (exclude the current message)
    const conversationHistory = convertToConversationHistory(
      messages.slice(0, -1) // Exclude the last message as it's the current one
    );

    // Build request payload for backend
    const requestPayload = {
      message: lastUserMessage.content,
      customer_id: customerID,
      conversation_history: conversationHistory,
      intelligence_enabled: intelligenceEnabled
    };

    // Emit thinking start event
    agentResultsEmitter.emit(messageId, {
      type: 'thinking-start'
    });

    // Make streaming request to backend
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
      body: JSON.stringify(requestPayload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    // Handle streaming response
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body reader available');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let agentType: string | null = null;
    let toolsUsed: ToolCall[] = [];
    let currentToolCalls = new Map<string, any>();
    let finalResponse = '';
    let traceId: string | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.trim() === '') continue;

          // Parse Server-Sent Event
          if (line.startsWith('data: ')) {
            const eventData = line.slice(6); // Remove 'data: ' prefix

            if (eventData === '[DONE]') {
              break;
            }

            try {
              const event: StreamingEvent = JSON.parse(eventData);

              // Handle different event types
              switch (event.type) {
                case 'routing':
                  agentType = event.agent_type || null;

                  if (intelligenceEnabled) {
                    agentResultsEmitter.emit(messageId, {
                      type: 'routing',
                      data: {
                        agent_type: agentType,
                        routing_decision: event.routing_decision
                      }
                    });
                  }
                  break;

                case 'tool_call':
                  if (intelligenceEnabled && event.tool_name && event.call_id) {
                    // Store tool call info for when we get the result
                    currentToolCalls.set(event.call_id, {
                      tool_name: event.tool_name,
                      arguments: event.arguments,
                      call_id: event.call_id
                    });

                    // Create tool call for frontend display
                    const toolCall: ToolCall = {
                      tool_name: humanizeToolName(event.tool_name),
                      description: createToolDescription(event.tool_name,
                        event.arguments ? JSON.parse(event.arguments) : {}),
                      reasoning: createNaturalReasoning(event.tool_name,
                        event.arguments ? JSON.parse(event.arguments) : {}),
                      type: 'EXTERNAL_API',
                      informations: [
                        humanizeToolName(event.tool_name),
                        createToolDescription(event.tool_name,
                          event.arguments ? JSON.parse(event.arguments) : {})
                      ]
                    };

                    toolsUsed.push(toolCall);

                    // Emit tool call started
                    agentResultsEmitter.emit(messageId, {
                      type: 'tool',
                      data: toolCall
                    });
                  }
                  break;

                case 'tool_result':
                  if (intelligenceEnabled && event.call_id) {
                    // Get the corresponding tool call
                    const toolCall = currentToolCalls.get(event.call_id);
                    if (toolCall) {
                      // Update tool with result information
                      const updatedInformations = summarizeToolResults(
                        toolCall.tool_name,
                        event.output,
                        toolCall.arguments ? JSON.parse(toolCall.arguments) : {}
                      );

                      // Find and update the tool in toolsUsed array
                      const toolIndex = toolsUsed.findIndex(tool =>
                        currentToolCalls.get(event.call_id!)?.tool_name === toolCall.tool_name
                      );

                      if (toolIndex !== -1) {
                        toolsUsed[toolIndex].informations = updatedInformations;
                      }

                      // Emit tool completion with results
                      agentResultsEmitter.emit(messageId, {
                        type: 'tool_result',
                        data: {
                          call_id: event.call_id,
                          informations: updatedInformations
                        }
                      });
                    }
                  }
                  break;

                case 'response_text':
                  // Accumulate response text
                  if (event.text) {
                    finalResponse = event.text;
                  }
                  break;

                case 'completion':
                  // Final completion event
                  agentType = event.agent_type || agentType;
                  finalResponse = event.final_response || finalResponse;
                  traceId = event.trace_id || traceId;

                  // Build final informations
                  const finalInformations: string[] = [];

                  if (agentType) {
                    finalInformations.push(`📍 Query routed to ${agentType.replace('_', ' ')} agent`);
                  }

                  if (toolsUsed.length > 0) {
                    const toolNames = toolsUsed.map(tool => {
                      let coreName = tool.tool_name
                        .replace(/^[^\w\s]+\s*/, '') // Remove emoji and leading non-word chars
                        .replace(/\s+(Search|Lookup|Details|Information|Analytics|Calculator)$/, '');
                      return coreName;
                    });

                    finalInformations.push(`🔧 Used ${toolsUsed.length} tool${toolsUsed.length !== 1 ? 's' : ''} to gather information (${toolNames.join(', ')})`);
                  }

                  finalInformations.push(
                    agentType ? `🤖 Response handled by ${agentType.replace('_', ' ')} agent` : '🤖 Response handled by AI assistant'
                  );

                  // Emit final answer
                  agentResultsEmitter.emit(messageId, {
                    type: 'final-answer',
                    data: {
                      final_answer: finalResponse || "I apologize, but I couldn't generate a proper response. Please try again.",
                      final_informations: finalInformations,
                      trace_id: traceId
                    }
                  });
                  return; // Exit the function

                case 'error':
                  throw new Error(event.error || 'Unknown streaming error');

                default:
                  console.log('Unknown event type:', event.type, event);
              }

            } catch (parseError) {
              console.error('Error parsing streaming event:', parseError, 'Raw data:', eventData);
            }
          }
        }
      }

    } finally {
      reader.releaseLock();
    }

  } catch (error) {
    console.error('Error in sendMessageToAgent:', error);

    // Emit error response
    agentResultsEmitter.emit(messageId, {
      type: 'final-answer',
      data: {
        final_answer: 'I apologize, but I encountered an error processing your request. Please try again.',
        final_informations: ['❌ Error occurred during processing']
      }
    });

    throw error;
  }
};

/**
 * Send a message to the telco support agent backend (non-streaming fallback)
 * @param messages The conversation history
 * @param messageId The ID of the message being responded to
 * @param intelligenceEnabled Whether to show the full intelligence process
 * @param customerID The customer ID for the request
 * @returns A promise that resolves when the agent response is emitted
 */
export const sendMessageToAgentNonStreaming = async (
  messages: ApiMessage[],
  messageId: string,
  intelligenceEnabled: boolean = true,
  customerID: string = 'CUS-10001'
): Promise<void> => {
  try {
    // Get the last user message
    const userMessages = messages.filter(msg => msg.role === 'user');
    const lastUserMessage = userMessages[userMessages.length - 1];

    if (!lastUserMessage) {
      throw new Error('No user message found');
    }

    // Convert conversation history (exclude the current message)
    const conversationHistory = convertToConversationHistory(
      messages.slice(0, -1) // Exclude the last message as it's the current one
    );

    // Build request payload for backend
    const requestPayload = {
      message: lastUserMessage.content,
      customer_id: customerID,
      conversation_history: conversationHistory,
      intelligence_enabled: intelligenceEnabled
    };

    // Emit thinking start event
    agentResultsEmitter.emit(messageId, {
      type: 'thinking-start'
    });

    // Make request to backend
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestPayload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const backendResponse = await response.json();

    // Convert backend response to frontend format
    const agentResponse = convertBackendToAgentResponse(backendResponse);
    agentResponse.question = lastUserMessage.content;

    // Extract trace_id from backend response
    const traceId = backendResponse.trace_id;

    // If intelligence is enabled, emit tool events
    if (intelligenceEnabled && agentResponse.tools.length > 0) {
      // Emit tools one by one with slight delays for animation
      for (let i = 0; i < agentResponse.tools.length; i++) {
        // Add delay between tools for smoother animation
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }

        agentResultsEmitter.emit(messageId, {
          type: 'tool',
          data: agentResponse.tools[i]
        });
      }

      // Small delay before final answer
      await new Promise(resolve => setTimeout(resolve, 800));
    }

    // Emit final answer
    agentResultsEmitter.emit(messageId, {
      type: 'final-answer',
      data: {
        final_answer: agentResponse.final_answer,
        final_informations: agentResponse.final_informations,
        trace_id: traceId
      }
    });

  } catch (error) {
    console.error('Error in sendMessageToAgentNonStreaming:', error);

    // Emit error response
    agentResultsEmitter.emit(messageId, {
      type: 'final-answer',
      data: {
        final_answer: 'I apologize, but I encountered an error processing your request. Please try again.',
        final_informations: ['❌ Error occurred during processing']
      }
    });

    throw error;
  }
};

/**
 * Submit customer service agent feedback for a specific trace
 */
export const submitFeedback = async (
  traceId: string,
  isPositive: boolean,
  comment: string | null,
  agentId: string
): Promise<{ success: boolean; experimentUrl?: string }> => {
  try {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        trace_id: traceId,
        is_positive: isPositive,
        comment: comment,
        agent_id: agentId,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('Feedback submitted successfully:', result);
    return {
      success: true,
      experimentUrl: result.experiment_url
    };
  } catch (error) {
    console.error('Error submitting feedback:', error);
    return { success: false };
  }
};

/**
 * Get predefined questions for telco use case
 * These are hardcoded since we're telco-specific now
 */
export const getPredefinedQuestions = async () => {
  // Return telco-specific questions
  return [
    {
      preview: "Current plan",
      text: "What plan is the customer currently on?"
    },
    {
      preview: "Active subscriptions",
      text: "Show all active subscriptions and their status for the customer"
    },
    {
      preview: "Customer segment",
      text: "What is the customer's segment and loyalty tier?"
    },
    {
      preview: "April charges",
      text: "What are the charges on the customer's bill from 2025-04-01 to 2025-04-30?"
    },
    {
      preview: "Payment due date",
      text: "When will the customer's current payment be due?"
    },
    {
      preview: "Bill breakdown",
      text: "Break down the total amount of the customer's latest billing statement"
    },
    {
      preview: "International roaming",
      text: "How do I set up international roaming for my upcoming trip?"
    },
    {
      preview: "Dropped calls",
      text: "My iPhone keeps dropping calls during conversations"
    },
    {
      preview: "Plan comparison",
      text: "What's the difference between the Standard and Premium plans?"
    },
    {
      preview: "iPhone promotions",
      text: "Are there any active promotions for iPhone upgrades?"
    },
    {
      preview: "SF outage check",
      text: "Is there a network outage in San Francisco?"
    },
    {
      preview: "LA network quality",
      text: "What's the network performance in Downtown LA?"
    }
  ];
};