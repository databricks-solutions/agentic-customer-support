# Databricks notebook source
# MAGIC %md
# MAGIC # Test Multi-Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

if root_path:
    sys.path.append(root_path)

# COMMAND ----------

import itertools

from mlflow.types.responses import ResponsesAgentRequest

from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.tools import initialize_tools
from telco_support_agent.agents.agent_types import AgentType
from telco_support_agent.config import UCConfig, AgentConfig
from telco_support_agent.tools.registry import DOMAIN_FUNCTION_MAP

# COMMAND ----------

print("Available agent types:")
for agent_type in AgentType:
    print(f"  - {agent_type.name}: {agent_type.value}")

# COMMAND ----------

# UC config for testing - only need data reading capabilities
uc_config = UCConfig(
    # Uses defaults: data_catalog="telco_customer_support_prod", data_schema="gold"
)

print("Unity Catalog Configuration:")
print(f"  Data Catalog: {uc_config.data_catalog}")
print(f"  Data Schema: {uc_config.data_schema}")

print("\nAvailable agent types and their functions:")
for agent_type, functions in DOMAIN_FUNCTION_MAP.items():
    print(f"  {agent_type.upper()}: {functions}")

# agent configs from YAML files
try:
    supervisor_config = AgentConfig.load_from_file("supervisor", uc_config)
    print(f"\nSupervisor LLM endpoint: {supervisor_config.llm.endpoint}")
except Exception as e:
    print(f"\nCould not load supervisor config: {e}")

try:
    account_config = AgentConfig.load_from_file("account", uc_config)
    print(f"\nAccount agent LLM endpoint: {account_config.llm.endpoint}")
    print(f"Account agent functions: {account_config.uc_functions}")
except Exception as e:
    print(f"\nCould not load account config: {e}")

try:
    billing_config = AgentConfig.load_from_file("billing", uc_config)
    print(f"\nBilling agent LLM endpoint: {billing_config.llm.endpoint}")
    print(f"Billing agent functions: {billing_config.uc_functions}")
except Exception as e:
    print(f"\nCould not load billing config: {e}")

try:
    tech_support_config = AgentConfig.load_from_file("tech_support", uc_config)
    print(f"\nTech support agent LLM endpoint: {tech_support_config.llm.endpoint}")
except Exception as e:
    print(f"\nCould not load tech_support config: {e}")

try:
    product_config = AgentConfig.load_from_file("product", uc_config)
    print(f"\nProduct agent LLM endpoint: {product_config.llm.endpoint}")
    print(f"Product agent functions: {product_config.uc_functions}")
except Exception as e:
    print(f"\nCould not load product config: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Supervisor Agent

# COMMAND ----------

supervisor = SupervisorAgent(uc_config=uc_config)

print(f"Agent type: {supervisor.agent_type}")
print(f"LLM endpoint: {supervisor.llm_endpoint}")
print(f"LLM parameters: {supervisor.llm_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Required UC Functions

# COMMAND ----------

print("Initializing UC functions...")

results = initialize_tools(uc_config=uc_config)

success_count = 0
total_count = 0
for domain, functions in results.items():
    for func_name, status in functions.items():
        total_count += 1
        if status:
            success_count += 1
        status_str = "PASS" if status else "FAIL"
        print(f"  {status_str} {func_name}")

print(f"\nInitialized {success_count}/{total_count} UC functions")

if any(not all(functions.values()) for functions in results.values()):
    print("\nWARNING: Some functions could not be initialized")
    print("Tests might fail without the necessary UC functions")
else:
    print("\nAll required functions are available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Routing Logic

# COMMAND ----------

def test_routing(query):
    agent_type = supervisor.route_query(query)
    print(f"Query: '{query}'")
    print(f"Routed to: {agent_type.name} agent ({agent_type.value})\n")
    return agent_type

# test routing with different query types
print("=== ROUTING TESTS ===\n")

routing_test_queries = [
    # Account queries
    "What plan am I currently on?",
    "When did I create my account?",

    # Billing queries
    "Why is my bill higher this month?",
    "When is my payment due?",
    "I see a charge I don't recognize",

    # Tech support queries
    "My phone won't connect to the network",
    "I can't make calls but data works",
    "How do I reset my voicemail password?",

    # Product queries
    "What's the difference between the Standard and Premium plans?",
    "Do you have any promotions for existing customers?",
    "Is my phone 5G compatible?",
]
routing_expected_results = [
    "account",
    "account",
    "billing",
    "billing",
    "billing",
    "tech_support",
    "tech_support",
    "tech_support",
    "product",
    "product",
    "product",
]
routing_results = {}
for query in routing_test_queries:
    routing_results[query] = test_routing(query)

for query, expected_result in zip(routing_test_queries, routing_expected_results):
    print(f"Query: '{query}'")
    print(f"Expected: {expected_result}")
    print(f"Actual: {routing_results[query].value}")
    assert routing_results[query].value == expected_result, f"Expected {expected_result} but got {routing_results[query].value}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test End-to-End Queries

# COMMAND ----------

def format_response(response):
    """Format and print a response for better readability."""
    print("\n=== RESPONSE ===")

    for output_item in response.output:
        if hasattr(output_item, "type") and output_item.type == "message":
            if hasattr(output_item, "content"):
                for content_item in output_item.content:
                    if hasattr(content_item, "type") and content_item.type == "output_text":
                        print("\n" + content_item.text)

        elif hasattr(output_item, "type") and output_item.type == "function_call_output":
            print("\nFunction Output:", output_item.output)

        elif hasattr(output_item, "type") and output_item.type == "function_call":
            print(f"\nFunction Call: {output_item.name}")
            print(f"Arguments: {output_item.arguments}")

    if response.custom_outputs:
        print("\n=== CUSTOM OUTPUTS ===")
        for key, value in response.custom_outputs.items():
            print(f"{key}: {value}")

    print("\n" + "="*50)

def test_end_to_end_query(query, custom_inputs, description=""):
    """Test an end-to-end query with the supervisor agent."""
    print(f"\n{'='*80}")
    print(f"END-TO-END TEST: {description}")
    print(f"Query: '{query}'")
    print('='*80)

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}],
        custom_inputs=custom_inputs
    )

    response = supervisor.predict(request)
    format_response(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Customer IDs

# COMMAND ----------

# mixed set of customer IDs for testing
test_customers = [
    "CUS-10001",  # Family  customer
    "CUS-10002",  # Individual customer
    "CUS-10006",  # Student customer
    "CUS-10023",  # Premium customer
    "CUS-10048",  # Business customer
    "CUS-10172",  # Platium customer
    "CUS-11081",  # New customer
    "CUS-10619",  # Long-term customer
]

customer_cycle = itertools.cycle(test_customers)

def get_next_customer():
    return next(customer_cycle)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Account Queries

# COMMAND ----------

account_quality_queries = [
    ("What plan am I currently on?", "Current Plan Query"),
    ("Is my autopay enabled in my subscriptions?", "Autopay Status Query"),
    ("What plan am I currently on and when does my contract expire?", "Plan and Contract Query"),
    ("Show me all my active subscriptions and their status", "All Subscriptions Query"),
    ("What's my customer segment and loyalty tier?", "Customer Profile Query"),
    ("How long have I been a customer and what's my account status?", "Customer History Query"),
]

for query, description in account_quality_queries:
    customer_id = get_next_customer()
    custom_inputs = {"customer": customer_id}
    print(f"\n[Using Customer ID: {customer_id}]")
    test_end_to_end_query(query, custom_inputs, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Billing Queries

# COMMAND ----------

billing_quality_queries = [
    ("What are the charges on my bill from 2025-04-01 to 2025-04-30?", "CUS-10001", "Date Range Billing"),
    ("When is my June payment due", "CUS-10001", "Payment Due Date"),
    ("I see an additional charge for $39 in my May bill that I don't recognize. show me my bill", "CUS-11094", "Unrecognized Charge"), ## validated customer ID and additional charge
    ("How much data did I use from 2025-04-01 to 2025-04-30?", "CUS-10001", "Current Usage Query"),
    ("Is there an unpaid amount from 2025-04-01 to 2025-04-30?", "CUS-10001", "Payment Status"),
    ("Show me my billing history for the last 3 months", "CUS-10002", "Historical Billing Query"),
    ("When will my current payment be due", "CUS-10002", "Payment Due Date"), ## test query for the current month
    ("Break down the total amount of my latest billing Statement", "CUS-10002", "Bill breakdown"),  ## test temporal interpretation of queries
    ("How much SMS did I use in the last 3 months", "CUS-10002", "Historical Usage Query"),  ## test temporal interpretation of queries
]
for query, customer_id, description in billing_quality_queries:
    custom_inputs = {"customer": customer_id}
    print(f"\n[Using Customer ID: {customer_id}]")
    test_end_to_end_query(query, custom_inputs, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Tech Support Queries

# COMMAND ----------

tech_support_quality_queries = [
    ("My iPhone keeps dropping calls during conversations", "Call Dropping Issue"),
    ("I can't connect to wifi and my data isn't working either", "Connectivity Issue"),
    ("How do I set up international roaming for my upcoming trip?", "Roaming Setup"),
    ("My voicemail notifications aren't working properly", "Voicemail Issue"),
    ("I'm getting poor signal strength at home, what can I do?", "Signal Strength Issue"),
]

for query, description in tech_support_quality_queries:
    test_end_to_end_query(query, {}, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Product Queries

# COMMAND ----------

product_quality_queries = [
    ("What's the difference between the Standard and Premium plans?", "Plan Comparison"),
    ("Compare the features and pricing of all available plans", "All Plans Comparison"),
    ("What Samsung devices are currently available and their specifications?", "Samsung Devices Query"),
    ("Are there any active promotions for iPhone upgrades?", "iPhone Promotions"),
    ("What devices do I currently have on my account?", "Customer Devices Query"),
    ("Which plans are compatible with 5G devices and what's the price difference?", "5G Plans Query"),
]

for query, description in product_quality_queries:
    customer_id = get_next_customer()
    custom_inputs = {"customer": customer_id}
    print(f"\n[Using Customer ID: {customer_id}]")
    test_end_to_end_query(query, custom_inputs, description)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Streaming Responses

# COMMAND ----------

def display_streaming_response(model_input, description=""):
    """Display a streaming response as it comes in."""
    print(f"\n{'='*80}")
    print(f"STREAMING TEST: {description}")
    print('='*80)

    full_text = ""
    function_calls = []
    function_outputs = []

    for i, event in enumerate(supervisor.predict_stream(model_input)):
        if event.type == "response.output_item.done":
            item = event.item

            if "type" in item and item["type"] == "message":
                if "content" in item:
                    for content_item in item["content"]:
                        if "type" in content_item and content_item["type"] == "output_text":
                            text = content_item["text"]
                            print(f"\n[Message Chunk {i}]")
                            print(text)
                            full_text += text

            elif "type" in item and item["type"] == "function_call":
                print(f"\n[Function Call {i}]: {item['name']}")
                print(f"Arguments: {item['arguments']}")
                function_calls.append(f"{item['name']}({item['arguments']})")

            elif "type" in item and item["type"] == "function_call_output":
                print(f"\n[Function Output {i}]:")
                print(item["output"][:200] + "..." if len(item["output"]) > 200 else item["output"])
                function_outputs.append(item["output"])

    print(f"\n{'='*50}")
    print("STREAMING SUMMARY:")
    print(f"Function Calls: {len(function_calls)}")
    print(f"Function Outputs: {len(function_outputs)}")
    print(f"Final Response Length: {len(full_text)} characters")
    print(f"{'='*80}\n")

# COMMAND ----------

streaming_test_queries = [
    # Account queries
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What are the details of my account?"}], custom_inputs={"customer": get_next_customer()}), "Account Query Streaming"),
    (ResponsesAgentRequest(input=[{"role": "user", "content": "Show me all my active subscriptions"}], custom_inputs={"customer": get_next_customer()}), "Subscriptions Streaming"),

    # Product queries
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What's the difference between the Standard and Premium plans?"}], custom_inputs={"customer": get_next_customer()}), "Plan Comparison Streaming"),
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What devices do I have on my account?"}], custom_inputs={"customer": get_next_customer()}), "Customer Devices Streaming"),

    # Billing queries
    (ResponsesAgentRequest(input=[{"role": "user", "content": "What are my billing charges for April 2025?"}], custom_inputs={"customer": get_next_customer()}), "Billing Query Streaming"),
    (ResponsesAgentRequest(input=[{"role": "user", "content": "How much data did I use last month?"}], custom_inputs={"customer": get_next_customer()}), "Usage Query Streaming"),

    # Tech support (no customer ID needed)
    (ResponsesAgentRequest(input=[{"role": "user", "content": "My phone won't connect to WiFi. How do I fix this?"}]), "Tech Support Streaming"),
]

for request, description in streaming_test_queries:
    if hasattr(request, 'custom_inputs') and request.custom_inputs and 'customer' in request.custom_inputs:
        print(f"\n[Using Customer ID: {request.custom_inputs['customer']}]")
    display_streaming_response(request, description)
