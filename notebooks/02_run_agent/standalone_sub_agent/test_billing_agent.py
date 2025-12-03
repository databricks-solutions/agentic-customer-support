# Databricks notebook source
# MAGIC %md
# MAGIC # Test Billing Agent

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
print(f"Root path: {root_path}")

if root_path:
    sys.path.append(root_path)
    print(f"Added {root_path} to Python path")

# COMMAND ----------

from telco_support_agent.tools import initialize_tools

from mlflow.types.responses import ResponsesAgentRequest
from telco_support_agent.agents.billing import BillingAgent

# COMMAND ----------

print("Initializing tools for billing agent...")
results = initialize_tools(domains=["billing"])

print("\nFunction initialization status:")
for domain, functions in results.items():
    print(f"\nDomain: {domain}")
    for func_name, status in functions.items():
        status_str = "Available" if status else "Unavailable"
        print(f"  - {func_name}: {status_str}")

if any(not all(functions.values()) for functions in results.values()):
    print("\nWARNING: Some functions could not be initialized")
    print("Tests might fail without the necessary UC functions")
else:
    print("\nAll required functions are available")

# COMMAND ----------

billing_agent = BillingAgent()

print(f"Agent type: {billing_agent.agent_type}")
print(f"LLM endpoint: {billing_agent.llm_endpoint}")
print(f"LLM parameters: {billing_agent.llm_params}")
print(f"Number of tools: {len(billing_agent.tools)}")

print("\nAvailable tools:")
for tool in billing_agent.tools:
    if "function" in tool:
        print(f"- {tool['function']['name']}")

# COMMAND ----------

def test_query(query,customer_id):
    print(f"\n=== TESTING QUERY: \"{query}\" ===\n")

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}],
        custom_inputs={"customer": customer_id}
    )

    try:
        response = billing_agent.predict(request)

        if response and hasattr(response, 'output') and response.output:
            output_item = response.output[-1]
            if "content" in output_item and isinstance(output_item["content"], list):
                print(output_item["content"][0]["text"])
            elif "content" in output_item:
                print(output_item["content"])
            else:
                print("Response:", output_item)
        else:
            print("No response or empty response received")
    except Exception as e:
        print(f"Error processing query: {e}")

    print("\n" + "="*80)

# COMMAND ----------

test_query("What are the charges on my bill for from 2025-04-01 to 2025-04-30?", "CUS-10001")

# COMMAND ----------

test_queries = [
    ("What are the charges on my bill for customer CUS-10001 from 2025-04-01 to 2025-04-30?", "CUS-10001"),
    ("When is my June payment due", "CUS-10001"),
    ("I see an additional charge for $39 in my May bill that I don't recognize. show me my bill", "CUS-11094"), ## validated customer ID and additional charge
    ("How much data did I use from 2025-04-01 to 2025-04-30?", "CUS-10001"),
    ("Is there an unpaid amount from 2025-04-01 to 2025-04-30?", "CUS-10001"),
    ("Show me my billing history for the last 3 months", "CUS-10002"),
]

# COMMAND ----------

for query, customer_id in test_queries:
    test_query(query, customer_id)
