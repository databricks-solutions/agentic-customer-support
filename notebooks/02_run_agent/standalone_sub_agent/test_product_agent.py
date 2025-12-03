# Databricks notebook source
# MAGIC %md
# MAGIC # Test Product Agent Class

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
from telco_support_agent.agents.product import ProductAgent

# COMMAND ----------

print("Initializing tools for product agent...")
results = initialize_tools(domains=["product"])

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

product_agent = ProductAgent()

print(f"Agent type: {product_agent.agent_type}")
print(f"LLM endpoint: {product_agent.llm_endpoint}")
print(f"LLM parameters: {product_agent.llm_params}")
print(f"Number of tools: {len(product_agent.tools)}")

print("\nAvailable tools:")
for tool in product_agent.tools:
    if "function" in tool:
        print(f"- {tool['function']['name']}")

# COMMAND ----------

def test_query(query, custom_inputs):
    print(f"\n=== TESTING QUERY: \"{query}\" ===\n")

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": query}],
        custom_inputs=custom_inputs,
    )

    try:
        response = product_agent.predict(request)
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

test_query("Is my phone 5G compatible?", custom_inputs={"customer": "CUS-10011"})

# COMMAND ----------

test_queries = [
    "What's the difference between the Standard and Premium plans?",
    "Show me the plans with unlimited data",
    "Do you have any promotions for existing customers?",
    "Is my phone 5G compatible?",
    "Which plan gives me the most data for under $50?",
    "Do I have a Google phone?",
    "Do you have any device with more storage than mine?"
]

# COMMAND ----------

for query in test_queries:
    test_query(query, custom_inputs={"customer": "CUS-10011"})
