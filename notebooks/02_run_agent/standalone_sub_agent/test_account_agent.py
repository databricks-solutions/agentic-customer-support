# Databricks notebook source
# MAGIC %md
# MAGIC # Test Account Agent

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

# init tools - will register UC functions if needed
from telco_support_agent.tools import initialize_tools

from mlflow.types.responses import ResponsesAgentRequest

from telco_support_agent.agents.account import AccountAgent

# COMMAND ----------

print("Initializing tools for account agent...")
results = initialize_tools(domains=["account"])

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

# init agent
account_agent = AccountAgent()

print(f"Agent type: {account_agent.agent_type}")
print(f"LLM endpoint: {account_agent.llm_endpoint}")
print(f"LLM parameters: {account_agent.llm_params}")
print(f"Number of tools: {len(account_agent.tools)}")

print("\nAvailable tools:")
for tool in account_agent.tools:
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
        response = account_agent.predict(request)
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

# Expect error if no custom_input customer is provided.
test_query("How many plans do I have on my account?", custom_inputs={})

# COMMAND ----------

test_query("How many plans do I have on my account?", custom_inputs={"customer": "CUS-10601"})

# COMMAND ----------

test_queries = [
    "What plan am I currently on?",
    "When did I create my account?",
    "Is my autopay enabled in my subscriptions?",
]

for query in test_queries:
    test_query(query, custom_inputs={"customer": "CUS-10601"})
