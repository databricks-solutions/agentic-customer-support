# Databricks notebook source
# MAGIC %md
# MAGIC # Log & Register Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("root_path", "")
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("git_commit", "")
dbutils.widgets.text("uc_catalog", "telco_customer_support_dev")
dbutils.widgets.text("agent_schema", "agent")
dbutils.widgets.text("data_schema", "gold")
dbutils.widgets.text("model_name", "telco_customer_support_agent")
dbutils.widgets.text("experiment_name", "/Shared/telco_support_agent/dev/dev_telco_support_agent")
dbutils.widgets.text("disable_tools", "")

# COMMAND ----------

import os
import sys

import mlflow
from mlflow.utils.databricks_utils import dbutils

if root_path := dbutils.widgets.get("root_path"):
    sys.path.append(root_path)



# COMMAND ----------

from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.config import WidgetConfigLoader, LogRegisterConfig
from telco_support_agent.ops.logging import log_agent
from telco_support_agent.ops.registry import register_agent_to_uc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# COMMAND ----------

config = WidgetConfigLoader(dbutils).load(LogRegisterConfig)

# COMMAND ----------

mlflow.set_experiment(config.experiment_name)

# COMMAND ----------

print("Configuration:")
print(f"  Name: {config.agent_name}")
print(f"  Description: {config.agent_description}")
print(f"  UC Model: {config.full_model_name}")
print(f"  Input Example: {config.input_example}")
print(f"  Environment: {config.env}")
print(f"  Git Commit: {config.git_commit}")
print(f"  Experiment Name: {config.experiment_name}")
print(f"  Disable Tools: {config.disable_tools}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC Functions

# COMMAND ----------

from telco_support_agent.tools import initialize_tools

print("Initializing required UC functions...")

# Create UC config from our notebook config
uc_config = config.to_uc_config()

results = initialize_tools(uc_config=uc_config)

success_count = 0
total_count = 0
for domain, functions in results.items():
    for func_name, status in functions.items():
        total_count += 1
        if status:
            success_count += 1
        status_str = "✅" if status else "❌"
        print(f"  {status_str} {func_name}")

print(f"\nInitialized {success_count}/{total_count} UC functions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Supervisor Agent

# COMMAND ----------

print("Testing supervisor agent...")

supervisor = SupervisorAgent(disable_tools=config.disable_tools, uc_config=uc_config)
print(f"Created supervisor agent (LLM: {supervisor.llm_endpoint})")

from mlflow.types.responses import ResponsesAgentRequest

test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "how much data did the customer use in May?"}],
    custom_inputs={"customer": "CUS-10001"}
)

response = supervisor.predict(test_request)
print("Test query completed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Intelligence Control

# COMMAND ----------

print("Testing intelligence control...")

# Test 1: Intelligence enabled (normal behavior)
test_request_intelligent = ResponsesAgentRequest(
    input=[{"role": "user", "content": "how much data did the customer use in May?"}],
    custom_inputs={"customer": "CUS-10001", "intelligence_enabled": True}
)

response_intelligent = supervisor.predict(test_request_intelligent)
print("✅ Intelligence enabled test completed")

# Test 2: Intelligence disabled (generic response)
test_request_generic = ResponsesAgentRequest(
    input=[{"role": "user", "content": "how much data did the customer use in May?"}],
    custom_inputs={"customer": "CUS-10001", "intelligence_enabled": False}
)

response_generic = supervisor.predict(test_request_generic)
print("✅ Intelligence disabled test completed")

# Show response previews
intelligent_text = ""
generic_text = ""

for output_item in response_intelligent.output:
    if hasattr(output_item, "content"):
        for content_item in output_item.content:
            if hasattr(content_item, "text"):
                intelligent_text = content_item.text
                break

for output_item in response_generic.output:
    if hasattr(output_item, "content"):
        for content_item in output_item.content:
            if hasattr(content_item, "text"):
                generic_text = content_item.text
                break

print(f"Intelligence enabled response: {intelligent_text[:200]}...")
print(f"Intelligence disabled response: {generic_text[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Agent to MLflow

# COMMAND ----------

print("Logging supervisor agent to MLflow...")

logged_model_info = log_agent(
    agent_class=SupervisorAgent,
    name=config.agent_name,
    input_example=config.input_example,
    environment=config.env,
    disable_tools=config.disable_tools,
    uc_config=uc_config,
)

print(f"Logged agent: {logged_model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Logged Model

# COMMAND ----------

print("Testing logged model...")

loaded_model = mlflow.pyfunc.load_model(logged_model_info.model_uri)

test_input = config.input_example
print(f"Testing with input: {test_input}")

response = loaded_model.predict(test_input)

print("✅ Logged model works correctly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog

# COMMAND ----------

print("Registering to Unity Catalog...")

model_version = register_agent_to_uc(
    model_uri=logged_model_info.model_uri,
    uc_model_name=config.full_model_name,
)

# tag model with the commit hash
if config.git_commit:
    print(f"Tagging model with git_commit: {config.git_commit}")
    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="git_commit",
        value=config.git_commit
    )

print(f"✅ Registered: {config.full_model_name} version {model_version.version}")
