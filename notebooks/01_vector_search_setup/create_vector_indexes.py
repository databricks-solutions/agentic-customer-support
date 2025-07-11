# Databricks notebook source
# MAGIC %md
# MAGIC # Tech Support Vector Search Index Setup
# MAGIC
# MAGIC Creates vector search indexes for tech support sub-agent:
# MAGIC 1. **Knowledge Base Index** - Policies, FAQs, guides
# MAGIC 2. **Support Tickets Index** - Historical support tickets
# MAGIC
# MAGIC Uses `formatted_content` column

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("env", "dev")

# COMMAND ----------

import os
import sys
from pathlib import Path
from mlflow.utils.databricks_utils import dbutils

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
print(f"Root path: {root_path}")

if root_path:
    sys.path.append(root_path)

env = dbutils.widgets.get("env")

os.environ['TELCO_SUPPORT_AGENT_ENV'] = env

# COMMAND ----------

from telco_support_agent.data.vector_search import VectorSearchManager
from telco_support_agent.agents import UCConfig
from telco_support_agent.utils.logging_utils import setup_logging

setup_logging()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Vector Search Manager

# COMMAND ----------

config_path = str(Path(root_path) / "configs" / "vector_search.yaml")
print(f"Config path: {config_path}")

# Create UC config based on environment
uc_config = UCConfig(
    catalog=f"telco_customer_support_{env}",
    agent_schema="agent",
    data_schema="gold",
    model_name="telco_customer_support_agent"
)

vs_manager = VectorSearchManager(config_path=config_path, uc_config=uc_config)

print("✅ Vector Search Manager initialized successfully")
print(f"   Endpoint: {vs_manager.endpoint_name}")
print(f"   Knowledge Base: {vs_manager.kb_table} -> {vs_manager.kb_index_name}")
print(f"   Support Tickets: {vs_manager.tickets_table} -> {vs_manager.tickets_index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Vector Search Infrastructure
# MAGIC
# MAGIC This will:
# MAGIC 1. Create vector search endpoint (if needed)
# MAGIC 2. Verify source tables with `formatted_content` columns exist
# MAGIC 3. Create both vector search indexes
# MAGIC 4. Sync indexes with source data
# MAGIC 5. Test indexes with sample queries

# COMMAND ----------

print("Starting complete vector search setup...")

indexes = vs_manager.setup_all_indexes()

print("\nSetup completed successfully!")
print(f"   Knowledge Base Index: {vs_manager.kb_index_name}")
print(f"   Support Tickets Index: {vs_manager.tickets_index_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Index Status

# COMMAND ----------

status_summary = vs_manager.get_index_status_summary()

print("=== Vector Search Index Status ===\n")

for index_type, status in status_summary.items():
    print(f"{index_type.replace('_', ' ').title()} Index:")

    if status.get('exists', False):
        print(f"  ✅ Name: {status['name']}")
        print(f"     State: {status['state']}")
        print(f"     Type: {status['index_type']}")
        print(f"     Endpoint: {status['endpoint']}")
    else:
        print(f"  ❌ Index does not exist: {status['name']}")
        if 'error' in status:
            print(f"     Error: {status['error']}")

    print()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Search Functionality

# COMMAND ----------

print("=== Testing Knowledge Base Search ===")
kb_index = vs_manager.client.get_index(index_name=vs_manager.kb_index_name)

test_query = "billing dispute policy"
print(f"Query: '{test_query}'")

results = kb_index.similarity_search(
    query_text=test_query,
    columns=["kb_id", "title", "category", "content_type"],
    num_results=3
)

data_array = results.get('result', {}).get('data_array', [])
print(f"Found {len(data_array)} results:")

for i, result in enumerate(data_array[:3]):
    print(f"  {i+1}. [{result[0]}] {result[1]} ({result[2]}, {result[3]})")

print("\n" + "="*50 + "\n")


print("=== Testing Support Tickets Search ===")
tickets_index = vs_manager.client.get_index(index_name=vs_manager.tickets_index_name)

test_query = "phone won't connect"
print(f"Query: '{test_query}'")

results = tickets_index.similarity_search(
    query_text=test_query,
    columns=["ticket_id", "category", "priority", "status"],
    num_results=3
)

data_array = results.get('result', {}).get('data_array', [])
print(f"Found {len(data_array)} results:")

for i, result in enumerate(data_array[:3]):
    print(f"  {i+1}. {result[0]} - {result[1]} ({result[2]}, {result[3]})")

# COMMAND ----------


