# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
import yaml
import mlflow

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)
print(f"Added {project_root} to Python path")

# COMMAND ----------

from telco_support_agent.ops.deployment import deploy_agent, AgentDeploymentError
from telco_support_agent.ops.registry import get_latest_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Deployment Config

# COMMAND ----------

CONFIG_PATH = "../../configs/deploy_agent.yaml"

with open(CONFIG_PATH) as f:
    deploy_agent_config = yaml.safe_load(f)

print("Loaded deployment configuration:")
print(yaml.dump(deploy_agent_config, sort_keys=False, default_flow_style=False))

# COMMAND ----------

uc_config = deploy_agent_config["uc_model"]
deployment_config = deploy_agent_config.get("deployment", {})
environment_vars = deploy_agent_config.get("environment_vars", {})
permissions = deploy_agent_config.get("permissions")
instructions = deploy_agent_config.get("instructions")

uc_model_name = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

if "version" in uc_config:
    model_version = uc_config["version"]
    print(f"Using specified model version: {model_version}")
else:
    model_version = get_latest_model_version(uc_model_name)
    if model_version is None:
        raise ValueError(f"No versions found for model: {uc_model_name}")
    print(f"Using latest model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation
# MAGIC 
# MAGIC Load and test the registered model before deployment.

# COMMAND ----------

model_uri = f"models:/{uc_model_name}/{model_version}"
print(f"Loading model: {model_uri}")

try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully")
    
    test_input = {
        "input": [{"role": "user", "content": "What plan am I currently on? My customer ID is CUS-10001."}]
    }
    
    print("Testing model prediction...")
    response = loaded_model.predict(test_input)
    
    if response and "output" in response and len(response["output"]) > 0:
        print("✅ Model prediction successful")
        print("Proceeding with deployment...")
    else:
        raise ValueError("Model returned empty or invalid response")
        
except Exception as e:
    print(f"❌ Pre-deployment validation failed: {str(e)}")
    raise RuntimeError("Model validation failed. Deployment aborted.") from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

print("Deploying agent..")
print(f"Model: {uc_model_name} version {model_version}")
print(f"Endpoint: {deployment_config.get('endpoint_name')}")
print(f"Workload Size: {deployment_config.get('workload_size', 'Small')}")
print(f"Scale-to-zero: {deployment_config.get('scale_to_zero_enabled', False)}")
print(f"Wait for endpoint to be ready: {deployment_config.get('wait_for_ready', True)}")

if environment_vars:
    print(f"Environment variables: {list(environment_vars.keys())}")

if permissions:
    print(f"Setting permissions for: {permissions.get('users', [])}")

if instructions:
    print("Setting review instructions")

try:
    deployment_result = deploy_agent(
        uc_model_name=uc_model_name,
        model_version=model_version,
        deployment_name=deployment_config.get("endpoint_name"),
        tags=deployment_config.get("tags"),
        scale_to_zero_enabled=deployment_config.get("scale_to_zero_enabled", False),
        environment_vars=environment_vars if environment_vars else None,
        workload_size=deployment_config.get("workload_size", "Small"),
        wait_for_ready=deployment_config.get("wait_for_ready", True), 
        permissions=permissions,
        instructions=instructions,
        budget_policy_id=deployment_config.get("budget_policy_id"),
    )

    print("✅ Deployment completed successfully!")

except AgentDeploymentError as e:
    print(f"❌ Deployment failed: {str(e)}")
    raise
except Exception as e:
    print(f"❌ Unexpected deployment error: {str(e)}")
    raise

# COMMAND ----------

print("\n" + "="*50)
print("DEPLOYMENT SUMMARY")
print("="*50)
print(f"Endpoint Name: {deployment_result.endpoint_name}")
print(f"Model: {uc_model_name} (version {model_version})")
print(f"Workload Size: {deployment_config.get('workload_size', 'Small')}")
print(f"Scale-to-zero: {'Enabled' if deployment_config.get('scale_to_zero_enabled', False) else 'Disabled'}")
print(f"Query Endpoint: {deployment_result.query_endpoint}")

if hasattr(deployment_result, 'review_app_url'):
    print(f"Review App: {deployment_result.review_app_url}")

print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployed Endpoint
# MAGIC 
# MAGIC Verify the deployed endpoint works correctly

# COMMAND ----------

from mlflow.deployments import get_deploy_client

print("Testing deployed endpoint...")

client = get_deploy_client()
test_queries = [
    "What plan am I currently on? My customer ID is CUS-10001.",
    "Hello, how can you help me today?",
    "I need help with my bill",
]

for i, test_query in enumerate(test_queries, 1):
    print(f"\n--- Test Query {i}: {test_query} ---")
    
    try:
        response = client.predict(
            endpoint=deployment_result.endpoint_name,
            inputs={
                "input": [{"role": "user", "content": test_query}],
                "databricks_options": {"return_trace": True}
            }
        )

        print("✅ Query successful!")
        
        for output in response["output"]:
            if "content" in output:
                for content in output["content"]:
                    if "text" in content:
                        print(f"Response: {content['text']}")
                        break
                        
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")

print("\n🎉 Endpoint testing completed!")