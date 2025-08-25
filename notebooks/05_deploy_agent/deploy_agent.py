# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
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
dbutils.widgets.text("model_name", "telco_customer_support_agent")
dbutils.widgets.text("endpoint_name", "dev-telco-customer-support-agent")
dbutils.widgets.text("scale_to_zero_enabled", "false")
dbutils.widgets.text("workload_size", "Small")

# COMMAND ----------

import os
import sys

import mlflow

if root_path := dbutils.widgets.get("root_path"):
    sys.path.append(root_path)

# COMMAND ----------

from telco_support_agent.config import DeployAgentConfig, WidgetConfigLoader
from telco_support_agent.ops.deployment import (AgentDeploymentError,
                                                cleanup_old_deployments,
                                                deploy_agent)
from telco_support_agent.ops.registry import get_latest_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

config = WidgetConfigLoader(dbutils).load(DeployAgentConfig)
print("Config loaded successfully!")

# COMMAND ----------

experiment = mlflow.set_experiment(f"/Shared/telco_support_agent/{config.env}/{config.env}_telco_support_agent")
os.environ['MLFLOW_EXPERIMENT_ID'] = experiment.experiment_id
print(experiment.experiment_id)

# COMMAND ----------

print("Deployment configuration:")
print(f"  Model: {config.full_model_name}")
print(f"  Endpoint: {config.endpoint_name}")
print(f"  Scale to zero: {config.scale_to_zero_enabled}")
print(f"  Workload size: {config.workload_size}")
print(f"  Environment: {config.env}")
print(f"  Git commit: {config.git_commit}")
print(f"  Experiment ID: {experiment.experiment_id}")

os.environ['ENV'] = config.env

# COMMAND ----------

# Get model version
if config.model_version:
    model_version = config.model_version
    print(f"Using specified model version: {model_version}")
else:
    model_version = get_latest_model_version(config.full_model_name)
    if model_version is None:
        raise ValueError(f"No versions found for model: {config.full_model_name}")
    print(f"Using latest model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation
# MAGIC
# MAGIC Load and test the registered model before deployment.

# COMMAND ----------

model_uri = f"models:/{config.full_model_name}/{model_version}"
print(f"Loading model: {model_uri}")

try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully")

    test_queries = [
        {
            "input": [{"role": "user", "content": "What was the customer's data in May?"}],
            "custom_inputs": {"customer": "CUS-10001"}
        }
    ]

    print("Testing model predictions...")
    for i, test_input in enumerate(test_queries, 1):
        print(f"Test {i}: {test_input['input'][0]['content']}")
        response = loaded_model.predict(test_input)
        
        if response and "output" in response and len(response["output"]) > 0:
            print(f"✅ Test {i} passed")
        else:
            raise ValueError(f"Test {i} failed: Model returned empty or invalid response")
    
    print("✅ All model predictions successful")
    print("Proceeding with deployment...")

except Exception as e:
    print(f"❌ Pre-deployment validation failed: {str(e)}")
    raise RuntimeError("Model validation failed. Deployment aborted.") from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

print("Deploying agent..")
print(f"Model: {config.full_model_name} version {model_version}")
print(f"Endpoint: {config.endpoint_name}")
print(f"Workload Size: {config.workload_size}")
print(f"Scale-to-zero: {config.scale_to_zero_enabled}")
print(f"Wait for endpoint to be ready: {config.wait_for_ready}")
print(f"Environment: {config.env}")
print(f"Git Commit: {config.git_commit}")

if config.permissions:
    print("Setting permissions for:")
    for perm_config in config.permissions:
        users = perm_config.get('users', [])
        permission_level = perm_config.get('permission_level', 'Unknown')
        print(f"  - {permission_level}: {', '.join(users)}")

if config.instructions:
    print("Setting review instructions")

try:
    deployment_result = deploy_agent(
        uc_model_name=config.full_model_name,
        model_version=model_version,
        deployment_name=config.endpoint_name,
        tags={"environment": config.env, "git_commit": config.git_commit} if config.git_commit else {"environment": config.env},
        scale_to_zero_enabled=config.scale_to_zero_enabled,
        environment_vars={"ENV": config.env},
        workload_size=config.workload_size,
        wait_for_ready=config.wait_for_ready,
        permissions=config.permissions,
        instructions=config.instructions,
        budget_policy_id=None,
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
print(f"Model: {config.full_model_name} (version {model_version})")
print(f"Workload Size: {config.workload_size}")
print(f"Scale-to-zero: {'Enabled' if config.scale_to_zero_enabled else 'Disabled'}")
print(f"Query Endpoint: {deployment_result.query_endpoint}")

if hasattr(deployment_result, 'review_app_url'):
    print(f"Review App: {deployment_result.review_app_url}")

print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create External Monitor
# MAGIC
# MAGIC Set up agent monitoring for the deployed agent

# COMMAND ----------

from telco_support_agent.evaluation import SCORERS
from telco_support_agent.ops.monitoring import (AgentMonitoringError,
                                                create_agent_monitor)

if config.monitoring_enabled:
    print("="*50)
    print("SETTING UP AGENT MONITORING")
    print("="*50)
    print(f"Experiment Name: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Agent catalog: {config.uc_catalog}")
    print(f"Agent schema: {config.agent_schema}")
    # display custom metrics
    print("Custom Telco Assessments:")
    scorers = SCORERS[:4]
    for scorer in scorers:
        print(f"  - {scorer.name}")
    print()

    try:
        # create external monitor with custom metrics
        uc_config = config.to_uc_config()
        monitor = create_agent_monitor(
            uc_config=uc_config,
            experiment_id=experiment.experiment_id,
            replace_existing=config.monitoring_replace_existing,
            custom_metrics=[scorer.get_custom_metric() for scorer in scorers],
        )

        print("✅ External monitor created successfully!")
        print(f"Monitor ID: {getattr(monitor, 'id', 'N/A')}")
        print(f"Experiment ID: {getattr(monitor, 'experiment_id', 'N/A')}")
        print(f"Evaluated traces table: {getattr(monitor, 'evaluated_traces_table', 'N/A')}")

        if hasattr(monitor, 'monitoring_page_url'):
            print(f"Monitoring page: {monitor.monitoring_page_url}")

        print(f"\nNote: Monitor created with {len(scorers)} custom scorer assessments.")

    except AgentMonitoringError as e:
        print(f"❌ Failed to create monitor: {str(e)}")
        if config.monitoring_fail_on_error:
            raise
        else:
            print("Continuing deployment...")
    except Exception as e:
        print(f"❌ Unexpected monitoring error: {str(e)}")
        if config.monitoring_fail_on_error:
            raise
        else:
            print("Continuing deployment...")

    print("="*50)
else:
    print("External monitoring is disabled in configuration")
    print("To enable, set monitoring_enabled: true in configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up old model serving endpoints

# COMMAND ----------

if config.cleanup_old_versions:
    print("="*50)
    print("CLEANING UP OLD DEPLOYMENT VERSIONS")
    print("="*50)
    print(f"Model: {config.full_model_name}")
    print(f"Current version: {model_version}")
    print(f"Endpoint: {deployment_result.endpoint_name}")
    print(f"Keep previous versions: {config.keep_previous_count}")
    print()

    try:
        cleanup_result = cleanup_old_deployments(
            model_name=config.full_model_name,
            current_version=str(model_version),
            endpoint_name=deployment_result.endpoint_name,
            keep_previous_count=config.keep_previous_count,
            max_deletion_attempts=3,
            wait_between_attempts=60,
            wait_after_deletion=180,
            raise_on_error=False,
        )

        print("✅ Cleanup completed!")
        print(f"Versions kept: {cleanup_result['versions_kept']}")
        print(f"Versions deleted: {cleanup_result['versions_deleted']}")

        if cleanup_result['versions_failed']:
            print(f"⚠️ Versions that failed to delete: {cleanup_result['versions_failed']}")
            print("These may need manual cleanup or will be retried in future deployments.")

        if not cleanup_result['versions_deleted'] and not cleanup_result['versions_failed']:
            print("No old versions found to clean up.")

    except AgentDeploymentError as e:
        print(f"❌ Cleanup failed with error: {str(e)}")
        print("Continuing despite cleanup failure")
    except Exception as e:
        print(f"❌ Unexpected cleanup error: {str(e)}")
        print("Continuing despite cleanup failure")

    print("="*50)
else:
    print("Cleanup of old versions is disabled in configuration")
    print("To enable, set cleanup_old_versions: true in configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployed Endpoint
# MAGIC
# MAGIC Verify deployed endpoint works correctly

# COMMAND ----------

from mlflow.deployments import get_deploy_client

print("Testing deployed endpoint with custom inputs...")

client = get_deploy_client()

test_cases = [
    {
        "input": [{"role": "user", "content": "What plan am I currently on?"}],
        "custom_inputs": {"customer": "CUS-10001"},
        "description": "Account query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "Show me my billing details for this month"}],
        "custom_inputs": {"customer": "CUS-10002"},
        "description": "Billing query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "What devices do I have on my account?"}],
        "custom_inputs": {"customer": "CUS-10003"},
        "description": "Product query with customer ID"
    },
    {
        "input": [{"role": "user", "content": "My phone won't connect to WiFi"}],
        "description": "Tech support query (no custom inputs required)"
    },
    # Test intelligence control
    {
        "input": [{"role": "user", "content": "What was my data usage in May?"}],
        "custom_inputs": {"customer": "CUS-10001", "intelligence_enabled": True},
        "description": "Intelligence enabled - should use tools"
    },
    {
        "input": [{"role": "user", "content": "What was my data usage in May?"}],
        "custom_inputs": {"customer": "CUS-10001", "intelligence_enabled": False},
        "description": "Intelligence disabled - should provide generic response"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i}: {test_case['description']} ---")

    try:
        request_data = {
            "input": test_case["input"],
            "databricks_options": {"return_trace": True}
        }

        if "custom_inputs" in test_case:
            request_data["custom_inputs"] = test_case["custom_inputs"]
            print(f"Custom inputs: {test_case['custom_inputs']}")

        response = client.predict(
            endpoint=deployment_result.endpoint_name,
            inputs=request_data
        )

        print("✅ Query successful!")

        for output in response["output"]:
            if "content" in output:
                for content in output["content"]:
                    if "text" in content:
                        print(f"Response: {content['text'][:200]}...")
                        break

        if "custom_outputs" in response:
            print(f"Custom outputs: {response['custom_outputs']}")

    except Exception as e:
        print(f"❌ Query failed: {str(e)}")

print("\n🎉 Custom inputs endpoint testing completed!")
