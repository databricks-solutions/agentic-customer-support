# Databricks notebook source
# MAGIC %md
# MAGIC # Run Offline Evals
# MAGIC
# MAGIC offline evals using custom scorers

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("experiment_name", "/Shared/telco_support_agent/dev/dev_telco_support_agent")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("uc_catalog", "telco_customer_support_dev")
dbutils.widgets.text("agent_schema", "agent")
dbutils.widgets.text("model_name", "telco_customer_support_agent")

# COMMAND ----------

import os
import sys

import mlflow
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)


# COMMAND ----------

from telco_support_agent.config import RunEvalsConfig, WidgetConfigLoader
from telco_support_agent.evaluation import SCORERS
from telco_support_agent.ops.registry import get_latest_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

config = WidgetConfigLoader(dbutils).load(RunEvalsConfig)

mlflow.set_experiment(config.experiment_name)

print("Config loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Model Version

# COMMAND ----------

uc_model_name = config.to_uc_config().get_uc_model_name()

model_version = config.model_version

if model_version:
    print(f"Using specified model version: {model_version}")
else:
    model_version = get_latest_model_version(uc_model_name)
    print(f"Using latest model version: {model_version}")

model_uri = f"models:/{uc_model_name}/{model_version}"
print(f"Model: {model_uri}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Evaluation Data

# COMMAND ----------

eval_data = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "What plan am I currently on?"}],
            "custom_inputs": {"customer": "CUS-10001"}
        }
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "My internet is very slow today"}],
            "custom_inputs": {"customer": "CUS-10002"}
        }
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Can you show me my last bill?"}],
            "custom_inputs": {"customer": "CUS-10003"}
        }
    }
]

print(f"Evaluation dataset: {len(eval_data)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation

# COMMAND ----------

print(f"Available scorers: {len(SCORERS)}")
for scorer in SCORERS:
    print(f"  - {scorer.name}")

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=model_uri)
def run_model(input, custom_inputs):
  return model.predict({'input': input, 'custom_inputs': custom_inputs})

# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_model,
    scorers=SCORERS,
)

print("Evaluation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# evaluation metrics
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.metrics.items():
    print(f"  {metric_name}: {metric_value:.3f}")

# COMMAND ----------

# detailed results - get traces with assessments
eval_traces = mlflow.search_traces(run_id=eval_results.run_id)
print(f"Detailed results: {eval_traces.shape[0]} traces x {eval_traces.shape[1]} columns")

print("\nTrace columns:")
for col in eval_traces.columns:
    print(f"  - {col}")

# COMMAND ----------

eval_traces.iloc[0]['trace']

# COMMAND ----------

if len(eval_traces) > 0:
    print("Sample assessments from first trace:")
    sample_assessments = eval_traces.iloc[0]['assessments']
    for assessment in sample_assessments:
        print(f"  - {assessment["assessment_name"]}: {assessment["feedback"]["value"]}")
        if "rationale" in assessment:
            print(f"    Rationale: {assessment["rationale"]}")
        print()
