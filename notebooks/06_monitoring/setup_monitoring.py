# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Production Monitoring
# MAGIC
# MAGIC This notebook sets up production monitoring for a deployed agent by registering
# MAGIC and activating scorers on the MLflow experiment.
# MAGIC
# MAGIC Run this notebook:
# MAGIC - After deploying a new agent
# MAGIC - When updating scorer definitions
# MAGIC - When adjusting sample rates
# MAGIC - To re-enable monitoring after stopping

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("experiment_name", "/Shared/telco_support_agent/dev/dev_telco_support_agent")
dbutils.widgets.text("env", "dev")
dbutils.widgets.dropdown("replace_existing", "true", ["true", "false"])

# COMMAND ----------

import mlflow

from telco_support_agent.evaluation import SCORER_CONFIGS, SCORER_VERSION, SCORERS
from telco_support_agent.ops.monitoring import (
    setup_production_monitoring,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

experiment_name = dbutils.widgets.get("experiment_name")
env = dbutils.widgets.get("env")
replace_existing = dbutils.widgets.get("replace_existing") == "true"

experiment = mlflow.set_experiment(experiment_name)

print("Monitoring Config:")
print(f"  Experiment: {experiment.name}")
print(f"  Experiment ID: {experiment.experiment_id}")
print(f"  Environment: {env}")
print(f"  Replace Existing: {replace_existing}")
print(f"  Scorer Version: {SCORER_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorer Config

# COMMAND ----------

print(f"Scorers to register: {len(SCORERS)}")
print()
for scorer in SCORERS:
    scorer_name = scorer.name
    config = SCORER_CONFIGS.get(scorer_name, {"sample_rate": 1.0})
    sample_rate = config["sample_rate"]
    print(f"  - {scorer_name}: {sample_rate * 100}% sampling")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Production Monitoring

# COMMAND ----------

print("=" * 50)
print("SETTING UP PRODUCTION MONITORING")
print("=" * 50)

scorers = setup_production_monitoring(
    experiment_id=experiment.experiment_id,
    replace_existing=replace_existing,
)

print()
print("=" * 50)
print("MONITORING SETUP COMPLETE")
print("=" * 50)
print(f"Active Scorers: {len(scorers)}")
print(f"Scorer Version: {SCORER_VERSION}")
print()
print("Scorers are now evaluating production traffic.")
print(f"View results in the MLflow experiment: {experiment.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Current Monitoring Status

# COMMAND ----------

from mlflow.genai.scorers import list_scorers

active_scorers = list_scorers(experiment_id=experiment.experiment_id)

print("Current Active Scorers:")
print()
for scorer in active_scorers:
    print(f"  - {scorer.name}")
    print(f"    Status: Active")
    if hasattr(scorer, "tags") and scorer.tags:
        print(f"    Tags: {scorer.tags}")

print()
print(f"Total: {len(active_scorers)} active scorers")