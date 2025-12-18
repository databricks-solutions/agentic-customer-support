# Databricks notebook source
# MAGIC %md
# MAGIC # Test Routing Cache
# MAGIC
# MAGIC This notebook tests the routing cache functionality:
# MAGIC 1. Creates the cache table and vector index (if needed)
# MAGIC 2. Runs test queries to populate the cache
# MAGIC 3. Tests similar queries to verify cache hits
# MAGIC 4. Analyzes cache performance metrics

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
import time

root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
print(f"Root path: {root_path}")

if root_path:
    sys.path.append(root_path)
    print(f"Added {root_path} to Python path")

# COMMAND ----------

from telco_support_agent.agents.supervisor import SupervisorAgent
from telco_support_agent.config import UCConfig
from telco_support_agent.data.vector_search.manager import VectorSearchManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Environment configuration
env = "prod"
catalog = f"telco_customer_support_{env}"
schema = "gold"
cache_table = f"{catalog}.{schema}.agent_cache"

# UC Configuration
uc_config = UCConfig(
    data_catalog=catalog,
    agent_catalog=catalog,
    data_schema=schema,
    agent_schema="agent",
    model_name="telco_customer_support_agent",
)

print(f"Environment: {env}")
print(f"Cache table: {cache_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialize Supervisor Agent
# MAGIC
# MAGIC Check that caching is enabled in the config

# COMMAND ----------

# Initialize supervisor agent (caching enabled in supervisor.yaml)
supervisor = SupervisorAgent(uc_config=uc_config)

# Check if cache is enabled
cache_enabled = supervisor.routing_cache is not None
print(f"Routing cache enabled: {cache_enabled}")

if cache_enabled:
    print(f"  Similarity threshold: {supervisor.routing_cache.similarity_threshold}")
else:
    print("⚠️  Caching is disabled! Enable in configs/agents/supervisor.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Populate Cache with Test Queries
# MAGIC
# MAGIC Run diverse queries to populate the cache

# COMMAND ----------

# Test queries covering different agent types
test_queries = [
    # Billing queries
    "What is my current bill amount?",
    "When is my payment due?",
    "Why was I charged extra this month?",
    "How much data did I use last month?",

    # Tech support queries
    "My phone won't connect to WiFi",
    "How do I reset my voicemail password?",
    "Is there a network outage in my area?",
    "My internet is very slow today",

    # Product queries
    "What plans do you offer?",
    "Tell me about the Premium plan",
    "Do you have any iPhone deals?",
    "What's the difference between Standard and Unlimited plans?",

    # Account queries
    "What is my account status?",
    "How long have I been a customer?",
    "What is my loyalty tier?",
    "Show me my subscription details",
]

print(f"Testing {len(test_queries)} queries to populate cache...\n")

# Track results
results = []
for i, query in enumerate(test_queries, 1):
    print(f"{i}. Query: {query[:60]}...")

    start_time = time.time()
    agent_type, cache_hit = supervisor.route_query(query)
    elapsed = time.time() - start_time

    result = {
        "query": query,
        "agent_type": agent_type.value,
        "cache_hit": cache_hit,
        "latency_ms": round(elapsed * 1000, 2)
    }
    results.append(result)

    status = "🎯 CACHE HIT" if cache_hit else "📝 CACHE MISS (cached)"
    print(f"   → {agent_type.value} agent | {status} | {elapsed*1000:.0f}ms\n")

    # Brief pause to let async cache writes complete
    time.sleep(0.5)

print(f"✅ Completed {len(test_queries)} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Sync Vector Index
# MAGIC
# MAGIC Cache entries have been written to the table. Now sync the vector index
# MAGIC so the cache lookups will work (index uses TRIGGERED pipeline type).

# COMMAND ----------

print("🔄 Syncing cache vector index...")
print("   (This may take 1-2 minutes depending on number of entries)")

vector_manager = VectorSearchManager(config_path=f"{root_path}/configs/data/create_vector_indexes.yaml")

if cache_enabled and supervisor.routing_cache:
    # Get cache index
    cache_index = supervisor.routing_cache.cache_index

    if cache_index:
        # Use VectorSearchManager to sync
        try:
            vector_manager.sync_index_and_wait(cache_index, "Cache")
            print("\n✅ Cache index sync complete!")
        except Exception as e:
            print(f"\n❌ Error syncing cache index: {e}")
    else:
        print("⚠️  Cache index not available")
else:
    print("⚠️  Cache not enabled, skipping sync")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Check Cache Population

# COMMAND ----------

# Query cache table to see what was stored
cache_entries = spark.sql(f"""
    SELECT
        query,
        response as agent_type,
        hit_count,
        created_time,
        last_hit_time
    FROM {cache_table}
    ORDER BY created_time DESC
""")

print(f"Cache entries: {cache_entries.count()}")
display(cache_entries)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test Cache Hits with Similar Queries
# MAGIC
# MAGIC Now test semantically similar queries that should hit the cache

# COMMAND ----------

# Similar queries that should match cached entries
similar_queries = [
    # Similar to: "What is my current bill amount?"
    ("What's my bill this month?", "billing"),
    ("How much do I owe?", "billing"),
    ("Show me my current charges", "billing"),

    # Similar to: "My phone won't connect to WiFi"
    ("Can't connect to wireless network", "tech_support"),
    ("WiFi connection issues", "tech_support"),
    ("Phone not connecting to internet", "tech_support"),

    # Similar to: "What plans do you offer?"
    ("What are the available plans?", "product"),
    ("Show me your plan options", "product"),
    ("Tell me about your plans", "product"),

    # Similar to: "What is my account status?"
    ("Check my account status", "account"),
    ("Is my account active?", "account"),
    ("What's my account state?", "account"),
]

print(f"Testing {len(similar_queries)} similar queries for cache hits...\n")

# Track cache hit performance
cache_hit_results = []

for i, (query, expected_agent) in enumerate(similar_queries, 1):
    print(f"{i}. Similar query: {query[:60]}...")

    start_time = time.time()
    agent_type, cache_hit = supervisor.route_query(query)
    elapsed = time.time() - start_time

    match = agent_type.value == expected_agent

    result = {
        "query": query,
        "expected_agent": expected_agent,
        "actual_agent": agent_type.value,
        "cache_hit": cache_hit,
        "match": match,
        "latency_ms": round(elapsed * 1000, 2)
    }
    cache_hit_results.append(result)

    status = "✅ CACHE HIT" if cache_hit else "❌ CACHE MISS"
    match_status = "✓ Correct" if match else f"✗ Wrong (expected {expected_agent})"
    print(f"   → {agent_type.value} | {status} | {match_status} | {elapsed*1000:.0f}ms\n")

    time.sleep(0.3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Analyze Cache Performance

# COMMAND ----------

# Calculate cache hit rate
total_similar = len(cache_hit_results)
cache_hits = sum(1 for r in cache_hit_results if r["cache_hit"])
cache_hit_rate = (cache_hits / total_similar * 100) if total_similar > 0 else 0

# Calculate accuracy
correct_routes = sum(1 for r in cache_hit_results if r["match"])
accuracy = (correct_routes / total_similar * 100) if total_similar > 0 else 0

# Calculate latency improvements
cache_hit_latencies = [r["latency_ms"] for r in cache_hit_results if r["cache_hit"]]
cache_miss_latencies = [r["latency_ms"] for r in cache_hit_results if not r["cache_hit"]]

avg_hit_latency = sum(cache_hit_latencies) / len(cache_hit_latencies) if cache_hit_latencies else 0
avg_miss_latency = sum(cache_miss_latencies) / len(cache_miss_latencies) if cache_miss_latencies else 0

print("=" * 60)
print("CACHE PERFORMANCE SUMMARY")
print("=" * 60)
print(f"\n📊 Cache Hit Rate: {cache_hit_rate:.1f}% ({cache_hits}/{total_similar})")
print(f"🎯 Routing Accuracy: {accuracy:.1f}% ({correct_routes}/{total_similar})")
print(f"\n⚡ Average Latency:")
print(f"   - Cache Hit:  {avg_hit_latency:.1f}ms")
print(f"   - Cache Miss: {avg_miss_latency:.1f}ms")
if avg_miss_latency > 0:
    speedup = ((avg_miss_latency - avg_hit_latency) / avg_miss_latency * 100)
    print(f"   - Speedup:    {speedup:.1f}% faster with cache")
print("=" * 60)
