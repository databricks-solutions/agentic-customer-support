"""Cache table generator.

Handles creation of the agent cache Delta table.
"""

from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from telco_support_agent.data.config import ConfigDict
from telco_support_agent.data.generators.base import BaseGenerator
from telco_support_agent.utils.spark_utils import spark


class CacheGenerator(BaseGenerator):
    """Generator for cache table creation."""

    def __init__(self, config: ConfigDict):
        """Initialize the cache generator.

        Args:
            config: Data generation configuration (CONFIG dict)
        """
        super().__init__(config)

    def get_cache_schema(self) -> StructType:
        """Get the schema for the cache table.

        Returns:
            StructType defining the cache table schema
        """
        return StructType(
            [
                StructField("cache_id", StringType(), False),
                StructField("query", StringType(), False),
                StructField("response", StringType(), False),
                StructField("agent_type", StringType(), False),
                StructField("customer_segment", StringType(), False),
                StructField("formatted_content", StringType(), False),
                StructField("hit_count", IntegerType(), False),
                StructField("last_hit_time", TimestampType(), False),
                StructField("created_time", TimestampType(), False),
            ]
        )

    def create_cache_table(self, table_name: str) -> None:
        """Create an empty cache table with proper schema and properties.

        The table starts empty and will be populated at runtime by the CacheManager
        as the agent processes queries.

        Args:
            table_name: Full table name (catalog.schema.table)
        """
        # Create empty DataFrame with cache schema
        cache_schema = self.get_cache_schema()
        empty_df = spark.createDataFrame([], cache_schema)

        # Save to Delta table
        self.save_to_delta(empty_df, table_name, mode="overwrite")

        # Enable Change Data Feed (required for vector index sync)
        spark.sql(
            f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES (
                'delta.enableChangeDataFeed' = 'true',
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true'
            )
        """
        )

        # Add table documentation
        spark.sql(
            f"""
            COMMENT ON TABLE {table_name} IS
            'Agent routing cache for semantic similarity-based query matching.
            Stores routing decisions to reduce LLM calls and improve latency.
            Table starts empty and populates at runtime.'
        """
        )
