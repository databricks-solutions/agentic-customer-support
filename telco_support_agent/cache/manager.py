"""Cache manager for agent responses.

Provides functionality to read from and write to the agent response cache,
including background async writes and duplicate detection via similarity search.
"""

import threading
import uuid
from datetime import datetime
from typing import Optional

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

from telco_support_agent.config import UCConfig
from telco_support_agent.data.schemas.cache import CacheEntry
from telco_support_agent.utils.logging_utils import get_logger, setup_logging
from telco_support_agent.utils.spark_utils import spark

setup_logging()
logger = get_logger(__name__)


class CacheManager:
    """Manager for agent response caching with vector similarity search."""

    def __init__(
        self,
        uc_config: UCConfig,
        cache_index_name: str = "agent_cache_index",
        cache_table_name: str = "agent_cache",
        similarity_threshold: float = 0.8,
    ):
        """Initialize the cache manager.

        Args:
            uc_config: Unity Catalog configuration
            cache_index_name: Name of the cache vector index (without catalog.schema)
            cache_table_name: Name of the cache Delta table (without catalog.schema)
            similarity_threshold: Threshold for duplicate detection (0.0-1.0)
        """
        self.uc_config = uc_config
        self.cache_table = uc_config.get_uc_table_name(cache_table_name)
        self.cache_index_name = uc_config.get_uc_index_name(cache_index_name)
        self.similarity_threshold = similarity_threshold
        self.vs_client = VectorSearchClient()
        try:
            self.cache_index = self.vs_client.get_index(
                index_name=self.cache_index_name
            )
        except Exception as e:
            logger.warning(f"Cache index not yet available: {e}")
            self.cache_index = None

    def format_cache_content(self, query: str) -> str:
        """Format content for embedding.

        Args:
            query: User query text

        Returns:
            Formatted string for cache lookup
        """
        return f"Query: {query}"

    def get_cache(self, query: str) -> Optional[str]:
        """Search cache for matching response using vector similarity.

        Only returns cached response if similarity score meets or exceeds threshold.
        If a match is found, updates hit_count and last_hit_time.

        Args:
            query: User query text

        Returns:
            Cached response if found with sufficient similarity, None otherwise
        """
        if not self.cache_index:
            logger.debug("Cache index not available, skipping cache lookup")
            return None

        try:
            formatted_query = self.format_cache_content(query)

            results = self.cache_index.similarity_search(
                query_text=formatted_query,
                columns=["cache_id", "response", "query"],
                num_results=1,
                query_type="ANN",
                reranker=None,
            )
            docs = results.get("result", {}).get("data_array", [])

            if docs and len(docs) > 0:
                # Vector search returns: [cache_id, response, query, score]
                cache_id = docs[0][0]
                response = docs[0][1]
                cached_query = docs[0][2]
                similarity_score = docs[0][3] if len(docs[0]) > 3 else 0.0

                logger.info(
                    f"Cache candidate found: '{cached_query}' "
                    f"(score: {similarity_score:.4f}, threshold: {self.similarity_threshold})"
                )

                # Check if similarity score meets threshold
                if similarity_score >= self.similarity_threshold:
                    logger.info(
                        f"Cache hit! Score {similarity_score:.4f} >= threshold {self.similarity_threshold}"
                    )
                    self._update_cache_hit(cache_id)
                    return response
                else:
                    logger.info(
                        f"Cache miss: Score {similarity_score:.4f} below threshold {self.similarity_threshold}"
                    )
                    return None

            logger.debug("Cache miss: No results found")
            return None

        except Exception as e:
            logger.error(f"Error searching cache: {e}")
            return None

    def _update_cache_hit(self, cache_id: str) -> None:
        """Update hit_count and last_hit_time for a cache entry.

        Args:
            cache_id: ID of the cache entry to update
        """
        try:
            spark.sql(
                f"""
                UPDATE {self.cache_table}
                SET
                    hit_count = hit_count + 1,
                    last_hit_time = current_timestamp()
                WHERE cache_id = '{cache_id}'
                """
            )
            logger.debug(f"Updated hit count for cache_id: {cache_id}")

        except Exception as e:
            logger.error(f"Error updating cache hit count: {e}")

    def put_cache(self, query: str, response: str) -> None:
        """Write cache entry to Delta table.

        Args:
            query: User query text
            response: Cached response (e.g., agent_type for routing)
        """
        try:
            now = datetime.now()

            cache_entry = CacheEntry(
                cache_id=str(uuid.uuid4()),
                query=query,
                response=response,
                agent_type="supervisor",  # Fixed for routing cache
                customer_segment="routing",  # Fixed for routing cache
                formatted_content=self.format_cache_content(query),
                hit_count=0,
                last_hit_time=now,
                created_time=now,
            )

            df = spark.createDataFrame([cache_entry.model_dump()])
            df = df.withColumn("hit_count", F.col("hit_count").cast("int"))
            df.write.format("delta").mode("append").saveAsTable(self.cache_table)

            logger.info(f"Added cache entry: {cache_entry.cache_id}")

        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def add_to_cache_async(self, query: str, response: str) -> None:
        """Write cache entry asynchronously in background thread.

        This method returns immediately and writes the cache entry in a daemon thread.

        Args:
            query: User query text
            response: Cached response (e.g., agent_type for routing)
        """
        thread = threading.Thread(
            target=self.put_cache,
            args=(query, response),
            daemon=True,
        )
        thread.start()
        logger.debug("Background cache write initiated")
