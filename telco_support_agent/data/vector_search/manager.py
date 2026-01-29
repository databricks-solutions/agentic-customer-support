"""Vector search index management.

Provides functionality to create, sync, and manage vector search
indexes for knowledge base and support tickets data.
"""

import time
from typing import Any, Optional

import yaml
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex

from telco_support_agent.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger, setup_logging
from telco_support_agent.utils.spark_utils import spark

setup_logging()
logger = get_logger(__name__)


class VectorSearchManager:
    """Manager for vector search indexes and endpoints."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        uc_config: Optional[UCConfig] = None,
        endpoint_name: Optional[str] = None,
    ):
        """Initialize the vector search manager.

        Args:
            config_path: Optional path to vector search config file
            uc_config: Optional Unity Catalog configuration
            endpoint_name: Optional vector search endpoint name (overrides config)
        """
        self.client = VectorSearchClient()
        self.config = self._load_config(config_path)
        self.uc_config = uc_config or UCConfig(
            agent_catalog="telco_customer_support_prod",
            agent_schema="agent",
            data_schema="gold",
            model_name="telco_customer_support_agent",
        )
        self.endpoint_name = endpoint_name

        # Set Spark session to use Unity Catalog as default
        spark.catalog.setCurrentCatalog(self.uc_config.data_catalog)
        spark.catalog.setCurrentDatabase(self.uc_config.data_schema)
        logger.info(f"Set Spark default catalog to: {self.uc_config.data_catalog}")
        logger.info(f"Set Spark default schema to: {self.uc_config.data_schema}")

        self._setup_names()

        logger.info("Vector search manager initialized")

    def _load_config(self, config_path: Optional[str] = None) -> dict[str, Any]:
        """Load vector search configuration.

        Args:
            config_path: Optional path to config file

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            raise FileNotFoundError(
                "Could not find create_vector_indexes.yaml config file. "
                "Please specify config_path or ensure file exists in configs/data/ directory"
            )

        logger.info(f"Loading vector search config from: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    def _setup_names(self) -> None:
        """Setup full table and index names from config."""
        # source tables
        self.kb_table = self.uc_config.get_uc_table_name(
            self.config["indexes"]["knowledge_base"]["source_table"]
        )
        self.tickets_table = self.uc_config.get_uc_table_name(
            self.config["indexes"]["support_tickets"]["source_table"]
        )

        # index names
        self.kb_index_name = self.uc_config.get_uc_index_name(
            self.config["indexes"]["knowledge_base"]["name"]
        )
        self.tickets_index_name = self.uc_config.get_uc_index_name(
            self.config["indexes"]["support_tickets"]["name"]
        )

        self.cache_table = self.uc_config.get_uc_table_name(
            self.config["indexes"]["cache"]["source_table"]
        )
        self.cache_index_name = self.uc_config.get_uc_index_name(
            self.config["indexes"]["cache"]["name"]
        )
        # vector search endpoint - use parameter if provided, otherwise fall back to config
        if self.endpoint_name is None:
            self.endpoint_name = self.config.get("endpoint", {}).get(
                "name", "telco-support-agent-vector-search"
            )

        logger.info("Setup complete:")
        logger.info(f"  Endpoint: {self.endpoint_name}")
        logger.info(f"  KB Table: {self.kb_table} -> {self.kb_index_name}")
        logger.info(
            f"  Tickets Table: {self.tickets_table} -> {self.tickets_index_name}"
        )
        logger.info(f"  Cache Table: {self.cache_table} -> {self.cache_index_name}")

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't already exist."""
        try:
            endpoint = self.client.get_endpoint(self.endpoint_name)
            logger.info(f"Endpoint '{self.endpoint_name}' already exists")
            logger.info(
                f"   Status: {endpoint.get('endpoint_status', {}).get('state', 'Unknown')}"
            )
            return endpoint
        except Exception:
            logger.info(f"Creating new endpoint '{self.endpoint_name}'...")

        try:
            self.client.create_endpoint(
                name=self.endpoint_name, endpoint_type=self.config["endpoint"]["type"]
            )
            logger.info(f"Created endpoint '{self.endpoint_name}'")

            timeout_minutes = self.config["timeouts"]["endpoint_creation"]
            logger.info(
                f"Waiting for endpoint to come online (timeout: {timeout_minutes} minutes)..."
            )
            self.client.wait_for_endpoint(self.endpoint_name)
            logger.info("Endpoint is online")

        except Exception as e:
            logger.error(f"Error creating endpoint: {e}")
            raise

    def verify_source_tables(self) -> dict[str, dict[str, Any]]:
        """Verify source tables exist and return their metadata.

        Returns:
            Dictionary with table metadata
        """
        results = {}

        try:
            kb_df = spark.table(self.kb_table)
            row_count = kb_df.count()
            logger.info(f"Knowledge base table exists: {self.kb_table}")
            logger.info(f"   Row count: {row_count}")

            results["knowledge_base"] = {
                "table_name": self.kb_table,
                "exists": True,
                "row_count": row_count,
                "schema": kb_df.schema,
            }
        except Exception as e:
            logger.error(f"Error accessing knowledge base table: {e}")
            results["knowledge_base"] = {
                "table_name": self.kb_table,
                "exists": False,
                "error": str(e),
            }

        try:
            tickets_df = spark.table(self.tickets_table)
            row_count = tickets_df.count()
            logger.info(f"Support tickets table exists: {self.tickets_table}")
            logger.info(f"   Row count: {row_count}")

            results["support_tickets"] = {
                "table_name": self.tickets_table,
                "exists": True,
                "row_count": row_count,
                "schema": tickets_df.schema,
            }
        except Exception as e:
            logger.error(f"Error accessing support tickets table: {e}")
            results["support_tickets"] = {
                "table_name": self.tickets_table,
                "exists": False,
                "error": str(e),
            }

        try:
            cache_df = spark.table(self.cache_table)
            row_count = cache_df.count()
            logger.info(f"Cache table exists: {self.cache_table}")
            logger.info(f"   Row count: {row_count}")

            results["cache"] = {
                "table_name": self.cache_table,
                "exists": True,
                "row_count": row_count,
                "schema": cache_df.schema,
            }
        except Exception as e:
            logger.error(f"Error accessing cache table: {e}")
            results["cache"] = {
                "table_name": self.cache_table,
                "exists": False,
                "error": str(e),
            }

        return results

    def create_knowledge_base_index(self) -> VectorSearchIndex:
        """Create vector search index for knowledge base content.

        Returns:
            VectorSearchIndex object
        """
        logger.info(f"Creating knowledge base index: {self.kb_index_name}")

        try:
            try:
                existing_index = self.client.get_index(index_name=self.kb_index_name)
                logger.info(f"WARNING: Index '{self.kb_index_name}' already exists")
                status = (
                    existing_index.describe()
                    .get("status", {})
                    .get("detailed_state", "Unknown")
                )
                logger.info(f"   Status: {status}")
                return existing_index
            except Exception as e:
                logger.debug(f"Index does not exist, proceeding with creation: {e}")

            kb_config = self.config["indexes"]["knowledge_base"]
            embedding_config = self.config["embedding"]

            # create index
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.kb_table,
                index_name=self.kb_index_name,
                pipeline_type=kb_config["pipeline_type"],
                primary_key=kb_config["primary_key"],
                embedding_source_column=kb_config["embedding_source_column"],
                embedding_model_endpoint_name=embedding_config["model_endpoint"],
                columns_to_sync=kb_config["columns_to_sync"],
            )

            logger.info(f"Created knowledge base index: {self.kb_index_name}")
            return index

        except Exception as e:
            logger.error(f"Error creating knowledge base index: {e}")
            raise

    def create_support_tickets_index(self) -> VectorSearchIndex:
        """Create vector search index for support tickets.

        Returns:
            VectorSearchIndex object
        """
        logger.info(f"Creating support tickets index: {self.tickets_index_name}")

        try:
            try:
                existing_index = self.client.get_index(
                    index_name=self.tickets_index_name
                )
                logger.info(
                    f"WARNING: Index '{self.tickets_index_name}' already exists"
                )
                status = (
                    existing_index.describe()
                    .get("status", {})
                    .get("detailed_state", "Unknown")
                )
                logger.info(f"   Status: {status}")
                return existing_index
            except Exception as e:
                logger.debug(f"Index does not exist, proceeding with creation: {e}")

            tickets_config = self.config["indexes"]["support_tickets"]
            embedding_config = self.config["embedding"]

            # Create the index
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.tickets_table,
                index_name=self.tickets_index_name,
                pipeline_type=tickets_config["pipeline_type"],
                primary_key=tickets_config["primary_key"],
                embedding_source_column=tickets_config["embedding_source_column"],
                embedding_model_endpoint_name=embedding_config["model_endpoint"],
                columns_to_sync=tickets_config["columns_to_sync"],
            )

            logger.info(
                f"Success! Created support tickets index: {self.tickets_index_name}"
            )
            return index

        except Exception as e:
            logger.error(f"Error creating support tickets index: {e}")
            raise

    def create_cache_index(self) -> VectorSearchIndex:
        """Create vector search index for agent response caching.

        Returns:
            VectorSearchIndex object

        Raises:
            ValueError: If cache index is not configured
        """
        if not self.cache_index_name:
            raise ValueError(
                "Cache index not configured. Add 'cache' to indexes in config file."
            )

        logger.info(f"Creating cache index: {self.cache_index_name}")

        try:
            try:
                existing_index = self.client.get_index(index_name=self.cache_index_name)
                logger.info(f"WARNING: Index '{self.cache_index_name}' already exists")
                status = (
                    existing_index.describe()
                    .get("status", {})
                    .get("detailed_state", "Unknown")
                )
                logger.info(f"   Status: {status}")
                return existing_index
            except Exception as e:
                logger.debug(f"Index does not exist, proceeding with creation: {e}")

            cache_config = self.config["indexes"]["cache"]
            embedding_config = self.config["embedding"]

            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.cache_table,
                index_name=self.cache_index_name,
                pipeline_type=cache_config["pipeline_type"],
                primary_key=cache_config["primary_key"],
                embedding_source_column=cache_config["embedding_source_column"],
                embedding_model_endpoint_name=embedding_config["model_endpoint"],
                columns_to_sync=cache_config["columns_to_sync"],
            )

            logger.info(f"Success! Created cache index: {self.cache_index_name}")
            return index

        except Exception as e:
            logger.error(f"Error creating cache index: {e}")
            raise

    def sync_index_and_wait(self, index: VectorSearchIndex, index_name: str) -> None:
        """Sync an index and wait for it to complete.

        Args:
            index: VectorSearchIndex object
            index_name: Human-readable name for logging
        """
        logger.info(f"Syncing {index_name}...")

        try:
            logger.info(f"Sync triggered for {index_name}")
            index.sync()

            timeout_minutes = self.config["timeouts"]["index_sync"]
            check_interval = self.config["timeouts"]["status_check_interval"] * 60

            logger.info(
                f"Waiting for {index_name} to be ready (timeout: {timeout_minutes} minutes)..."
            )
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60

            while True:
                status_info = index.describe()
                state = status_info.get("status", {}).get("detailed_state", "UNKNOWN")

                logger.info(f"   Current state: {state}")

                if state == "ONLINE" or state == "ONLINE_NO_PENDING_UPDATE":
                    logger.info(f"{index_name} is ONLINE and ready!")
                    break
                elif state in ["FAILED", "CANCELLED"]:
                    raise Exception(f"Index sync failed with state: {state}")
                elif time.time() - start_time > timeout_seconds:
                    raise Exception(f"Timeout waiting for {index_name} to be ready")

                time.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error syncing {index_name}: {e}")
            raise

    def test_index_search(self, index: VectorSearchIndex, index_type: str) -> None:
        """Test vector search functionality.

        Args:
            index: VectorSearchIndex object to test
            index_type: Type of index ('knowledge_base' or 'support_tickets')
        """
        logger.info(f"=== Testing {index_type} Search ===")

        test_queries = self.config["indexes"][index_type]["test_queries"]

        for query in test_queries:
            logger.info(f"Query: '{query}'")
            try:
                if index_type == "knowledge_base":
                    columns = [
                        "kb_id",
                        "title",
                        "category",
                        "subcategory",
                        "content_type",
                    ]
                else:
                    columns = ["ticket_id", "category", "priority", "status"]

                results = index.similarity_search(
                    query_text=query, columns=columns, num_results=3
                )

                data_array = results.get("result", {}).get("data_array", [])
                logger.info(f"Found {len(data_array)} results")

                # display top 2 results
                for i, result in enumerate(data_array[:2]):
                    if index_type == "knowledge_base":
                        logger.info(
                            f"  {i + 1}. {result[1]} ({result[2]}/{result[3]}) - {result[4]}"
                        )
                    else:
                        logger.info(
                            f"  {i + 1}. {result[0]} - {result[1]} ({result[2]}, {result[3]})"
                        )

            except Exception as e:
                logger.error(f"Error searching {index_type}: {e}")

    def get_index_status_summary(self) -> dict[str, dict[str, Any]]:
        """Get status summary for both indexes.

        Returns:
            Dictionary with status information for both indexes
        """
        summary = {}

        try:
            kb_index = self.client.get_index(index_name=self.kb_index_name)
            kb_status = kb_index.describe()
            summary["knowledge_base"] = {
                "name": self.kb_index_name,
                "state": kb_status.get("status", {}).get("detailed_state", "Unknown"),
                "index_type": kb_status.get("index_type", "Unknown"),
                "endpoint": kb_status.get("endpoint_name", "Unknown"),
                "exists": True,
            }
        except Exception as e:
            summary["knowledge_base"] = {
                "name": self.kb_index_name,
                "exists": False,
                "error": str(e),
            }

        try:
            tickets_index = self.client.get_index(index_name=self.tickets_index_name)
            tickets_status = tickets_index.describe()
            summary["support_tickets"] = {
                "name": self.tickets_index_name,
                "state": tickets_status.get("status", {}).get(
                    "detailed_state", "Unknown"
                ),
                "index_type": tickets_status.get("index_type", "Unknown"),
                "endpoint": tickets_status.get("endpoint_name", "Unknown"),
                "exists": True,
            }
        except Exception as e:
            summary["support_tickets"] = {
                "name": self.tickets_index_name,
                "exists": False,
                "error": str(e),
            }

        try:
            cache_index = self.client.get_index(index_name=self.cache_index_name)
            cache_status = cache_index.describe()
            summary["cache"] = {
                "name": self.cache_index_name,
                "state": cache_status.get("status", {}).get(
                    "detailed_state", "Unknown"
                ),
                "index_type": cache_status.get("index_type", "Unknown"),
                "endpoint": cache_status.get("endpoint_name", "Unknown"),
                "exists": True,
            }
        except Exception as e:
            summary["cache"] = {
                "name": self.cache_index_name,
                "exists": False,
                "error": str(e),
            }

        return summary

    def setup_all_indexes(self) -> dict[str, VectorSearchIndex]:
        """Complete setup process for all vector search indexes.

        Returns:
            Dictionary with created indexes
        """
        logger.info("Starting vector search setup...")

        # Step 1: Create endpoint
        self.create_endpoint_if_not_exists()

        # Step 2: Verify source tables
        table_status = self.verify_source_tables()

        # check if both tables exist
        if not table_status["knowledge_base"]["exists"]:
            raise RuntimeError(f"Knowledge base table does not exist: {self.kb_table}")
        if not table_status["support_tickets"]["exists"]:
            raise RuntimeError(
                f"Support tickets table does not exist: {self.tickets_table}"
            )
        if not table_status["cache"]["exists"]:
            raise RuntimeError(f"Cache table does not exist: {self.cache_table}")

        # Step 3: Create indexes
        kb_index = self.create_knowledge_base_index()
        tickets_index = self.create_support_tickets_index()
        cache_index = self.create_cache_index()
        # Step 4: Sync indexes
        self.sync_index_and_wait(kb_index, "Knowledge Base")
        self.sync_index_and_wait(tickets_index, "Support Tickets")
        self.sync_index_and_wait(cache_index, "Cache")
        # Step 5: Test indexes
        self.test_index_search(kb_index, "knowledge_base")
        self.test_index_search(tickets_index, "support_tickets")
        self.test_index_search(cache_index, "cache")

        logger.info("Vector search setup complete!")

        return {"knowledge_base": kb_index, "support_tickets": tickets_index}
