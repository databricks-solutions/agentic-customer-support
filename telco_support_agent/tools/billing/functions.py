"""UC functions for billing-related operations."""

from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

from telco_support_agent.config import UCConfig
from telco_support_agent.utils.logging_utils import get_logger
from telco_support_agent.utils.uc_permissions import grant_function_permissions

logger = get_logger(__name__)

client = DatabricksFunctionClient()
workspace_client = WorkspaceClient()


def register_get_billing_info(uc_config: UCConfig):
    """Register the get_billing_info UC function."""
    function_name = uc_config.get_uc_function_name("get_billing_info")

    try:
        sql = f"""
        CREATE OR REPLACE FUNCTION {function_name}(
          customer STRING COMMENT 'The customer ID in the format CUS-XXXXX',
          billing_start_date_input STRING COMMENT 'The billing start date in YYYY-MM-DD format. Defaults to first day of current month.',
          billing_end_date_input STRING COMMENT 'The billing end date in YYYY-MM-DD format. Defaults to first day of next month.',
          additional_charges_input DOUBLE DEFAULT NULL COMMENT 'Filter on additional_charges. Must be non negative. If 0, only rows with non-NULL additional_charges are included. Can be NULL to skip filtering.',
          total_amount_input DOUBLE DEFAULT NULL COMMENT 'Filter on total_amount. Must be non negative. If 0, only rows with non-NULL total_amount are included. Can be NULL to skip filtering.',
          status_input STRING DEFAULT 'All' COMMENT 'Billing status. Possible values: Paid, Unpaid, Late, Partial, All. Defaults to All. If All, return rows of all statuses.'
        )
        RETURNS STRING
        COMMENT 'Retrieves all columns of the billing table for all rows for a customer within the specified date range, with optional filters for additional_charges, total_amount, and status. Example usage: SELECT {function_name}("CUS-10601","2025-06-01","2025-06-01",NULL,NULL,"Paid")'
        RETURN
        SELECT to_json(
                collect_list(
                  named_struct(
                    'customer_id', customer_id,
                    'billing_id', billing_id,
                    'subscription_id', subscription_id,
                    'billing_date', cast(billing_date as string),
                    'base_amount', cast(base_amount as string),
                    'additional_charges', cast(additional_charges as string),
                    'tax_amount', cast(tax_amount as string),
                    'total_amount', cast(total_amount as string),
                    'payment_amount', cast(payment_amount as string),
                    'payment_date', cast(payment_date as string),
                    'payment_method', payment_method,
                    'status', status
                  )
                )
              )
        FROM {uc_config.get_uc_table_name("billing")} AS billing_table
        WHERE billing_table.customer_id = customer
          AND billing_table.billing_date >= billing_start_date_input
          AND billing_table.billing_date < billing_end_date_input
          AND (
                (additional_charges_input IS NULL)
                OR (additional_charges_input = 0.0 AND billing_table.additional_charges IS NOT NULL)
                OR (additional_charges_input > 0.0 AND ABS(billing_table.additional_charges - additional_charges_input) <= 1.0)
              )
          AND (
                (total_amount_input IS NULL)
                OR (total_amount_input = 0.0 AND billing_table.total_amount IS NOT NULL)
                OR (total_amount_input > 0.0 AND ABS(billing_table.total_amount - total_amount_input) <= 1.0)
              )
          AND (
                (status_input = "All" AND billing_table.status IS NOT NULL)
                OR
                (status_input != "All" AND billing_table.status = status_input)
          )
        """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_billing_info: {str(e)}")


def register_get_usage_info(uc_config: UCConfig):
    """Register the get_usage_info UC function for usage details."""
    function_name = uc_config.get_uc_function_name("get_usage_info")

    try:
        sql = f"""
        CREATE OR REPLACE FUNCTION {function_name}(
          customer STRING COMMENT 'The customer ID in the format CUS-XXXXX',
          usage_start_date STRING COMMENT 'The usage start date in the format YYYY-MM-DD',
          usage_end_date STRING COMMENT 'The usage end date in the format YYYY-MM-DD'
        )
        RETURNS STRING
        COMMENT 'Retrieves usage data for a customer within the specified date range including data, voice, and SMS usage'
        RETURN
        SELECT to_json(
          named_struct(
            'customer_id', u.subscription_id,
            'total_usage', named_struct(
              'data_usage_gb', cast(sum(u.data_usage_mb) / 1024.0 as decimal(10,2)),
              'voice_minutes', cast(sum(u.voice_minutes) as int),
              'sms_count', cast(sum(u.sms_count) as int)
            ),
            'daily_usage', collect_list(
              named_struct(
                'date', cast(u.date as string),
                'data_usage_mb', cast(u.data_usage_mb as decimal(10,2)),
                'voice_minutes', cast(u.voice_minutes as int),
                'sms_count', cast(u.sms_count as int),
                'billing_cycle', u.billing_cycle
              )
            )
          )
        )
        FROM {uc_config.get_uc_table_name("usage")} u
        JOIN {uc_config.get_uc_table_name("subscriptions")} s ON u.subscription_id = s.subscription_id
        WHERE s.customer_id = customer
          AND u.date >= usage_start_date
          AND u.date < usage_end_date
        GROUP BY u.subscription_id
        LIMIT 1
        """

        client.create_function(sql_function_body=sql)
        print(f"Registered {function_name} UC function")

        # grant permissions
        if grant_function_permissions(function_name, uc_config, workspace_client):
            print(f"Granted permissions on {function_name}")
        else:
            print(
                f"Warning: Some permissions may not have been granted on {function_name}"
            )

    except Exception as e:
        print(f"Error registering get_usage_info: {str(e)}")


# Auto-registration with default UC config
# Note: This should ideally be called explicitly with proper UC config
if __name__ == "__main__":
    # Default UC config for auto-registration
    uc_config = UCConfig(
        agent_catalog="telco_customer_support_prod",
        agent_schema="agent",
        data_schema="gold",
        model_name="telco_customer_support_agent",
    )
    register_get_billing_info(uc_config)
    register_get_usage_info(uc_config)
