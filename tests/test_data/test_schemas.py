"""
Tests for data schema validation.

Module to test the Pydantic models to ensure proper validation of data.
"""

from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from telco_support_agent.data.schemas.billing import Billing, Usage
from telco_support_agent.data.schemas.cache import CacheEntry
from telco_support_agent.data.schemas.customers import Customer, Subscription
from telco_support_agent.data.schemas.knowledge_base import KnowledgeBase, SupportTicket
from telco_support_agent.data.schemas.products import Device, Plan, Promotion


# Helper function to get current date and future date
def get_test_dates():
    """Get current date and a future date for testing."""
    today = date.today()
    future = today + timedelta(days=30)
    return today, future


class TestCustomerSchemas:
    """Tests for customer and subscription schemas."""

    def test_valid_customer(self):
        """Test that a valid customer passes validation."""
        customer = Customer(
            customer_id="CUS-10001",
            customer_segment="Individual",
            city="New York",
            state="NY",
            registration_date=date(2020, 1, 1),
            customer_status="Active",
            preferred_contact_method="Email",
        )
        assert customer.customer_id == "CUS-10001"
        assert customer.customer_segment == "Individual"

    def test_invalid_customer_id(self):
        """Test that an invalid customer ID fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Customer(
                customer_id="CUSTOMER-10001",  # Invalid format
                customer_segment="Individual",
                city="New York",
                state="NY",
                registration_date=date(2020, 1, 1),
                customer_status="Active",
                preferred_contact_method="Email",
            )
        # Check for pattern mismatch error
        assert "string_pattern_mismatch" in str(exc_info.value)
        assert "CUS-\\d{5}" in str(exc_info.value)

    def test_invalid_customer_segment(self):
        """Test that an invalid customer segment fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Customer(
                customer_id="CUS-10001",
                customer_segment="Unknown",  # Invalid segment
                city="New York",
                state="NY",
                registration_date=date(2020, 1, 1),
                customer_status="Active",
                preferred_contact_method="Email",
            )
        assert "Customer segment must be one of" in str(exc_info.value)

    def test_valid_subscription(self):
        """Test that a valid subscription passes validation."""
        subscription = Subscription(
            subscription_id="SUB-12345678",
            customer_id="CUS-10001",
            plan_id="PLAN-1001",
            device_id="DEV-2001",
            promo_id="PROMO-4001",
            subscription_start_date=date(2020, 2, 1),
            contract_length_months=24,
            monthly_charge=49.99,
            status="Active",
            autopay_enabled=True,
        )
        assert subscription.subscription_id == "SUB-12345678"
        assert subscription.contract_length_months == 24

    def test_invalid_contract_length(self):
        """Test that an invalid contract length fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Subscription(
                subscription_id="SUB-12345678",
                customer_id="CUS-10001",
                plan_id="PLAN-1001",
                device_id="DEV-2001",
                promo_id="PROMO-4001",
                subscription_start_date=date(2020, 2, 1),
                contract_length_months=13,  # Invalid length (not 0, 12, 24, or 36)
                monthly_charge=49.99,
                status="Active",
                autopay_enabled=True,
            )
        assert "Contract length must be one of" in str(exc_info.value)


class TestProductSchemas:
    """Tests for plan, device, and promotion schemas."""

    def test_valid_plan(self):
        """Test that a valid plan passes validation."""
        plan = Plan(
            plan_id="PLAN-1001",
            plan_name="Premium",
            plan_type="Individual",
            monthly_price=49.99,
            data_limit_gb=0,  # Unlimited
            unlimited_calls=True,
            unlimited_texts=True,
            contract_required=True,
            description="Our premium unlimited plan",
            is_active=True,
        )
        assert plan.plan_id == "PLAN-1001"
        assert plan.data_limit_gb == 0  # Unlimited

    def test_invalid_plan_type(self):
        """Test that an invalid plan type fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Plan(
                plan_id="PLAN-1001",
                plan_name="Premium",
                plan_type="Personal",  # Invalid type
                monthly_price=49.99,
                data_limit_gb=0,
                unlimited_calls=True,
                unlimited_texts=True,
                contract_required=True,
                description="Our premium unlimited plan",
                is_active=True,
            )
        assert "Plan type must be one of" in str(exc_info.value)

    def test_valid_device(self):
        """Test that a valid device passes validation."""
        device = Device(
            device_id="DEV-2001",
            device_name="iPhone 15",
            manufacturer="Apple",
            device_type="Smartphone",
            retail_price=999.99,
            monthly_installment=41.67,
            storage_gb=128,
            color_options="Black, White, Blue",
            release_date=date(2023, 9, 15),
            is_5g_compatible=True,
            is_active=True,
        )
        assert device.device_id == "DEV-2001"
        assert device.device_type == "Smartphone"
        assert device.color_options == "Black, White, Blue"

    def test_invalid_device_type(self):
        """Test that an invalid device type fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Device(
                device_id="DEV-2001",
                device_name="iPhone 15",
                manufacturer="Apple",
                device_type="Laptop",  # Invalid type
                retail_price=999.99,
                monthly_installment=41.67,
                storage_gb=128,
                color_options="Black, White, Blue",
                release_date=date(2023, 9, 15),
                is_5g_compatible=True,
                is_active=True,
            )
        assert "Device type must be one of" in str(exc_info.value)

    def test_valid_promotion(self):
        """Test that a valid promotion passes validation."""
        today, future = get_test_dates()
        promotion = Promotion(
            promo_id="PROMO-4001",
            promo_name="Summer Sale",
            discount_type="Percentage",
            discount_value=20.0,
            start_date=today,
            end_date=future,
            description="Get 20% off on all plans this summer",
            is_active=True,
        )
        assert promotion.promo_id == "PROMO-4001"
        assert promotion.discount_type == "Percentage"

    def test_promotion_date_validation(self):
        """Test that end date must be after start date."""
        today, future = get_test_dates()

        # should fail because end_date is before start_date
        with pytest.raises(ValidationError) as exc_info:
            Promotion(
                promo_id="PROMO-4001",
                promo_name="Summer Sale",
                discount_type="Percentage",
                discount_value=20.0,
                start_date=future,  # Later date
                end_date=today,  # Earlier date
                description="Get 20% off on all plans this summer",
                is_active=True,
            )
        assert "End date must be after start date" in str(exc_info.value)


class TestBillingSchemas:
    """Tests for billing and usage schemas."""

    def test_valid_billing(self):
        """Test that a valid billing record passes validation."""
        today, future = get_test_dates()
        billing = Billing(
            billing_id="BILL-1234567890",
            subscription_id="SUB-12345678",
            customer_id="CUS-10001",
            billing_date=today,
            due_date=future,
            base_amount=49.99,
            additional_charges=10.00,
            tax_amount=6.00,
            total_amount=65.99,
            payment_amount=65.99,
            payment_date=today,
            payment_method="Credit Card",
            status="Paid",
            billing_cycle="2023-07",
        )
        assert billing.billing_id == "BILL-1234567890"
        assert billing.status == "Paid"

    def test_billing_amount_validation(self):
        """Test that total amount must match sum of components."""
        today, future = get_test_dates()

        # should fail because total_amount doesn't match the sum
        with pytest.raises(ValidationError) as exc_info:
            Billing(
                billing_id="BILL-1234567890",
                subscription_id="SUB-12345678",
                customer_id="CUS-10001",
                billing_date=today,
                due_date=future,
                base_amount=49.99,
                additional_charges=10.00,
                tax_amount=6.00,
                total_amount=100.00,  # Incorrect total
                payment_amount=100.00,
                payment_date=today,
                payment_method="Credit Card",
                status="Paid",
                billing_cycle="2023-07",
            )
        assert "Total amount" in str(exc_info.value)
        assert "should equal the sum" in str(exc_info.value)

    def test_payment_method_required_with_payment(self):
        """Test that payment method is required when payment amount is set."""
        today, future = get_test_dates()

        # should fail because payment_amount is set but payment_method is not
        with pytest.raises(ValidationError) as exc_info:
            Billing(
                billing_id="BILL-1234567890",
                subscription_id="SUB-12345678",
                customer_id="CUS-10001",
                billing_date=today,
                due_date=future,
                base_amount=49.99,
                additional_charges=10.00,
                tax_amount=6.00,
                total_amount=65.99,
                payment_amount=65.99,  # Payment amount is set
                payment_date=today,
                payment_method=None,  # But method is not
                status="Paid",
                billing_cycle="2023-07",
            )
        assert "Payment method must be provided" in str(exc_info.value)

    def test_valid_usage(self):
        """Test that a valid usage record passes validation."""
        usage = Usage(
            usage_id="USG-123456789012",
            subscription_id="SUB-12345678",
            date=date(2023, 7, 5),
            data_usage_mb=1024.5,
            voice_minutes=120.5,
            sms_count=45,
            billing_cycle="2023-07",
        )
        assert usage.usage_id == "USG-123456789012"
        assert usage.data_usage_mb == 1024.5

    def test_invalid_billing_cycle_format(self):
        """Test that invalid billing cycle format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Usage(
                usage_id="USG-123456789012",
                subscription_id="SUB-12345678",
                date=date(2023, 7, 5),
                data_usage_mb=1024.5,
                voice_minutes=120.5,
                sms_count=45,
                billing_cycle="2023/07",  # Invalid format
            )
        # Check for pattern mismatch error
        assert "string_pattern_mismatch" in str(exc_info.value)
        assert "\\d{4}-\\d{2}" in str(exc_info.value)


class TestKnowledgeSchemas:
    """Tests for knowledge base and support ticket schemas."""

    def test_valid_knowledge_base(self):
        """Test that a valid knowledge base article passes validation."""
        kb = KnowledgeBase(
            kb_id="KB-1001",
            content_type="FAQ",
            category="Billing",
            subcategory="Payment Methods",
            title="How do I update my payment method?",
            content="To update your payment method, follow these steps...",
            tags="billing,payment,credit card",
            last_updated=date(2023, 6, 15),
        )
        assert kb.kb_id == "KB-1001"
        assert kb.content_type == "FAQ"

        # check that tags are normalized
        kb = KnowledgeBase(
            kb_id="KB-1001",
            content_type="FAQ",
            category="Billing",
            subcategory="Payment Methods",
            title="How do I update my payment method?",
            content="To update your payment method, follow these steps...",
            tags="billing, payment, credit card",  # Spaces after commas
            last_updated=date(2023, 6, 15),
        )
        assert kb.tags == "billing,payment,credit card"

    def test_invalid_kb_category(self):
        """Test that an invalid knowledge base category fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeBase(
                kb_id="KB-1001",
                content_type="FAQ",
                category="Marketing",  # Invalid category
                subcategory="Payment Methods",
                title="How do I update my payment method?",
                content="To update your payment method, follow these steps...",
                tags="billing,payment,credit card",
                last_updated=date(2023, 6, 15),
            )
        assert "Category must be one of" in str(exc_info.value)

    def test_valid_support_ticket(self):
        """Test that a valid support ticket passes validation."""
        now = datetime.now()
        later = now + timedelta(hours=2)

        ticket = SupportTicket(
            ticket_id="TICK-8001",
            customer_id="CUS-10001",
            subscription_id="SUB-12345678",
            created_date=now,
            status="Resolved",
            category="Billing",
            priority="Medium",
            description="Customer is seeing an unexpected charge on their bill",
            resolution="Explained that the charge was for international roaming",
            resolved_date=later,
            agent_id="AGT-1234",
        )
        assert ticket.ticket_id == "TICK-8001"
        assert ticket.status == "Resolved"

    def test_ticket_requires_resolution_when_resolved(self):
        """Test that resolved tickets require a resolution."""
        now = datetime.now()

        # should fail because status is Resolved but resolution is None
        with pytest.raises(ValidationError) as exc_info:
            SupportTicket(
                ticket_id="TICK-8001",
                customer_id="CUS-10001",
                subscription_id="SUB-12345678",
                created_date=now,
                status="Resolved",  # Resolved status
                category="Billing",
                priority="Medium",
                description="Customer is seeing an unexpected charge on their bill",
                resolution=None,  # But no resolution provided
                resolved_date=now + timedelta(hours=1),
                agent_id="AGT-1234",
            )
        assert "Resolution must be provided for Resolved" in str(exc_info.value)

    def test_ticket_resolved_date_validation(self):
        """Test that resolved_date must be after created_date."""
        now = datetime.now()
        earlier = now - timedelta(hours=2)

        # should fail because resolved_date is before created_date
        with pytest.raises(ValidationError) as exc_info:
            SupportTicket(
                ticket_id="TICK-8001",
                customer_id="CUS-10001",
                subscription_id="SUB-12345678",
                created_date=now,
                status="Resolved",
                category="Billing",
                priority="Medium",
                description="Customer is seeing an unexpected charge on their bill",
                resolution="Explained the charge",
                resolved_date=earlier,  # Earlier than created_date
                agent_id="AGT-1234",
            )
        assert "Resolved date must be after created date" in str(exc_info.value)


class TestCacheSchemas:
    """Tests for cache entry schema."""

    def test_valid_cache_entry(self):
        """Test that a valid cache entry passes validation."""
        now = datetime.now()
        cache_entry = CacheEntry(
            cache_id="123e4567-e89b-12d3-a456-426614174000",
            query="What is my current bill?",
            response="billing",
            agent_type="supervisor",
            customer_segment="routing",
            formatted_content="Query: What is my current bill?",
            hit_count=0,
            last_hit_time=now,
            created_time=now,
        )
        assert cache_entry.cache_id == "123e4567-e89b-12d3-a456-426614174000"
        assert cache_entry.query == "What is my current bill?"
        assert cache_entry.response == "billing"
        assert cache_entry.agent_type == "supervisor"
        assert cache_entry.customer_segment == "routing"
        assert cache_entry.hit_count == 0

    def test_cache_entry_with_hits(self):
        """Test cache entry with multiple hits."""
        now = datetime.now()
        cache_entry = CacheEntry(
            cache_id="123e4567-e89b-12d3-a456-426614174000",
            query="What is my current bill?",
            response="billing",
            agent_type="supervisor",
            customer_segment="routing",
            formatted_content="Query: What is my current bill?",
            hit_count=42,
            last_hit_time=now,
            created_time=now,
        )
        assert cache_entry.hit_count == 42

    def test_cache_entry_default_hit_count(self):
        """Test that hit_count defaults to 0."""
        now = datetime.now()
        cache_entry = CacheEntry(
            cache_id="123e4567-e89b-12d3-a456-426614174000",
            query="What is my current bill?",
            response="billing",
            agent_type="supervisor",
            customer_segment="routing",
            formatted_content="Query: What is my current bill?",
            last_hit_time=now,
            created_time=now,
        )
        assert cache_entry.hit_count == 0

    def test_negative_hit_count_fails(self):
        """Test that negative hit_count fails validation."""
        now = datetime.now()
        with pytest.raises(ValidationError) as exc_info:
            CacheEntry(
                cache_id="123e4567-e89b-12d3-a456-426614174000",
                query="What is my current bill?",
                response="billing",
                agent_type="supervisor",
                customer_segment="routing",
                formatted_content="Query: What is my current bill?",
                hit_count=-1,  # Invalid negative count
                last_hit_time=now,
                created_time=now,
            )
        assert "greater_than_equal" in str(exc_info.value)

    def test_missing_required_fields(self):
        """Test that missing required fields fail validation."""
        now = datetime.now()
        with pytest.raises(ValidationError) as exc_info:
            CacheEntry(
                cache_id="123e4567-e89b-12d3-a456-426614174000",
                query="What is my current bill?",
                # Missing response
                agent_type="supervisor",
                customer_segment="routing",
                formatted_content="Query: What is my current bill?",
                hit_count=0,
                last_hit_time=now,
                created_time=now,
            )
        assert "response" in str(exc_info.value).lower()
        assert "field required" in str(exc_info.value).lower()

    def test_cache_entry_serialization(self):
        """Test that cache entry can be serialized to dict for Spark."""
        now = datetime.now()
        cache_entry = CacheEntry(
            cache_id="123e4567-e89b-12d3-a456-426614174000",
            query="What is my current bill?",
            response="billing",
            agent_type="supervisor",
            customer_segment="routing",
            formatted_content="Query: What is my current bill?",
            hit_count=5,
            last_hit_time=now,
            created_time=now,
        )

        # Test model_dump() for Spark DataFrame creation
        data_dict = cache_entry.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["cache_id"] == "123e4567-e89b-12d3-a456-426614174000"
        assert data_dict["query"] == "What is my current bill?"
        assert data_dict["response"] == "billing"
        assert data_dict["hit_count"] == 5
        assert isinstance(data_dict["last_hit_time"], datetime)
        assert isinstance(data_dict["created_time"], datetime)
