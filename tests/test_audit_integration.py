
from datetime import date
from decimal import Decimal

import pytest

from environment.business_store import BusinessStore
from environment.models import Invoice, PurchaseOrder
from environment.audit_log import AuditEventType
from authorization.identity_store import IdentityStore
from authorization.policies import UserIdentity, UserRole
from agent.tools import SafePilotTools


@pytest.fixture
def tools():
    store = BusinessStore()

    store.add_invoice(
        Invoice(
            invoice_id="INV-3001",
            vendor_id="V-301",
            vendor_name="Nova Systems",
            amount=Decimal("25000.00"),
            currency="INR",
            beneficiary_account="ACCT-3001",
            invoice_date=date(2026, 11, 1),
            description="Software services",
        )
    )

    store.add_purchase_order(
        PurchaseOrder(
            po_id="PO-3001",
            vendor_id="V-301",
            vendor_name="Nova Systems",
            approved_amount=Decimal("30000.00"),
            currency="INR",
            beneficiary_account="ACCT-3001",
            description="Approved software services",
            valid_from=date(2026, 1, 1),
            valid_until=date(2026, 12, 31),
        )
    )

    identities = IdentityStore()

    for user in [
        UserIdentity(user_id="EMP-301", role=UserRole.EMPLOYEE),
        UserIdentity(user_id="MGR-301", role=UserRole.FINANCE_MANAGER),
        UserIdentity(user_id="ADMIN-301", role=UserRole.FINANCE_ADMIN),
        UserIdentity(user_id="SYS-301", role=UserRole.SYSTEM_ADMIN),
    ]:
        identities.register_user(user)

    return SafePilotTools(
        business_store=store,
        identity_store=identities,
    )


def prepare_payment(tools):
    return tools.prepare_payment(
        invoice_id="INV-3001",
        po_id="PO-3001",
        requester_id="EMP-301",
    )


def request_approval(tools):
    prepare_payment(tools)
    return tools.request_approval(
        payment_id="PAY-INV-3001",
        approver_id="MGR-301",
        requester_id="EMP-301",
    )


def approve_payment(tools, approval):
    return tools.approve_payment(
        approval_token_id=approval["approval_token_id"],
        approver_id="MGR-301",
    )


def execute_payment(tools, approval):
    return tools.execute_payment(
        payment_id="PAY-INV-3001",
        approval_token_id=approval["approval_token_id"],
        executor_id="ADMIN-301",
    )


def test_preparation_records_policy_and_execution_events(tools):
    result = prepare_payment(tools)
    assert result["success"] is True

    events = tools.audit_log.list_events(
        resource_id="PAY-INV-3001"
    )

    assert any(
        e.event_type == AuditEventType.POLICY_DECISION
        and e.decision == "ALLOW"
        for e in events
    )

    assert any(
        e.event_type == AuditEventType.TOOL_EXECUTION
        and e.decision == "PREPARED"
        for e in events
    )


def test_unauthorized_preparation_records_denial(tools):
    with pytest.raises(PermissionError):
        tools.prepare_payment(
            invoice_id="INV-3001",
            po_id="PO-3001",
            requester_id="SYS-301",
        )

    events = tools.audit_log.list_events(
        event_type=AuditEventType.POLICY_DECISION
    )

    assert any(
        e.actor_id == "SYS-301" and e.decision == "DENY"
        for e in events
    )


def test_failed_invoice_verification_is_audited(tools):
    tools.business_store.add_invoice(
        Invoice(
            invoice_id="INV-3002",
            vendor_id="V-301",
            vendor_name="Nova Systems",
            amount=Decimal("25000.00"),
            currency="INR",
            beneficiary_account="ATTACKER-ACCOUNT",
            invoice_date=date(2026, 11, 1),
            description="Modified invoice",
        )
    )

    tools.business_store.add_purchase_order(
        PurchaseOrder(
            po_id="PO-3002",
            vendor_id="V-301",
            vendor_name="Nova Systems",
            approved_amount=Decimal("30000.00"),
            currency="INR",
            beneficiary_account="ACCT-3001",
            description="Approved software services",
            valid_from=date(2026, 1, 1),
            valid_until=date(2026, 12, 31),
        )
    )

    result = tools.prepare_payment(
        invoice_id="INV-3002",
        po_id="PO-3002",
        requester_id="EMP-301",
    )

    assert result["success"] is False

    events = tools.audit_log.list_events(
        resource_id="PAY-INV-3002"
    )

    assert any(
        e.decision == "REJECTED"
        and e.action == "prepare_payment"
        for e in events
    )


def test_approval_request_is_audited(tools):
    result = request_approval(tools)
    assert result["success"] is True

    events = tools.audit_log.list_events(
        event_type=AuditEventType.TOOL_EXECUTION,
        resource_id="PAY-INV-3001",
    )

    assert any(
        e.action == "request_approval"
        and e.decision == "PENDING"
        for e in events
    )


def test_human_approval_is_audited(tools):
    approval = request_approval(tools)
    result = approve_payment(tools, approval)

    assert result["success"] is True

    events = tools.audit_log.list_events(
        event_type=AuditEventType.HUMAN_APPROVAL,
        resource_id="PAY-INV-3001",
    )

    assert len(events) == 1
    assert events[0].decision == "APPROVED"
    assert events[0].actor_id == "MGR-301"


def test_pending_payment_execution_is_audited_as_blocked(tools):
    approval = request_approval(tools)
    result = execute_payment(tools, approval)

    assert result["success"] is False
    assert result["status"] == "BLOCKED"

    events = tools.audit_log.list_events(
        resource_id="PAY-INV-3001"
    )

    assert any(
        e.action == "execute_payment"
        and e.decision == "BLOCKED"
        for e in events
    )


def test_successful_execution_is_audited(tools):
    approval = request_approval(tools)
    approve_payment(tools, approval)

    result = execute_payment(tools, approval)

    assert result["success"] is True
    assert result["status"] == "EXECUTED"

    events = tools.audit_log.list_events(
        event_type=AuditEventType.TOOL_EXECUTION,
        resource_id="PAY-INV-3001",
    )

    assert any(
        e.action == "execute_payment"
        and e.decision == "EXECUTED"
        for e in events
    )


def test_audit_events_preserve_actor_identity(tools):
    request_approval(tools)

    events = tools.audit_log.list_events(
        resource_id="PAY-INV-3001"
    )

    actors = {e.actor_id for e in events}

    assert "EMP-301" in actors
    assert "MGR-301" in actors


def test_audit_log_preserves_approval_token_reference(tools):
    approval = request_approval(tools)

    events = tools.audit_log.list_events(
        event_type=AuditEventType.TOOL_EXECUTION,
        resource_id="PAY-INV-3001",
    )

    request_events = [
        e for e in events
        if e.action == "request_approval"
    ]

    assert len(request_events) == 1
    assert (
        request_events[0].details["approval_token_id"]
        == approval["approval_token_id"]
    )
