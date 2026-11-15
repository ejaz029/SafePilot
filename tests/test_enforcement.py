
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from authorization.enforcement import AuthorizationEnforcer
from environment.approval_manager import ApprovalManager
from environment.models import Payment, PaymentStatus
from environment.payment_ledger import PaymentLedger


@pytest.fixture
def setup():
    ledger = PaymentLedger()
    manager = ApprovalManager()
    enforcer = AuthorizationEnforcer(ledger, manager)

    payment = Payment(
        payment_id="PAY-001",
        invoice_id="INV-1042",
        purchase_order_id="PO-1001",
        vendor_id="VENDOR-001",
        amount=Decimal("25000"),
        currency="INR",
        beneficiary_account="ACCOUNT-A",
    )

    ledger.prepare_payment(payment)

    return ledger, manager, enforcer, payment


def create_approval(manager, payment):
    return manager.request_approval(
        payment,
        "APPROVER-001",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_unapproved_payment_rejected(setup):
    _, _, enforcer, payment = setup

    with pytest.raises(KeyError):
        enforcer.execute_payment(payment.payment_id, "INVALID-TOKEN")


def test_pending_approval_cannot_execute(setup):
    _, manager, enforcer, payment = setup

    approval = create_approval(manager, payment)

    with pytest.raises(PermissionError, match="not approved"):
        enforcer.execute_payment(
            payment.payment_id,
            approval.token_id,
        )


def test_valid_approval_executes_payment(setup):
    ledger, manager, enforcer, payment = setup

    approval = create_approval(manager, payment)
    manager.approve(approval.token_id, "APPROVER-001")

    result = enforcer.execute_payment(
        payment.payment_id,
        approval.token_id,
    )

    assert result.status == PaymentStatus.EXECUTED
    assert result.approval_token_id == approval.token_id
    assert result.executed_at is not None


def test_duplicate_execution_rejected(setup):
    _, manager, enforcer, payment = setup

    approval = create_approval(manager, payment)
    manager.approve(approval.token_id, "APPROVER-001")

    enforcer.execute_payment(payment.payment_id, approval.token_id)

    with pytest.raises(PermissionError, match="cannot execute"):
        enforcer.execute_payment(payment.payment_id, approval.token_id)


def test_approval_for_another_payment_rejected(setup):
    _, manager, enforcer, payment = setup

    approval = create_approval(manager, payment)
    manager.approve(approval.token_id, "APPROVER-001")

    another_payment = payment.model_copy(
        update={"payment_id": "PAY-002"}
    )

    with pytest.raises(PermissionError, match="Payment ID mismatch"):
        manager.consume(approval.token_id, another_payment)
