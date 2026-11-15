
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from environment.approval_manager import ApprovalManager
from environment.models import (
    ApprovalStatus,
    Payment,
)


@pytest.fixture
def manager():
    return ApprovalManager()


@pytest.fixture
def payment():
    return Payment(
        payment_id="PAY-001",
        invoice_id="INV-1042",
        purchase_order_id="PO-1001",
        vendor_id="VENDOR-001",
        amount=Decimal("25000"),
        currency="INR",
        beneficiary_account="ACCOUNT-A",
    )


@pytest.fixture
def approval(manager, payment):
    return manager.request_approval(
        payment=payment,
        approver_id="APPROVER-001",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_request_approval(manager, payment):
    result = manager.request_approval(
        payment,
        "APPROVER-001",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert result.status == ApprovalStatus.PENDING
    assert result.payment_id == payment.payment_id


def test_approve_request(manager, approval):
    result = manager.approve(
        approval.token_id,
        "APPROVER-001",
    )

    assert result.status == ApprovalStatus.APPROVED


def test_wrong_approver_rejected(manager, approval):
    with pytest.raises(PermissionError, match="identity mismatch"):
        manager.approve(approval.token_id, "WRONG-USER")


def test_reject_request(manager, approval):
    result = manager.reject(
        approval.token_id,
        "APPROVER-001",
    )

    assert result.status == ApprovalStatus.REJECTED


def test_expired_approval_rejected(manager, payment):
    approval = manager.request_approval(
        payment,
        "APPROVER-001",
        datetime.now(timezone.utc) + timedelta(seconds=1),
    )

    import time
    time.sleep(1.1)

    with pytest.raises(ValueError, match="expired"):
        manager.approve(approval.token_id, "APPROVER-001")


def test_approval_cannot_be_reused(manager, approval, payment):
    manager.approve(approval.token_id, "APPROVER-001")

    manager.consume(approval.token_id, payment)

    with pytest.raises(PermissionError, match="not approved"):
        manager.consume(approval.token_id, payment)


def test_payment_amount_mismatch(manager, approval, payment):
    manager.approve(approval.token_id, "APPROVER-001")

    altered_payment = payment.model_copy(
        update={"amount": Decimal("50000")}
    )

    with pytest.raises(PermissionError, match="amount mismatch"):
        manager.consume(approval.token_id, altered_payment)


def test_beneficiary_mismatch(manager, approval, payment):
    manager.approve(approval.token_id, "APPROVER-001")

    altered_payment = payment.model_copy(
        update={"beneficiary_account": "ACCOUNT-B"}
    )

    with pytest.raises(PermissionError, match="Beneficiary mismatch"):
        manager.consume(approval.token_id, altered_payment)


def test_approval_consumed_successfully(manager, approval, payment):
    manager.approve(approval.token_id, "APPROVER-001")

    result = manager.consume(approval.token_id, payment)

    assert result.status == ApprovalStatus.CONSUMED
    assert result.consumed_at is not None


def test_expired_timestamp_rejected(manager, payment):
    with pytest.raises(ValueError, match="future"):
        manager.request_approval(
            payment,
            "APPROVER-001",
            datetime.now(timezone.utc) - timedelta(minutes=1),
        )
