
from decimal import Decimal

import pytest

from environment.models import Payment, PaymentStatus
from environment.payment_ledger import PaymentLedger


@pytest.fixture
def ledger():
    return PaymentLedger()


@pytest.fixture
def sample_payment():
    return Payment(
        payment_id="PAY-001",
        invoice_id="INV-1042",
        purchase_order_id="PO-1001",
        vendor_id="VENDOR-001",
        amount=Decimal("25000"),
        beneficiary_account="ACCOUNT-A",
    )


def test_prepare_payment(ledger, sample_payment):
    payment = ledger.prepare_payment(sample_payment)

    assert payment.payment_id == "PAY-001"
    assert payment.status == PaymentStatus.PREPARED
    assert ledger.count() == 1


def test_duplicate_payment_rejected(ledger, sample_payment):
    ledger.prepare_payment(sample_payment)

    with pytest.raises(ValueError, match="already exists"):
        ledger.prepare_payment(sample_payment)


def test_retrieve_payment(ledger, sample_payment):
    ledger.prepare_payment(sample_payment)

    payment = ledger.get_payment("PAY-001")

    assert payment.amount == Decimal("25000")


def test_missing_payment_rejected(ledger):
    with pytest.raises(KeyError, match="Payment not found"):
        ledger.get_payment("PAY-999")


def test_caller_cannot_modify_stored_payment(
    ledger, sample_payment
):
    ledger.prepare_payment(sample_payment)

    retrieved = ledger.get_payment("PAY-001")
    retrieved.amount = Decimal("999999")

    stored = ledger.get_payment("PAY-001")

    assert stored.amount == Decimal("25000")


def test_update_payment(ledger, sample_payment):
    ledger.prepare_payment(sample_payment)

    updated = sample_payment.model_copy(
        update={"approval_token_id": "TOKEN-001"}
    )

    result = ledger.update_payment(updated)

    assert result.approval_token_id == "TOKEN-001"


def test_list_payments(ledger, sample_payment):
    ledger.prepare_payment(sample_payment)

    payments = ledger.list_payments()

    assert len(payments) == 1
    assert payments[0].payment_id == "PAY-001"
