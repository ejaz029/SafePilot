
from datetime import date
from decimal import Decimal

from environment.models import Invoice, PurchaseOrder
from environment.verification import (
    VerificationStatus,
    verify_invoice,
)


def create_invoice(**overrides):
    data = {
        "invoice_id": "INV-1042",
        "vendor_id": "VENDOR-001",
        "vendor_name": "ABC Technologies",
        "amount": Decimal("25000"),
        "currency": "INR",
        "beneficiary_account": "ACCOUNT-A",
        "invoice_date": date(2026, 11, 5),
        "description": "Software services",
    }
    data.update(overrides)
    return Invoice(**data)


def create_purchase_order(**overrides):
    data = {
        "po_id": "PO-1001",
        "vendor_id": "VENDOR-001",
        "vendor_name": "ABC Technologies",
        "approved_amount": Decimal("30000"),
        "currency": "INR",
        "beneficiary_account": "ACCOUNT-A",
        "description": "Software services",
        "valid_from": date(2026, 1, 1),
        "valid_until": date(2026, 12, 31),
    }
    data.update(overrides)
    return PurchaseOrder(**data)


def test_valid_invoice_matches_purchase_order():
    result = verify_invoice(
        create_invoice(),
        create_purchase_order(),
        date(2026, 11, 6),
    )

    assert result.status == VerificationStatus.MATCH
    assert result.is_valid
    assert result.discrepancies == []


def test_vendor_mismatch():
    result = verify_invoice(
        create_invoice(vendor_id="VENDOR-999"),
        create_purchase_order(),
        date(2026, 11, 6),
    )

    assert "VENDOR_MISMATCH" in result.discrepancies


def test_amount_exceeds_approved_amount():
    result = verify_invoice(
        create_invoice(amount=Decimal("35000")),
        create_purchase_order(),
        date(2026, 11, 6),
    )

    assert "AMOUNT_EXCEEDS_APPROVED" in result.discrepancies


def test_currency_mismatch():
    result = verify_invoice(
        create_invoice(currency="USD"),
        create_purchase_order(),
        date(2026, 11, 6),
    )

    assert "CURRENCY_MISMATCH" in result.discrepancies


def test_beneficiary_mismatch():
    result = verify_invoice(
        create_invoice(beneficiary_account="ACCOUNT-B"),
        create_purchase_order(),
        date(2026, 11, 6),
    )

    assert "BENEFICIARY_MISMATCH" in result.discrepancies


def test_purchase_order_not_yet_valid():
    result = verify_invoice(
        create_invoice(),
        create_purchase_order(valid_from=date(2026, 12, 1)),
        date(2026, 11, 6),
    )

    assert "PURCHASE_ORDER_NOT_YET_VALID" in result.discrepancies


def test_purchase_order_expired():
    result = verify_invoice(
        create_invoice(),
        create_purchase_order(valid_until=date(2026, 10, 31)),
        date(2026, 11, 6),
    )

    assert "PURCHASE_ORDER_EXPIRED" in result.discrepancies
