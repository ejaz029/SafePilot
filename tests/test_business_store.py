
from datetime import date
from decimal import Decimal

import pytest

from environment.business_store import BusinessStore
from environment.models import Invoice, PurchaseOrder


@pytest.fixture
def store():
    return BusinessStore()


@pytest.fixture
def invoice():
    return Invoice(
        invoice_id="INV-1042",
        vendor_id="VENDOR-001",
        vendor_name="ABC Technologies",
        amount=Decimal("25000"),
        currency="INR",
        beneficiary_account="ACCOUNT-A",
        invoice_date=date(2026, 11, 5),
        description="Software services",
    )


@pytest.fixture
def purchase_order():
    return PurchaseOrder(
        po_id="PO-1001",
        vendor_id="VENDOR-001",
        vendor_name="ABC Technologies",
        approved_amount=Decimal("30000"),
        currency="INR",
        beneficiary_account="ACCOUNT-A",
        description="Software services",
        valid_from=date(2026, 1, 1),
    )


def test_add_and_get_invoice(store, invoice):
    store.add_invoice(invoice)
    result = store.get_invoice("INV-1042")
    assert result.vendor_id == "VENDOR-001"


def test_duplicate_invoice_rejected(store, invoice):
    store.add_invoice(invoice)

    with pytest.raises(ValueError, match="already exists"):
        store.add_invoice(invoice)


def test_missing_invoice_rejected(store):
    with pytest.raises(KeyError, match="Invoice not found"):
        store.get_invoice("INV-9999")


def test_add_and_get_purchase_order(store, purchase_order):
    store.add_purchase_order(purchase_order)
    result = store.get_purchase_order("PO-1001")
    assert result.approved_amount == Decimal("30000")


def test_duplicate_purchase_order_rejected(store, purchase_order):
    store.add_purchase_order(purchase_order)

    with pytest.raises(ValueError, match="already exists"):
        store.add_purchase_order(purchase_order)


def test_missing_purchase_order_rejected(store):
    with pytest.raises(KeyError, match="Purchase order not found"):
        store.get_purchase_order("PO-9999")


def test_store_returns_independent_invoice_copy(store, invoice):
    store.add_invoice(invoice)

    retrieved = store.get_invoice("INV-1042")
    retrieved.amount = Decimal("999999")

    assert store.get_invoice("INV-1042").amount == Decimal("25000")


def test_store_counts(store, invoice, purchase_order):
    store.add_invoice(invoice)
    store.add_purchase_order(purchase_order)

    assert store.invoice_count() == 1
    assert store.purchase_order_count() == 1
