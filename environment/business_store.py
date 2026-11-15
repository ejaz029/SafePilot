
from threading import RLock

from environment.models import Invoice, PurchaseOrder


class BusinessStore:
    """In-memory repository for invoices and purchase orders."""

    def __init__(self):
        self._invoices: dict[str, Invoice] = {}
        self._purchase_orders: dict[str, PurchaseOrder] = {}
        self._lock = RLock()

    def add_invoice(self, invoice: Invoice) -> Invoice:
        with self._lock:
            if invoice.invoice_id in self._invoices:
                raise ValueError(
                    f"Invoice already exists: {invoice.invoice_id}"
                )

            self._invoices[invoice.invoice_id] = invoice.model_copy(
                deep=True
            )

            return invoice.model_copy(deep=True)

    def get_invoice(self, invoice_id: str) -> Invoice:
        with self._lock:
            invoice = self._invoices.get(invoice_id)

            if invoice is None:
                raise KeyError(f"Invoice not found: {invoice_id}")

            return invoice.model_copy(deep=True)

    def add_purchase_order(
        self,
        purchase_order: PurchaseOrder,
    ) -> PurchaseOrder:
        with self._lock:
            if purchase_order.po_id in self._purchase_orders:
                raise ValueError(
                    f"Purchase order already exists: {purchase_order.po_id}"
                )

            self._purchase_orders[purchase_order.po_id] = (
                purchase_order.model_copy(deep=True)
            )

            return purchase_order.model_copy(deep=True)

    def get_purchase_order(self, po_id: str) -> PurchaseOrder:
        with self._lock:
            purchase_order = self._purchase_orders.get(po_id)

            if purchase_order is None:
                raise KeyError(f"Purchase order not found: {po_id}")

            return purchase_order.model_copy(deep=True)

    def invoice_count(self) -> int:
        with self._lock:
            return len(self._invoices)

    def purchase_order_count(self) -> int:
        with self._lock:
            return len(self._purchase_orders)
