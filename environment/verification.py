
from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict

from environment.models import Invoice, PurchaseOrder


class VerificationStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    invoice_id: str
    purchase_order_id: str
    discrepancies: list[str]

    @property
    def is_valid(self) -> bool:
        return self.status == VerificationStatus.MATCH


def verify_invoice(
    invoice: Invoice,
    purchase_order: PurchaseOrder,
    verification_date: date | None = None,
) -> VerificationResult:

    check_date = verification_date or date.today()
    discrepancies = []

    if invoice.vendor_id != purchase_order.vendor_id:
        discrepancies.append("VENDOR_MISMATCH")

    if invoice.currency != purchase_order.currency:
        discrepancies.append("CURRENCY_MISMATCH")

    if invoice.amount > purchase_order.approved_amount:
        discrepancies.append("AMOUNT_EXCEEDS_APPROVED")

    if invoice.beneficiary_account != purchase_order.beneficiary_account:
        discrepancies.append("BENEFICIARY_MISMATCH")

    if check_date < purchase_order.valid_from:
        discrepancies.append("PURCHASE_ORDER_NOT_YET_VALID")

    if (
        purchase_order.valid_until is not None
        and check_date > purchase_order.valid_until
    ):
        discrepancies.append("PURCHASE_ORDER_EXPIRED")

    status = (
        VerificationStatus.MATCH
        if not discrepancies
        else VerificationStatus.MISMATCH
    )

    return VerificationResult(
        status=status,
        invoice_id=invoice.invoice_id,
        purchase_order_id=purchase_order.po_id,
        discrepancies=discrepancies,
    )
