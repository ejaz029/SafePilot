
"""Convert AuthorityBench v1 scenarios into SafePilot runtime models.

Missing descriptive metadata is supplied only as documented fixture defaults.
Transaction-critical fields are never silently fabricated.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from environment.models import Invoice, PurchaseOrder


FIXTURE_INVOICE_DATE = date(2026, 1, 1)
FIXTURE_PO_VALID_FROM = date(2000, 1, 1)


class BenchmarkAdapterError(ValueError):
    """Raised when benchmark data cannot safely form a runtime model."""


@dataclass
class BenchmarkRuntimeFixture:
    scenario_id: str
    invoice: Invoice | None
    purchase_order: PurchaseOrder | None
    fixture_metadata: dict[str, str] = field(default_factory=dict)


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _parse_text_fields(content: str) -> dict[str, str]:
    fields = {}

    for line in content.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        fields[_normalise_key(key.strip())] = value.strip()

    return fields


def _get_text_value(fields: dict, *names: str) -> str | None:
    for name in names:
        value = fields.get(_normalise_key(name))
        if value:
            return value
    return None


def _decimal_value(value: Any, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        raise BenchmarkAdapterError(
            f"Invalid or missing {field_name}: {value!r}"
        )

    if not result.is_finite() or result <= 0:
        raise BenchmarkAdapterError(
            f"{field_name} must be a positive finite number."
        )

    return result


def _required(value: Any, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise BenchmarkAdapterError(
            f"Missing required transaction field: {field_name}"
        )
    return str(value).strip()


def _parse_amount_and_currency(
    fields: dict[str, str],
) -> tuple[Decimal, str]:
    amount_text = _get_text_value(fields, "amount", "invoice amount")

    if not amount_text:
        raise BenchmarkAdapterError("Invoice amount is missing.")

    match = re.search(
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z]{3})?",
        amount_text,
    )

    if not match:
        raise BenchmarkAdapterError(
            f"Cannot parse invoice amount: {amount_text!r}"
        )

    amount = _decimal_value(match.group(1), "invoice amount")
    currency = match.group(2)

    if not currency:
        currency = _get_text_value(fields, "currency")

    return amount, (currency or "INR").upper()


def adapt_scenario(scenario: dict) -> BenchmarkRuntimeFixture:
    """Convert one AuthorityBench scenario into runtime models.

    Missing invoice/PO records remain None. Required transaction facts
    must be present in the benchmark and are never invented.
    """

    scenario_id = _required(
        scenario.get("scenario_id"),
        "scenario_id",
    )

    workflow_state = scenario.get("workflow_state") or {}
    invoice_available = workflow_state.get(
        "invoice_available", True
    )
    po_available = workflow_state.get(
        "purchase_order_available", True
    )

    metadata = {}

    invoice = None
    purchase_order = None

    if invoice_available:
        content = scenario.get("invoice_content")

        if not isinstance(content, str) or not content.strip():
            raise BenchmarkAdapterError(
                f"{scenario_id}: invoice is marked available "
                "but invoice_content is empty."
            )

        invoice_fields = _parse_text_fields(content)

        invoice_id = _get_text_value(
            invoice_fields, "invoice_id", "invoice id"
        )

        if not invoice_id:
            invoice_id = f"INV-{scenario_id}"
            metadata["invoice_id"] = (
                "Synthetic ID derived from scenario_id; "
                "not supplied in benchmark."
            )

        vendor_id = _get_text_value(
            invoice_fields, "vendor_id", "vendor id"
        )
        vendor_name = _get_text_value(
            invoice_fields, "vendor", "vendor_name", "vendor name"
        )
        beneficiary = _get_text_value(
            invoice_fields,
            "beneficiary",
            "beneficiary_account",
            "beneficiary account",
        )

        amount, currency = _parse_amount_and_currency(
            invoice_fields
        )

        vendor_id = _required(vendor_id, "invoice vendor_id")
        beneficiary = _required(
            beneficiary, "invoice beneficiary_account"
        )

        if not vendor_name:
            vendor_name = f"Vendor {vendor_id}"
            metadata["invoice_vendor_name"] = (
                "Synthetic descriptive name; not supplied in benchmark."
            )

        description = _get_text_value(
            invoice_fields, "description"
        )

        if not description:
            description = (
                "AuthorityBench fixture; description not supplied."
            )
            metadata["invoice_description"] = (
                "Synthetic descriptive metadata."
            )

        invoice = Invoice(
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            amount=amount,
            currency=currency,
            beneficiary_account=beneficiary,
            invoice_date=FIXTURE_INVOICE_DATE,
            description=description,
            embedded_instructions=_get_text_value(
                invoice_fields,
                "embedded_instructions",
                "instructions",
            ),
        )

        metadata["invoice_date"] = (
            "Fixed fixture date; benchmark v1.0 does not supply it."
        )

    if po_available:
        po_data = scenario.get("purchase_order")

        if not isinstance(po_data, dict) or not po_data:
            raise BenchmarkAdapterError(
                f"{scenario_id}: purchase order is marked available "
                "but purchase_order data is empty."
            )

        po_id = _required(
            po_data.get("po_id"), "purchase_order.po_id"
        )
        vendor_id = _required(
            po_data.get("vendor_id"), "purchase_order.vendor_id"
        )
        beneficiary = _required(
            po_data.get(
                "beneficiary_account",
                po_data.get("beneficiary"),
            ),
            "purchase_order.beneficiary_account",
        )

        raw_amount = po_data.get(
            "approved_amount",
            po_data.get("amount"),
        )
        amount = _decimal_value(
            raw_amount, "purchase_order amount"
        )

        currency = str(po_data.get("currency", "INR")).upper()

        vendor_name = po_data.get("vendor_name")
        if not vendor_name:
            vendor_name = f"Vendor {vendor_id}"
            metadata["purchase_order_vendor_name"] = (
                "Synthetic descriptive name; not supplied in benchmark."
            )

        description = po_data.get("description")
        if not description:
            description = (
                "AuthorityBench fixture; description not supplied."
            )
            metadata["purchase_order_description"] = (
                "Synthetic descriptive metadata."
            )

        purchase_order = PurchaseOrder(
            po_id=po_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            approved_amount=amount,
            currency=currency,
            beneficiary_account=beneficiary,
            description=description,
            valid_from=FIXTURE_PO_VALID_FROM,
            valid_until=None,
        )

        metadata["purchase_order_valid_from"] = (
            "Fixed fixture date; benchmark v1.0 does not supply it."
        )

    return BenchmarkRuntimeFixture(
        scenario_id=scenario_id,
        invoice=invoice,
        purchase_order=purchase_order,
        fixture_metadata=metadata,
    )
