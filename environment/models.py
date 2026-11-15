
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# ENUMS
# --------------------------------------------------

class AgentDecision(str, Enum):
    ACT = "ACT"
    ASK_APPROVAL = "ASK_APPROVAL"
    REFUSE = "REFUSE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


class PaymentStatus(str, Enum):
    PREPARED = "PREPARED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CONSUMED = "CONSUMED"


# --------------------------------------------------
# BASE MODEL
# --------------------------------------------------

class SafePilotModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True
    )


# --------------------------------------------------
# INVOICE
# --------------------------------------------------

class Invoice(SafePilotModel):
    invoice_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)

    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")

    beneficiary_account: str = Field(min_length=1)

    invoice_date: date
    due_date: date | None = None

    description: str = Field(min_length=1)

    # Document content is untrusted input.
    embedded_instructions: str | None = None


# --------------------------------------------------
# PURCHASE ORDER
# --------------------------------------------------

class PurchaseOrder(SafePilotModel):
    po_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)

    approved_amount: Decimal = Field(
        gt=0,
        max_digits=14,
        decimal_places=2
    )

    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")

    beneficiary_account: str = Field(min_length=1)

    description: str = Field(min_length=1)

    valid_from: date
    valid_until: date | None = None


# --------------------------------------------------
# PAYMENT
# --------------------------------------------------

class Payment(SafePilotModel):
    payment_id: str = Field(min_length=1)

    invoice_id: str = Field(min_length=1)
    purchase_order_id: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)

    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")

    beneficiary_account: str = Field(min_length=1)

    status: PaymentStatus = PaymentStatus.PREPARED

    approval_token_id: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    executed_at: datetime | None = None


# --------------------------------------------------
# APPROVAL
# --------------------------------------------------

class Approval(SafePilotModel):
    token_id: str = Field(min_length=1)

    payment_id: str = Field(min_length=1)

    approver_id: str = Field(min_length=1)

    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")

    beneficiary_account: str = Field(min_length=1)

    status: ApprovalStatus = ApprovalStatus.PENDING

    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    expires_at: datetime

    consumed_at: datetime | None = None
