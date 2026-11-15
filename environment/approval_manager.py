
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from environment.models import Approval, ApprovalStatus, Payment


class ApprovalManager:
    """Manages transaction-bound human approval tokens."""

    def __init__(self):
        self._approvals: dict[str, Approval] = {}
        self._lock = RLock()

    def request_approval(
        self,
        payment: Payment,
        approver_id: str,
        expires_at: datetime,
    ) -> Approval:
        if expires_at.tzinfo is None:
            raise ValueError("Expiry timestamp must be timezone-aware")

        if expires_at <= datetime.now(timezone.utc):
            raise ValueError("Expiry timestamp must be in the future")

        approval = Approval(
            token_id=str(uuid4()),
            payment_id=payment.payment_id,
            approver_id=approver_id,
            amount=payment.amount,
            currency=payment.currency,
            beneficiary_account=payment.beneficiary_account,
            expires_at=expires_at,
        )

        with self._lock:
            self._approvals[approval.token_id] = approval.model_copy(
                deep=True
            )

        return approval.model_copy(deep=True)

    def get_approval(self, token_id: str) -> Approval:
        with self._lock:
            approval = self._approvals.get(token_id)

            if approval is None:
                raise KeyError(f"Approval not found: {token_id}")

            return approval.model_copy(deep=True)

    def approve(
        self,
        token_id: str,
        approver_id: str,
    ) -> Approval:
        with self._lock:
            approval = self._approvals.get(token_id)

            if approval is None:
                raise KeyError(f"Approval not found: {token_id}")

            if approval.approver_id != approver_id:
                raise PermissionError("Approver identity mismatch")

            if approval.status != ApprovalStatus.PENDING:
                raise ValueError("Approval is not pending")

            now = datetime.now(timezone.utc)

            if now >= approval.expires_at:
                approval.status = ApprovalStatus.EXPIRED
                raise ValueError("Approval has expired")

            approval.status = ApprovalStatus.APPROVED

            return approval.model_copy(deep=True)

    def reject(
        self,
        token_id: str,
        approver_id: str,
    ) -> Approval:
        with self._lock:
            approval = self._approvals.get(token_id)

            if approval is None:
                raise KeyError(f"Approval not found: {token_id}")

            if approval.approver_id != approver_id:
                raise PermissionError("Approver identity mismatch")

            if approval.status != ApprovalStatus.PENDING:
                raise ValueError("Approval is not pending")

            approval.status = ApprovalStatus.REJECTED

            return approval.model_copy(deep=True)

    def consume(
        self,
        token_id: str,
        payment: Payment,
    ) -> Approval:
        """Validate and atomically consume an approved token."""

        with self._lock:
            approval = self._approvals.get(token_id)

            if approval is None:
                raise KeyError(f"Approval not found: {token_id}")

            if approval.status != ApprovalStatus.APPROVED:
                raise PermissionError("Approval is not approved")

            now = datetime.now(timezone.utc)

            if now >= approval.expires_at:
                approval.status = ApprovalStatus.EXPIRED
                raise PermissionError("Approval has expired")

            if approval.payment_id != payment.payment_id:
                raise PermissionError("Payment ID mismatch")

            if approval.amount != payment.amount:
                raise PermissionError("Payment amount mismatch")

            if approval.currency != payment.currency:
                raise PermissionError("Payment currency mismatch")

            if (
                approval.beneficiary_account
                != payment.beneficiary_account
            ):
                raise PermissionError("Beneficiary mismatch")

            approval.status = ApprovalStatus.CONSUMED
            approval.consumed_at = now

            return approval.model_copy(deep=True)
