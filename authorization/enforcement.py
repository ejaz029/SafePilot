
from datetime import datetime, timezone
from threading import RLock

from environment.approval_manager import ApprovalManager
from environment.models import Payment, PaymentStatus
from environment.payment_ledger import PaymentLedger


class AuthorizationEnforcer:
    """Controls payment execution using independent approval checks."""

    def __init__(
        self,
        ledger: PaymentLedger,
        approval_manager: ApprovalManager,
    ):
        self.ledger = ledger
        self.approval_manager = approval_manager
        self._lock = RLock()

    def execute_payment(
        self,
        payment_id: str,
        approval_token_id: str,
    ) -> Payment:
        """Execute a simulated payment after validating approval."""

        with self._lock:
            # 1. Retrieve the existing transaction.
            payment = self.ledger.get_payment(payment_id)

            # 2. Prevent duplicate execution or invalid state transitions.
            if payment.status != PaymentStatus.PREPARED:
                raise PermissionError(
                    f"Payment cannot execute from status: {payment.status}"
                )

            # 3. Validate and consume approval.
            self.approval_manager.consume(
                token_id=approval_token_id,
                payment=payment,
            )

            # 4. Mark payment as executed.
            executed_payment = payment.model_copy(
                update={
                    "status": PaymentStatus.EXECUTED,
                    "approval_token_id": approval_token_id,
                    "executed_at": datetime.now(timezone.utc),
                }
            )

            # 5. Persist the updated simulated transaction.
            return self.ledger.update_payment(executed_payment)
