
from threading import RLock

from environment.models import Payment, PaymentStatus


class PaymentLedger:
    """Thread-safe in-memory ledger for simulated payments."""

    def __init__(self):
        self._payments: dict[str, Payment] = {}
        self._lock = RLock()

    def prepare_payment(self, payment: Payment) -> Payment:
        """Store a new payment in PREPARED state."""

        with self._lock:
            if payment.payment_id in self._payments:
                raise ValueError(
                    f"Payment already exists: {payment.payment_id}"
                )

            if payment.status != PaymentStatus.PREPARED:
                raise ValueError(
                    "New payments must start in PREPARED state"
                )

            self._payments[payment.payment_id] = payment.model_copy(
                deep=True
            )

            return self._payments[payment.payment_id].model_copy(
                deep=True
            )

    def get_payment(self, payment_id: str) -> Payment:
        """Retrieve a payment by ID."""

        with self._lock:
            payment = self._payments.get(payment_id)

            if payment is None:
                raise KeyError(f"Payment not found: {payment_id}")

            return payment.model_copy(deep=True)

    def update_payment(self, payment: Payment) -> Payment:
        """Update an existing payment record."""

        with self._lock:
            if payment.payment_id not in self._payments:
                raise KeyError(
                    f"Payment not found: {payment.payment_id}"
                )

            self._payments[payment.payment_id] = payment.model_copy(
                deep=True
            )

            return self._payments[payment.payment_id].model_copy(
                deep=True
            )

    def list_payments(self) -> list[Payment]:
        """Return copies of all payment records."""

        with self._lock:
            return [
                payment.model_copy(deep=True)
                for payment in self._payments.values()
            ]

    def count(self) -> int:
        """Return the number of stored payments."""

        with self._lock:
            return len(self._payments)
