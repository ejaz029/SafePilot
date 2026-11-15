
from datetime import datetime, timedelta, timezone

from environment.business_store import BusinessStore
from environment.payment_ledger import PaymentLedger
from environment.approval_manager import ApprovalManager
from environment.verification import verify_invoice
from environment.audit_log import AuditLog, AuditEvent, AuditEventType
from environment.models import Payment, PaymentStatus

from authorization.enforcement import AuthorizationEnforcer
from authorization.identity_store import IdentityStore
from authorization.guard import AuthorizationGuard


class SafePilotTools:
    def __init__(
        self,
        business_store: BusinessStore,
        identity_store: IdentityStore,
        payment_ledger: PaymentLedger | None = None,
        approval_manager: ApprovalManager | None = None,
        audit_log: AuditLog | None = None,
    ):
        self.business_store = business_store
        self.identity_store = identity_store
        self.authorization_guard = AuthorizationGuard(identity_store)

        self.payment_ledger = payment_ledger or PaymentLedger()
        self.approval_manager = approval_manager or ApprovalManager()
        self.audit_log = audit_log or AuditLog()

        self.authorization_enforcer = AuthorizationEnforcer(
            self.payment_ledger,
            self.approval_manager,
        )

    def _record_event(
        self,
        event_type: AuditEventType,
        actor_id: str,
        action: str,
        decision: str,
        resource_id: str | None = None,
        details: dict | None = None,
    ):
        """Write a structured event to the audit log."""
        self.audit_log.record(
            AuditEvent(
                event_type=event_type,
                actor_id=actor_id,
                action=action,
                decision=decision,
                resource_id=resource_id,
                details=details or {},
            )
        )

    def _authorize(
        self,
        user_id: str,
        permission: str,
        action: str,
        resource_id: str | None = None,
    ):
        """Check a user's permission and record the policy decision."""
        try:
            user = self.authorization_guard.get_authorized_user(
                user_id,
                permission,
            )

        except (PermissionError, KeyError, ValueError) as exc:
            self._record_event(
                event_type=AuditEventType.POLICY_DECISION,
                actor_id=user_id,
                action=action,
                decision="DENY",
                resource_id=resource_id,
                details={"reason": str(exc), "permission": permission},
            )
            raise

        self._record_event(
            event_type=AuditEventType.POLICY_DECISION,
            actor_id=user_id,
            action=action,
            decision="ALLOW",
            resource_id=resource_id,
            details={"permission": permission, "role": user.role.value},
        )

        return user

    def read_invoice(self, invoice_id: str) -> dict:
        """Retrieve invoice details without modifying stored data."""
        invoice = self.business_store.get_invoice(invoice_id)

        return {
            "invoice_id": invoice.invoice_id,
            "vendor_id": invoice.vendor_id,
            "vendor_name": invoice.vendor_name,
            "amount": str(invoice.amount),
            "currency": invoice.currency,
            "beneficiary_account": invoice.beneficiary_account,
            "invoice_date": invoice.invoice_date.isoformat(),
            "description": invoice.description,
        }

    def lookup_purchase_order(self, po_id: str) -> dict:
        """Retrieve purchase order details without modifying stored data."""
        purchase_order = self.business_store.get_purchase_order(po_id)

        return {
            "po_id": purchase_order.po_id,
            "vendor_id": purchase_order.vendor_id,
            "vendor_name": purchase_order.vendor_name,
            "approved_amount": str(purchase_order.approved_amount),
            "currency": purchase_order.currency,
            "beneficiary_account": purchase_order.beneficiary_account,
            "description": purchase_order.description,
            "valid_from": purchase_order.valid_from.isoformat(),
            "valid_until": (
                purchase_order.valid_until.isoformat()
                if purchase_order.valid_until
                else None
            ),
        }

    def prepare_payment(
        self,
        invoice_id: str,
        po_id: str,
        requester_id: str,
    ) -> dict:
        """Verify and prepare a payment for an authorized requester."""
        payment_id = f"PAY-{invoice_id}"

        self._authorize(
            requester_id,
            "prepare_payment",
            "prepare_payment",
            payment_id,
        )

        invoice = self.business_store.get_invoice(invoice_id)
        purchase_order = self.business_store.get_purchase_order(po_id)

        verification = verify_invoice(invoice, purchase_order)

        if not verification.is_valid:
            self._record_event(
                event_type=AuditEventType.TOOL_EXECUTION,
                actor_id=requester_id,
                action="prepare_payment",
                decision="REJECTED",
                resource_id=payment_id,
                details={
                    "reason": "Invoice verification failed",
                    "verification": verification.model_dump(mode="json"),
                },
            )

            return {
                "success": False,
                "status": "REJECTED",
                "message": "Invoice verification failed.",
                "verification": verification.model_dump(mode="json"),
            }

        payment = Payment(
            payment_id=payment_id,
            invoice_id=invoice.invoice_id,
            purchase_order_id=purchase_order.po_id,
            vendor_id=invoice.vendor_id,
            amount=invoice.amount,
            currency=invoice.currency,
            beneficiary_account=invoice.beneficiary_account,
            status=PaymentStatus.PREPARED,
        )

        self.payment_ledger.prepare_payment(payment)

        self._record_event(
            event_type=AuditEventType.TOOL_EXECUTION,
            actor_id=requester_id,
            action="prepare_payment",
            decision="PREPARED",
            resource_id=payment.payment_id,
            details={
                "amount": str(payment.amount),
                "currency": payment.currency,
                "beneficiary_account": payment.beneficiary_account,
            },
        )

        return {
            "success": True,
            "status": payment.status.value,
            "message": "Payment prepared. Authorized approval is still required.",
            "payment_id": payment.payment_id,
            "amount": str(payment.amount),
            "currency": payment.currency,
            "beneficiary_account": payment.beneficiary_account,
        }

    def request_approval(
        self,
        payment_id: str,
        approver_id: str,
        requester_id: str,
    ) -> dict:
        """Request approval after checking both registered identities."""
        self._authorize(
            requester_id,
            "prepare_payment",
            "request_approval",
            payment_id,
        )

        self._authorize(
            approver_id,
            "approve_payment",
            "request_approval",
            payment_id,
        )

        if approver_id == requester_id:
            self._record_event(
                event_type=AuditEventType.POLICY_DECISION,
                actor_id=requester_id,
                action="request_approval",
                decision="DENY",
                resource_id=payment_id,
                details={"reason": "Self-approval is prohibited"},
            )

            return {
                "success": False,
                "status": "REJECTED",
                "message": "Requester cannot approve their own payment.",
            }

        payment = self.payment_ledger.get_payment(payment_id)

        if payment.status != PaymentStatus.PREPARED:
            return {
                "success": False,
                "status": "REJECTED",
                "message": "Only prepared payments can be submitted for approval.",
            }

        approval = self.approval_manager.request_approval(
            payment=payment,
            approver_id=approver_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

        self._record_event(
            event_type=AuditEventType.TOOL_EXECUTION,
            actor_id=requester_id,
            action="request_approval",
            decision="PENDING",
            resource_id=payment_id,
            details={
                "approver_id": approver_id,
                "approval_token_id": approval.token_id,
            },
        )

        return {
            "success": True,
            "status": approval.status.value,
            "message": "Approval requested. Payment has not been executed.",
            "payment_id": payment.payment_id,
            "approval_token_id": approval.token_id,
            "approver_id": approval.approver_id,
            "expires_at": approval.expires_at.isoformat(),
        }

    def approve_payment(
        self,
        approval_token_id: str,
        approver_id: str,
    ) -> dict:
        """Approve a pending request using a registered finance approver."""
        approval = self.approval_manager.get_approval(approval_token_id)

        self._authorize(
            approver_id,
            "approve_payment",
            "approve_payment",
            approval.payment_id,
        )

        try:
            approved = self.approval_manager.approve(
                token_id=approval_token_id,
                approver_id=approver_id,
            )

        except (PermissionError, ValueError, KeyError) as exc:
            self._record_event(
                event_type=AuditEventType.HUMAN_APPROVAL,
                actor_id=approver_id,
                action="approve_payment",
                decision="REJECTED",
                resource_id=approval.payment_id,
                details={
                    "approval_token_id": approval_token_id,
                    "reason": str(exc),
                },
            )
            raise

        self._record_event(
            event_type=AuditEventType.HUMAN_APPROVAL,
            actor_id=approver_id,
            action="approve_payment",
            decision="APPROVED",
            resource_id=approved.payment_id,
            details={"approval_token_id": approved.token_id},
        )

        return {
            "success": True,
            "status": approved.status.value,
            "payment_id": approved.payment_id,
            "approval_token_id": approved.token_id,
            "message": "Payment approval recorded.",
        }

    def execute_payment(
        self,
        payment_id: str,
        approval_token_id: str,
        executor_id: str,
    ) -> dict:
        """Execute payment after role and transaction-bound approval checks."""
        self._authorize(
            executor_id,
            "execute_payment",
            "execute_payment",
            payment_id,
        )

        if not approval_token_id:
            self._record_event(
                event_type=AuditEventType.TOOL_EXECUTION,
                actor_id=executor_id,
                action="execute_payment",
                decision="BLOCKED",
                resource_id=payment_id,
                details={"reason": "Missing approval token"},
            )

            return {
                "success": False,
                "status": "BLOCKED",
                "message": "An approved authorization token is required.",
            }

        try:
            executed_payment = self.authorization_enforcer.execute_payment(
                payment_id=payment_id,
                approval_token_id=approval_token_id,
            )

        except (PermissionError, ValueError, KeyError) as exc:
            self._record_event(
                event_type=AuditEventType.TOOL_EXECUTION,
                actor_id=executor_id,
                action="execute_payment",
                decision="BLOCKED",
                resource_id=payment_id,
                details={"reason": str(exc)},
            )

            return {
                "success": False,
                "status": "BLOCKED",
                "message": str(exc),
            }

        self._record_event(
            event_type=AuditEventType.TOOL_EXECUTION,
            actor_id=executor_id,
            action="execute_payment",
            decision="EXECUTED",
            resource_id=payment_id,
            details={
                "approval_token_id": approval_token_id,
                "amount": str(executed_payment.amount),
                "currency": executed_payment.currency,
            },
        )

        return {
            "success": True,
            "status": executed_payment.status.value,
            "message": "Payment successfully executed.",
            "payment_id": executed_payment.payment_id,
            "amount": str(executed_payment.amount),
            "currency": executed_payment.currency,
        }
