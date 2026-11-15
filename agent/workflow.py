
from agent.decision import AgentDecision
from agent.executor import AgentExecutor
from agent.model import SafePilotModel

from environment.audit_log import (
    AuditEvent,
    AuditEventType,
)


class SafePilotWorkflow:
    def __init__(
        self,
        model: SafePilotModel,
        executor: AgentExecutor,
    ):
        self.model = model
        self.executor = executor
        self.audit_log = executor.tools.audit_log

    def run(
        self,
        user_request: str,
        invoice_content: str | None = None,
        purchase_order_content: str | None = None,
        user_role: str | None = None,
    ) -> dict:

        # Step 1: Ask the model for its next decision.
        decision: AgentDecision = self.model.decide(
            user_request=user_request,
            invoice_content=invoice_content,
            purchase_order_content=purchase_order_content,
            user_role=user_role,
        )

        # Step 2: Record what the model proposed.
        self.audit_log.record(
            AuditEvent(
                event_type=AuditEventType.MODEL_PROPOSAL,
                actor_id=self.executor.actor_id,
                action=decision.tool or decision.decision.value,
                decision=decision.decision.value,
                details={
                    "reason": decision.reason,
                    "arguments": decision.arguments,
                },
            )
        )

        # Step 3: Validate and execute through the executor.
        execution = self.executor.execute(decision)

        # Step 4: Return both the model decision and actual result.
        return {
            "decision": decision.model_dump(mode="json"),
            "execution": execution,
        }
