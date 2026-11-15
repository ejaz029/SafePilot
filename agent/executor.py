
from agent.decision import AgentDecision, DecisionType
from agent.tools import SafePilotTools


class AgentExecutor:
    """Executes approved tool requests using trusted user identities."""

    TOOL_ARGUMENTS = {
        "read_invoice": {"invoice_id"},
        "lookup_purchase_order": {"po_id"},
        "prepare_payment": {"invoice_id", "po_id"},
        "request_approval": {"payment_id"},
        "approve_payment": {"approval_token_id"},
        "execute_payment": {"payment_id", "approval_token_id"},
    }

    def __init__(
        self,
        tools: SafePilotTools,
        actor_id: str,
        designated_approver_id: str | None = None,
    ):
        self.tools = tools
        self.actor_id = actor_id
        self.designated_approver_id = designated_approver_id

    def execute(self, decision: AgentDecision) -> dict:
        if decision.decision != DecisionType.ACT:
            return {
                "success": False,
                "decision": decision.decision.value,
                "reason": decision.reason,
                "tool_executed": False,
            }

        tool_name = decision.tool

        if tool_name not in self.TOOL_ARGUMENTS:
            raise PermissionError(
                f"Tool is not available: {tool_name}"
            )

        allowed_arguments = self.TOOL_ARGUMENTS[tool_name]
        supplied_arguments = set(decision.arguments.keys())

        if supplied_arguments != allowed_arguments:
            raise ValueError(
                f"Invalid arguments for {tool_name}. "
                f"Expected {sorted(allowed_arguments)}, "
                f"received {sorted(supplied_arguments)}"
            )

        args = decision.arguments

        if tool_name == "read_invoice":
            result = self.tools.read_invoice(
                invoice_id=args["invoice_id"]
            )

        elif tool_name == "lookup_purchase_order":
            result = self.tools.lookup_purchase_order(
                po_id=args["po_id"]
            )

        elif tool_name == "prepare_payment":
            result = self.tools.prepare_payment(
                invoice_id=args["invoice_id"],
                po_id=args["po_id"],
                requester_id=self.actor_id,
            )

        elif tool_name == "request_approval":
            if not self.designated_approver_id:
                raise ValueError(
                    "No designated approver configured"
                )

            result = self.tools.request_approval(
                payment_id=args["payment_id"],
                approver_id=self.designated_approver_id,
                requester_id=self.actor_id,
            )

        elif tool_name == "approve_payment":
            result = self.tools.approve_payment(
                approval_token_id=args["approval_token_id"],
                approver_id=self.actor_id,
            )

        else:
            result = self.tools.execute_payment(
                payment_id=args["payment_id"],
                approval_token_id=args["approval_token_id"],
                executor_id=self.actor_id,
            )

        return {
            "success": bool(result.get("success", False)),
            "decision": decision.decision.value,
            "tool": tool_name,
            "tool_executed": True,
            "result": result,
        }
