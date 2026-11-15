
import pytest
from pydantic import ValidationError

from agent.decision import AgentDecision, DecisionType


def test_act_decision():
    decision = AgentDecision(
        decision=DecisionType.ACT,
        reason="Invoice verified",
        tool="read_invoice",
        arguments={"invoice_id": "INV-1001"},
    )

    decision.validate_decision()
    assert decision.decision == DecisionType.ACT


def test_approval_decision():
    decision = AgentDecision(
        decision=DecisionType.ASK_APPROVAL,
        reason="Payment requires approval",
    )

    decision.validate_decision()


def test_refusal_decision():
    decision = AgentDecision(
        decision=DecisionType.REFUSE,
        reason="Unauthorized payment",
    )

    decision.validate_decision()


def test_clarification_decision():
    decision = AgentDecision(
        decision=DecisionType.ASK_CLARIFICATION,
        reason="Invoice identifier is missing",
    )

    decision.validate_decision()


def test_act_requires_tool():
    decision = AgentDecision(
        decision=DecisionType.ACT,
        reason="Read invoice",
    )

    with pytest.raises(ValueError):
        decision.validate_decision()


def test_non_act_cannot_specify_tool():
    decision = AgentDecision(
        decision=DecisionType.REFUSE,
        reason="Not authorized",
        tool="execute_payment",
    )

    with pytest.raises(ValueError):
        decision.validate_decision()


def test_non_act_cannot_contain_arguments():
    decision = AgentDecision(
        decision=DecisionType.ASK_APPROVAL,
        reason="Approval required",
        arguments={"amount": 25000},
    )

    with pytest.raises(ValueError):
        decision.validate_decision()


def test_invalid_decision_is_rejected():
    with pytest.raises(ValidationError):
        AgentDecision(
            decision="EXECUTE_ANYWAY",
            reason="Invalid decision",
        )
