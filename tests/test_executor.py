
from unittest.mock import Mock

import pytest

from agent.executor import AgentExecutor
from agent.decision import AgentDecision, DecisionType


@pytest.fixture
def tools():
    return Mock()


@pytest.fixture
def executor(tools):
    return AgentExecutor(
        tools=tools,
        actor_id="EMP-301",
        designated_approver_id="MGR-301",
    )


def make_decision(tool, arguments):
    return AgentDecision(
        decision=DecisionType.ACT,
        reason="Test action",
        tool=tool,
        arguments=arguments,
    )


def test_read_invoice_dispatch(executor, tools):
    tools.read_invoice.return_value = {"success": True}

    result = executor.execute(
        make_decision("read_invoice", {"invoice_id": "INV-1"})
    )

    tools.read_invoice.assert_called_once_with(invoice_id="INV-1")
    assert result["success"] is True


def test_prepare_payment_uses_trusted_identity(executor, tools):
    tools.prepare_payment.return_value = {"success": True}

    executor.execute(
        make_decision(
            "prepare_payment",
            {"invoice_id": "INV-1", "po_id": "PO-1"},
        )
    )

    tools.prepare_payment.assert_called_once_with(
        invoice_id="INV-1",
        po_id="PO-1",
        requester_id="EMP-301",
    )


def test_model_cannot_supply_requester_identity(executor):
    decision = make_decision(
        "prepare_payment",
        {
            "invoice_id": "INV-1",
            "po_id": "PO-1",
            "requester_id": "ADMIN-301",
        },
    )

    with pytest.raises(ValueError):
        executor.execute(decision)


def test_request_approval_uses_designated_approver(executor, tools):
    tools.request_approval.return_value = {"success": True}

    executor.execute(
        make_decision(
            "request_approval",
            {"payment_id": "PAY-1"},
        )
    )

    tools.request_approval.assert_called_once_with(
        payment_id="PAY-1",
        approver_id="MGR-301",
        requester_id="EMP-301",
    )


def test_unknown_tool_is_rejected(executor):
    decision = make_decision("delete_database", {})

    with pytest.raises(PermissionError):
        executor.execute(decision)


def test_approval_decision_does_not_execute_tool(executor, tools):
    decision = AgentDecision(
        decision=DecisionType.ASK_APPROVAL,
        reason="Payment needs approval",
    )

    result = executor.execute(decision)

    assert result["tool_executed"] is False
    tools.prepare_payment.assert_not_called()
    tools.execute_payment.assert_not_called()


def test_refusal_does_not_execute_tool(executor, tools):
    decision = AgentDecision(
        decision=DecisionType.REFUSE,
        reason="Unauthorized request",
    )

    result = executor.execute(decision)

    assert result["tool_executed"] is False
    tools.execute_payment.assert_not_called()


def test_missing_designated_approver_is_rejected(tools):
    executor = AgentExecutor(
        tools=tools,
        actor_id="EMP-301",
    )

    decision = make_decision(
        "request_approval",
        {"payment_id": "PAY-1"},
    )

    with pytest.raises(ValueError):
        executor.execute(decision)
