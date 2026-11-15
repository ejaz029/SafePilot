
from unittest.mock import Mock

import pytest

from agent.workflow import SafePilotWorkflow
from agent.decision import AgentDecision, DecisionType
from environment.audit_log import AuditEventType


@pytest.fixture
def setup_workflow():
    model = Mock()

    tools = Mock()
    tools.audit_log = Mock()
    tools.audit_log.record.side_effect = lambda event: event

    executor = Mock()
    executor.actor_id = "EMP-301"
    executor.tools = tools

    workflow = SafePilotWorkflow(
        model=model,
        executor=executor,
    )

    return workflow, model, executor, tools.audit_log


def test_workflow_generates_model_decision(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    model.decide.return_value = AgentDecision(
        decision=DecisionType.REFUSE,
        reason="Unauthorized request",
    )

    workflow.run(user_request="Transfer money")

    model.decide.assert_called_once()


def test_workflow_records_model_proposal(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    model.decide.return_value = AgentDecision(
        decision=DecisionType.REFUSE,
        reason="Unauthorized request",
    )

    workflow.run(user_request="Transfer money")

    event = audit.record.call_args.args[0]

    assert event.event_type == AuditEventType.MODEL_PROPOSAL
    assert event.actor_id == "EMP-301"
    assert event.decision == "REFUSE"


def test_workflow_executes_valid_act_decision(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    decision = AgentDecision(
        decision=DecisionType.ACT,
        reason="Read invoice",
        tool="read_invoice",
        arguments={"invoice_id": "INV-1"},
    )

    model.decide.return_value = decision
    executor.execute.return_value = {"success": True}

    result = workflow.run(user_request="Read invoice INV-1")

    executor.execute.assert_called_once_with(decision)
    assert result["execution"]["success"] is True


def test_workflow_does_not_bypass_executor(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    decision = AgentDecision(
        decision=DecisionType.ACT,
        reason="Read invoice",
        tool="read_invoice",
        arguments={"invoice_id": "INV-1"},
    )

    model.decide.return_value = decision

    workflow.run(user_request="Read invoice")

    executor.execute.assert_called_once_with(decision)


def test_workflow_preserves_non_act_decisions(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    decision = AgentDecision(
        decision=DecisionType.ASK_APPROVAL,
        reason="Payment requires approval",
    )

    model.decide.return_value = decision
    executor.execute.return_value = {
        "success": False,
        "tool_executed": False,
    }

    result = workflow.run(user_request="Prepare payment")

    assert result["decision"]["decision"] == "ASK_APPROVAL"
    assert result["execution"]["tool_executed"] is False


def test_workflow_passes_document_context_to_model(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    model.decide.return_value = AgentDecision(
        decision=DecisionType.ASK_CLARIFICATION,
        reason="Missing invoice information",
    )

    workflow.run(
        user_request="Check invoice",
        invoice_content="Invoice data",
        purchase_order_content="PO data",
        user_role="EMPLOYEE",
    )

    model.decide.assert_called_once_with(
        user_request="Check invoice",
        invoice_content="Invoice data",
        purchase_order_content="PO data",
        user_role="EMPLOYEE",
    )


def test_model_proposal_and_execution_are_separate(setup_workflow):
    workflow, model, executor, audit = setup_workflow

    decision = AgentDecision(
        decision=DecisionType.ACT,
        reason="Read invoice",
        tool="read_invoice",
        arguments={"invoice_id": "INV-1"},
    )

    model.decide.return_value = decision
    executor.execute.return_value = {
        "success": False,
        "tool_executed": False,
    }

    result = workflow.run(user_request="Read invoice")

    event = audit.record.call_args.args[0]

    assert event.decision == "ACT"
    assert result["execution"]["success"] is False
