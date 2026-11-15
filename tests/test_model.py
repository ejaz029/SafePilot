
import pytest

from agent.model import SafePilotModel
from agent.decision import DecisionType


@pytest.fixture
def model():
    return SafePilotModel(device="cpu")


def test_extract_valid_json(model):
    result = model._extract_json(
        '{"decision":"REFUSE","reason":"Unauthorized"}'
    )

    assert result["decision"] == "REFUSE"


def test_extract_json_from_markdown(model):
    result = model._extract_json(
        '```json\n{"decision":"ACT","reason":"Verified",'
        '"tool":"read_invoice","arguments":{}}\n```'
    )

    assert result["decision"] == "ACT"


def test_extract_json_with_surrounding_text(model):
    result = model._extract_json(
        'Result:\n{"decision":"REFUSE","reason":"Not allowed"}'
    )

    assert result["decision"] == "REFUSE"


def test_invalid_json_raises_error(model):
    with pytest.raises(ValueError):
        model._extract_json("No JSON response")


def test_decision_output_is_validated(model):
    data = model._extract_json(
        '{"decision":"ASK_APPROVAL",'
        '"reason":"Payment needs approval",'
        '"tool":null,"arguments":{}}'
    )

    from agent.decision import AgentDecision

    decision = AgentDecision.model_validate(data)
    decision.validate_decision()

    assert decision.decision == DecisionType.ASK_APPROVAL


def test_invalid_decision_is_rejected(model):
    data = model._extract_json(
        '{"decision":"BYPASS","reason":"Execute anyway"}'
    )

    from agent.decision import AgentDecision

    with pytest.raises(Exception):
        AgentDecision.model_validate(data)
