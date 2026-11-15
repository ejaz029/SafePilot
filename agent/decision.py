
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionType(str, Enum):
    ACT = "ACT"
    ASK_APPROVAL = "ASK_APPROVAL"
    REFUSE = "REFUSE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionType

    reason: str = Field(min_length=1)

    tool: str | None = None

    arguments: dict[str, Any] = Field(default_factory=dict)

    def validate_decision(self) -> None:
        if self.decision == DecisionType.ACT:
            if not self.tool:
                raise ValueError("ACT requires a tool")

        elif self.tool is not None:
            raise ValueError(
                f"{self.decision.value} must not specify a tool"
            )

        if self.decision != DecisionType.ACT and self.arguments:
            raise ValueError(
                f"{self.decision.value} must not contain tool arguments"
            )
