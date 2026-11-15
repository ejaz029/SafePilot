from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DecisionType = Literal[
    "ACT",
    "ASK_APPROVAL",
    "REFUSE",
    "ASK_CLARIFICATION",
]


class ModelPrediction(BaseModel):
    """One model decision for a single AuthorityBench scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    decision: DecisionType
    reason: str = Field(min_length=1)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision(self) -> ModelPrediction:
        if self.decision == "ACT":
            if not self.tool:
                raise ValueError("ACT predictions must specify a tool.")
        else:
            if self.tool is not None:
                raise ValueError(
                    "Non-ACT predictions must have tool=None."
                )
            if self.arguments:
                raise ValueError(
                    "Non-ACT predictions must have empty arguments."
                )

        return self


class EvaluationRecord(BaseModel):
    """Ground truth and model prediction for one benchmark scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    category: str
    expected_decision: DecisionType
    expected_tool: str | None = None
    expected_final_state: str
    workflow_state: dict[str, Any]
    purchase_order: dict[str, Any]
    prediction: ModelPrediction
    actual_tool: str | None = None
    actual_arguments: dict[str, Any] = Field(default_factory=dict)
    actual_final_state: str | None = None

    @model_validator(mode="after")
    def validate_scenario_match(self) -> EvaluationRecord:
        if self.scenario_id != self.prediction.scenario_id:
            raise ValueError(
                "Prediction scenario_id must match evaluation scenario_id."
            )
        return self