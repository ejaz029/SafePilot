
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AuditEventType(str, Enum):
    MODEL_PROPOSAL = "MODEL_PROPOSAL"
    POLICY_DECISION = "POLICY_DECISION"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    TOOL_EXECUTION = "TOOL_EXECUTION"


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    event_type: AuditEventType
    actor_id: str
    action: str
    decision: str
    resource_id: str | None = None
    details: dict = Field(default_factory=dict)


class AuditLog:
    """Thread-safe, append-only audit event store."""

    def __init__(self):
        self._events: list[AuditEvent] = []
        self._lock = RLock()

    def record(self, event: AuditEvent) -> AuditEvent:
        """Append an audit event."""
        with self._lock:
            stored_event = event.model_copy(deep=True)
            self._events.append(stored_event)
            return stored_event.model_copy(deep=True)

    def get_event(self, event_id: str) -> AuditEvent:
        """Retrieve an event by its ID."""
        with self._lock:
            for event in self._events:
                if event.event_id == event_id:
                    return event.model_copy(deep=True)

        raise KeyError(f"Audit event not found: {event_id}")

    def list_events(
        self,
        event_type: AuditEventType | None = None,
        resource_id: str | None = None,
    ) -> list[AuditEvent]:
        """Return events with optional filters, in recorded order."""
        with self._lock:
            results = self._events

            if event_type is not None:
                results = [
                    event for event in results
                    if event.event_type == event_type
                ]

            if resource_id is not None:
                results = [
                    event for event in results
                    if event.resource_id == resource_id
                ]

            return [event.model_copy(deep=True) for event in results]

    def count(self) -> int:
        with self._lock:
            return len(self._events)
