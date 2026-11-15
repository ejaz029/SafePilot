
from datetime import datetime, timezone

import pytest

from environment.audit_log import (
    AuditLog,
    AuditEvent,
    AuditEventType,
)


@pytest.fixture
def audit_log():
    return AuditLog()


def create_event(
    event_type=AuditEventType.MODEL_PROPOSAL,
    actor_id="AGENT-001",
    action="prepare_payment",
    decision="ACT",
    resource_id="PAY-1001",
    details=None,
):
    return AuditEvent(
        event_type=event_type,
        actor_id=actor_id,
        action=action,
        decision=decision,
        resource_id=resource_id,
        details=details or {},
    )


def test_record_event(audit_log):
    event = audit_log.record(create_event())

    assert event.event_type == AuditEventType.MODEL_PROPOSAL
    assert event.actor_id == "AGENT-001"
    assert audit_log.count() == 1


def test_record_generates_unique_event_ids(audit_log):
    first = audit_log.record(create_event())
    second = audit_log.record(create_event())

    assert first.event_id != second.event_id


def test_get_event_by_id(audit_log):
    event = audit_log.record(create_event())

    retrieved = audit_log.get_event(event.event_id)

    assert retrieved.event_id == event.event_id
    assert retrieved.action == "prepare_payment"


def test_get_unknown_event_raises_error(audit_log):
    with pytest.raises(KeyError):
        audit_log.get_event("UNKNOWN-EVENT")


def test_list_all_events(audit_log):
    audit_log.record(create_event())
    audit_log.record(
        create_event(
            event_type=AuditEventType.POLICY_DECISION,
            decision="ALLOW",
        )
    )

    assert len(audit_log.list_events()) == 2


def test_filter_events_by_type(audit_log):
    audit_log.record(create_event())
    audit_log.record(
        create_event(
            event_type=AuditEventType.HUMAN_APPROVAL,
            action="approve_payment",
            decision="APPROVED",
        )
    )

    results = audit_log.list_events(
        event_type=AuditEventType.HUMAN_APPROVAL
    )

    assert len(results) == 1
    assert results[0].decision == "APPROVED"


def test_filter_events_by_resource(audit_log):
    audit_log.record(create_event(resource_id="PAY-1001"))
    audit_log.record(create_event(resource_id="PAY-1002"))

    results = audit_log.list_events(resource_id="PAY-1001")

    assert len(results) == 1
    assert results[0].resource_id == "PAY-1001"


def test_filter_events_by_type_and_resource(audit_log):
    audit_log.record(
        create_event(
            event_type=AuditEventType.MODEL_PROPOSAL,
            resource_id="PAY-1001",
        )
    )

    audit_log.record(
        create_event(
            event_type=AuditEventType.TOOL_EXECUTION,
            resource_id="PAY-1001",
        )
    )

    results = audit_log.list_events(
        event_type=AuditEventType.TOOL_EXECUTION,
        resource_id="PAY-1001",
    )

    assert len(results) == 1
    assert results[0].event_type == AuditEventType.TOOL_EXECUTION


def test_returned_event_cannot_mutate_stored_event(audit_log):
    event = audit_log.record(
        create_event(details={"amount": "25000"})
    )

    event.details["amount"] = "1"

    stored = audit_log.get_event(event.event_id)

    assert stored.details["amount"] == "25000"


def test_event_timestamp_is_timezone_aware(audit_log):
    event = audit_log.record(create_event())

    assert event.timestamp.tzinfo is not None


def test_event_timestamp_can_be_supplied(audit_log):
    timestamp = datetime(2026, 11, 6, tzinfo=timezone.utc)

    event = audit_log.record(
        AuditEvent(
            timestamp=timestamp,
            event_type=AuditEventType.TOOL_EXECUTION,
            actor_id="ADMIN-101",
            action="execute_payment",
            decision="EXECUTED",
        )
    )

    assert event.timestamp == timestamp
