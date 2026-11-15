
import pytest

from authorization.policies import (
    UserRole,
    UserIdentity,
    AuthorizationPolicy,
)


@pytest.mark.parametrize(
    "role, expected",
    [
        (UserRole.EMPLOYEE, True),
        (UserRole.FINANCE_MANAGER, True),
        (UserRole.FINANCE_ADMIN, True),
        (UserRole.SYSTEM_ADMIN, False),
    ],
)
def test_prepare_payment_permissions(role, expected):
    user = UserIdentity(user_id="USER-101", role=role)

    assert AuthorizationPolicy.can_prepare_payment(user) is expected


@pytest.mark.parametrize(
    "role, expected",
    [
        (UserRole.EMPLOYEE, False),
        (UserRole.FINANCE_MANAGER, True),
        (UserRole.FINANCE_ADMIN, True),
        (UserRole.SYSTEM_ADMIN, False),
    ],
)
def test_approve_payment_permissions(role, expected):
    user = UserIdentity(user_id="USER-101", role=role)

    assert AuthorizationPolicy.can_approve_payment(user) is expected


@pytest.mark.parametrize(
    "role, expected",
    [
        (UserRole.EMPLOYEE, False),
        (UserRole.FINANCE_MANAGER, False),
        (UserRole.FINANCE_ADMIN, True),
        (UserRole.SYSTEM_ADMIN, False),
    ],
)
def test_execute_payment_permissions(role, expected):
    user = UserIdentity(user_id="USER-101", role=role)

    assert AuthorizationPolicy.can_execute_payment(user) is expected


@pytest.mark.parametrize(
    "permission",
    [
        AuthorizationPolicy.can_prepare_payment,
        AuthorizationPolicy.can_approve_payment,
        AuthorizationPolicy.can_execute_payment,
    ],
)
def test_inactive_user_has_no_permissions(permission):
    user = UserIdentity(
        user_id="USER-101",
        role=UserRole.FINANCE_ADMIN,
        active=False,
    )

    assert permission(user) is False
