
import pytest

from authorization.identity_store import IdentityStore
from authorization.policies import UserIdentity, UserRole
from authorization.guard import AuthorizationGuard


@pytest.fixture
def guard():
    store = IdentityStore()

    users = [
        UserIdentity(user_id="EMP-101", role=UserRole.EMPLOYEE),
        UserIdentity(user_id="MGR-101", role=UserRole.FINANCE_MANAGER),
        UserIdentity(user_id="ADMIN-101", role=UserRole.FINANCE_ADMIN),
        UserIdentity(user_id="SYS-101", role=UserRole.SYSTEM_ADMIN),
        UserIdentity(
            user_id="INACTIVE-101",
            role=UserRole.FINANCE_ADMIN,
            active=False,
        ),
    ]

    for user in users:
        store.register_user(user)

    return AuthorizationGuard(store)


def test_employee_can_prepare_payment(guard):
    user = guard.require_prepare_payment("EMP-101")
    assert user.role == UserRole.EMPLOYEE


def test_employee_cannot_approve_payment(guard):
    with pytest.raises(PermissionError):
        guard.require_approve_payment("EMP-101")


def test_manager_can_approve_payment(guard):
    user = guard.require_approve_payment("MGR-101")
    assert user.role == UserRole.FINANCE_MANAGER


def test_manager_cannot_execute_payment(guard):
    with pytest.raises(PermissionError):
        guard.require_execute_payment("MGR-101")


def test_finance_admin_can_execute_payment(guard):
    user = guard.require_execute_payment("ADMIN-101")
    assert user.role == UserRole.FINANCE_ADMIN


def test_system_admin_has_no_payment_permissions(guard):
    with pytest.raises(PermissionError):
        guard.require_prepare_payment("SYS-101")


def test_inactive_admin_cannot_execute_payment(guard):
    with pytest.raises(PermissionError):
        guard.require_execute_payment("INACTIVE-101")


def test_unknown_user_is_rejected(guard):
    with pytest.raises(KeyError):
        guard.require_prepare_payment("UNKNOWN-101")


def test_unknown_permission_is_rejected(guard):
    with pytest.raises(ValueError):
        guard.get_authorized_user("ADMIN-101", "delete_payment")
