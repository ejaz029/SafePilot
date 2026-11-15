
import pytest

from authorization.identity_store import IdentityStore
from authorization.policies import UserIdentity, UserRole


@pytest.fixture
def store():
    identity_store = IdentityStore()

    identity_store.register_user(
        UserIdentity(
            user_id="EMPLOYEE-201",
            role=UserRole.EMPLOYEE,
        )
    )

    identity_store.register_user(
        UserIdentity(
            user_id="MANAGER-101",
            role=UserRole.FINANCE_MANAGER,
        )
    )

    return identity_store


def test_register_and_retrieve_user(store):
    user = store.get_user("EMPLOYEE-201")

    assert user.user_id == "EMPLOYEE-201"
    assert user.role == UserRole.EMPLOYEE


def test_register_duplicate_user_is_rejected(store):
    duplicate = UserIdentity(
        user_id="EMPLOYEE-201",
        role=UserRole.FINANCE_ADMIN,
    )

    with pytest.raises(ValueError):
        store.register_user(duplicate)


def test_unknown_user_is_rejected(store):
    with pytest.raises(KeyError):
        store.get_user("UNKNOWN-USER")


def test_update_user_role(store):
    updated_user = UserIdentity(
        user_id="MANAGER-101",
        role=UserRole.FINANCE_ADMIN,
    )

    store.update_user(updated_user)

    assert store.get_user("MANAGER-101").role == UserRole.FINANCE_ADMIN


def test_deactivate_user(store):
    updated_user = UserIdentity(
        user_id="EMPLOYEE-201",
        role=UserRole.EMPLOYEE,
        active=False,
    )

    store.update_user(updated_user)

    assert store.get_user("EMPLOYEE-201").active is False


def test_update_unknown_user_is_rejected(store):
    unknown_user = UserIdentity(
        user_id="UNKNOWN-USER",
        role=UserRole.EMPLOYEE,
    )

    with pytest.raises(KeyError):
        store.update_user(unknown_user)


def test_identity_store_count(store):
    assert store.count() == 2


def test_returned_user_cannot_mutate_stored_identity(store):
    user = store.get_user("EMPLOYEE-201")
    user.active = False

    assert store.get_user("EMPLOYEE-201").active is True
