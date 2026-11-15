
from authorization.identity_store import IdentityStore
from authorization.policies import AuthorizationPolicy, UserIdentity


class AuthorizationGuard:
    """Enforces role-based permissions using registered identities."""

    def __init__(self, identity_store: IdentityStore):
        self.identity_store = identity_store

    def get_authorized_user(
        self,
        user_id: str,
        permission: str,
    ) -> UserIdentity:
        """Retrieve a registered user and verify the requested permission."""

        user = self.identity_store.get_user(user_id)

        permission_checks = {
            "prepare_payment": AuthorizationPolicy.can_prepare_payment,
            "approve_payment": AuthorizationPolicy.can_approve_payment,
            "execute_payment": AuthorizationPolicy.can_execute_payment,
        }

        check = permission_checks.get(permission)

        if check is None:
            raise ValueError(f"Unknown permission: {permission}")

        if not check(user):
            raise PermissionError(
                f"User {user_id} is not authorized to {permission}"
            )

        return user

    def require_prepare_payment(self, user_id: str) -> UserIdentity:
        return self.get_authorized_user(user_id, "prepare_payment")

    def require_approve_payment(self, user_id: str) -> UserIdentity:
        return self.get_authorized_user(user_id, "approve_payment")

    def require_execute_payment(self, user_id: str) -> UserIdentity:
        return self.get_authorized_user(user_id, "execute_payment")
