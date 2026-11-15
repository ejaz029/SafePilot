
from threading import RLock

from authorization.policies import UserIdentity


class IdentityStore:
    """Stores and retrieves registered SafePilot user identities."""

    def __init__(self):
        self._users: dict[str, UserIdentity] = {}
        self._lock = RLock()

    def register_user(self, user: UserIdentity) -> UserIdentity:
        """Register a new user identity."""
        with self._lock:
            if user.user_id in self._users:
                raise ValueError(f"User already exists: {user.user_id}")

            self._users[user.user_id] = user.model_copy(deep=True)
            return user.model_copy(deep=True)

    def get_user(self, user_id: str) -> UserIdentity:
        """Retrieve a registered user identity."""
        with self._lock:
            user = self._users.get(user_id)

            if user is None:
                raise KeyError(f"User not found: {user_id}")

            return user.model_copy(deep=True)

    def update_user(self, user: UserIdentity) -> UserIdentity:
        """Update an existing user's role or active status."""
        with self._lock:
            if user.user_id not in self._users:
                raise KeyError(f"User not found: {user.user_id}")

            self._users[user.user_id] = user.model_copy(deep=True)
            return user.model_copy(deep=True)

    def count(self) -> int:
        with self._lock:
            return len(self._users)
