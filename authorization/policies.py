
from enum import Enum

from pydantic import BaseModel, ConfigDict


class UserRole(str, Enum):
    EMPLOYEE = "EMPLOYEE"
    FINANCE_MANAGER = "FINANCE_MANAGER"
    FINANCE_ADMIN = "FINANCE_ADMIN"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"


class UserIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    role: UserRole
    active: bool = True


class AuthorizationPolicy:
    """Defines role-based permissions for SafePilot."""

    @staticmethod
    def can_prepare_payment(user: UserIdentity) -> bool:
        return (
            user.active
            and user.role in {
                UserRole.EMPLOYEE,
                UserRole.FINANCE_MANAGER,
                UserRole.FINANCE_ADMIN,
            }
        )

    @staticmethod
    def can_approve_payment(user: UserIdentity) -> bool:
        return (
            user.active
            and user.role in {
                UserRole.FINANCE_MANAGER,
                UserRole.FINANCE_ADMIN,
            }
        )

    @staticmethod
    def can_execute_payment(user: UserIdentity) -> bool:
        return (
            user.active
            and user.role == UserRole.FINANCE_ADMIN
        )
