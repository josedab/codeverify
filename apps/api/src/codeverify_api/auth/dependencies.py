"""Authentication dependencies for FastAPI."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from codeverify_api.auth.jwt import TokenData, decode_access_token

security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> TokenData | None:
    """Get current user if authenticated, None otherwise."""
    if credentials is None:
        return None

    token_data = decode_access_token(credentials.credentials)
    return token_data


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> TokenData:
    """Get current user, raise 401 if not authenticated."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_access_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data


# Dependency aliases
require_auth = Depends(get_current_user)
optional_auth = Depends(get_current_user_optional)


async def require_admin(
    user: Annotated[TokenData, Depends(get_current_user)],
) -> TokenData:
    """Require admin role.

    Checks if the user has admin privileges. Admin status is determined by:
    1. The 'is_admin' flag in the JWT token (set during login for admin users)
    2. Membership in an organization with admin role

    Note: For production, consider caching admin status or using a more
    sophisticated RBAC system.
    """
    # Check for admin flag in token
    if getattr(user, "is_admin", False):
        return user

    # Check for admin role in token roles
    roles = getattr(user, "roles", [])
    if "admin" in roles or "org_admin" in roles:
        return user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin privileges required",
    )
