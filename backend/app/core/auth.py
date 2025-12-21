from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_admin(x_admin_key: str | None = Header(default=None)) -> bool:
    """
    Simple admin auth for publish endpoints.
    Replace with JWT or internal network policies later.
    """
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_API_KEY not configured",
        )
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")
    return True


def require_premium(x_premium_key: str | None = Header(default=None)) -> bool:
    """
    Premium entitlement check (simple version).
    Later you can replace with Stripe subscription lookup or JWT claim verification.

    Current behavior:
    - If PREMIUM_API_KEY is set, require a matching `x-premium-key`.
    - If PREMIUM_API_KEY is empty, deny by default (safe).
    """
    if not settings.premium_api_key:
        # Safe default: premium disabled until configured
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Premium not enabled")
    if x_premium_key != settings.premium_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Premium required")
    return True
