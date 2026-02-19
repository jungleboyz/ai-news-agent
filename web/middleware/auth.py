"""Session-based authentication middleware."""
import hashlib
import hmac
import json
import time
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings


# Paths that don't require authentication
PUBLIC_PATHS = {"/login", "/health", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)

# Session duration: 30 days
SESSION_MAX_AGE = 30 * 24 * 60 * 60
SESSION_COOKIE = "session"


def _sign(payload: str) -> str:
    """Create HMAC signature for a payload."""
    return hmac.new(
        settings.secret_key.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()


def create_session_cookie() -> str:
    """Create a signed session cookie value."""
    data = json.dumps({"authenticated": True, "created": int(time.time())})
    signature = _sign(data)
    return f"{data}|{signature}"


def verify_session_cookie(cookie: str) -> bool:
    """Verify a signed session cookie."""
    try:
        parts = cookie.rsplit("|", 1)
        if len(parts) != 2:
            return False
        data, signature = parts
        expected = _sign(data)
        if not hmac.compare_digest(signature, expected):
            return False
        payload = json.loads(data)
        if not payload.get("authenticated"):
            return False
        # Check expiry
        created = payload.get("created", 0)
        if time.time() - created > SESSION_MAX_AGE:
            return False
        return True
    except Exception:
        return False


def verify_password(password: str) -> bool:
    """Check if the provided password matches SITE_PASSWORD."""
    site_password = settings.site_password
    if not site_password:
        return False
    return hmac.compare_digest(password, site_password)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that requires authentication for all routes except public ones."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for public paths
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)

        # Skip auth if no SITE_PASSWORD is configured (dev mode)
        if not settings.site_password:
            return await call_next(request)

        # Check session cookie
        session_cookie = request.cookies.get(SESSION_COOKIE)
        if session_cookie and verify_session_cookie(session_cookie):
            return await call_next(request)

        # Not authenticated - redirect to login
        # For API requests, return 401 instead of redirect
        if path.startswith("/api/"):
            return Response(
                content=json.dumps({"detail": "Authentication required"}),
                status_code=401,
                media_type="application/json",
            )

        return RedirectResponse(url="/login", status_code=302)
