"""Security headers middleware."""
import secrets

from starlette.middleware.base import BaseHTTPMiddleware

from config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request, call_next):
        # Generate a per-request CSP nonce
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce

        response = await call_next(request)

        # Remove Content-Length to avoid mismatch when BaseHTTPMiddleware
        # re-wraps the response body (causes "Response content shorter than
        # Content-Length" with GZipMiddleware). Uvicorn will recalculate it.
        if "content-length" in response.headers:
            del response.headers["content-length"]

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Cache headers
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=86400"
        else:
            response.headers["Cache-Control"] = "private, no-cache"

        # Note: 'unsafe-inline' is required because many templates use inline
        # event handlers (onclick, onsubmit, onchange).  Per the CSP spec,
        # 'unsafe-inline' is ignored when a nonce is present, so we cannot
        # use nonces and inline handlers at the same time.  The nonce is
        # still generated and available in templates for future migration
        # to addEventListener-only (chat.html already uses this pattern).
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )

        return response
