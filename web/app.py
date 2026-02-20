"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import settings
from web.database import init_db
from web.middleware.auth import (
    AuthMiddleware,
    create_session_cookie,
    verify_credentials,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
)
from web.middleware.security import SecurityHeadersMiddleware
from web.routes import digests, search, api, semantic_search, clusters, preferences, sources, chat


def get_real_ip(request: Request) -> str:
    """Get real client IP from behind Railway's proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Rate limiter (shared across routes)
limiter = Limiter(key_func=get_real_ip)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    print("Startup: initializing database...")
    init_db()
    print("Startup: importing feeds...")
    _auto_import_feeds()
    print("Startup: ready")
    yield


def _auto_import_feeds():
    """If feed_sources table is empty, seed from .txt files."""
    try:
        from web.database import SessionLocal
        from web.models import FeedSource
        with SessionLocal() as session:
            count = session.query(FeedSource).count()
            if count == 0:
                from web.routes.sources import import_feeds_from_files
                imported = import_feeds_from_files(session)
                if imported:
                    print(f"Auto-imported {imported} feed sources from .txt files")
    except Exception as e:
        print(f"Warning: feed auto-import failed: {e}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-Powered News Intelligence Platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)

# Attach limiter to app state (required by slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add middleware (order matters: last added = first executed)
# 1. GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. CORS - restrict in production
if settings.is_production:
    allowed_origins = [
        f"https://{host.strip()}" for host in settings.allowed_hosts if host.strip() != "*"
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["https://ripin.ai"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 3. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 4. Authentication
app.add_middleware(AuthMiddleware)


# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    if settings.is_development:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.state.templates = templates

# Include routers
app.include_router(digests.router)
app.include_router(search.router)
app.include_router(api.router)
app.include_router(semantic_search.router)
app.include_router(clusters.router)
app.include_router(preferences.router)
app.include_router(sources.router)
app.include_router(chat.router)


# Login routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    """Render the login page."""
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": error}
    )


@app.post("/login")
@limiter.limit("10/minute")
async def login_submit(request: Request):
    """Handle login form submission."""
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    if verify_credentials(username, password):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=create_session_cookie(),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=settings.is_production,
        )
        return response

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid password"},
        status_code=401,
    )


@app.get("/logout")
async def logout():
    """Log out and clear session."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    if settings.is_production:
        return {"status": "healthy"}
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


# --- Cron / Scheduler endpoints ---

# Runtime scheduler state (can be toggled without restarting)
_scheduler_enabled: bool = settings.scheduler_enabled


@app.post("/cron/run-digest")
async def cron_run_digest(request: Request):
    """Triggered by Railway cron. Runs the full digest pipeline."""
    # Verify cron secret to prevent unauthorized triggers
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {settings.cron_secret}":
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    # Check if scheduler is enabled
    if not _scheduler_enabled:
        return {"status": "skipped", "reason": "Scheduler disabled"}

    # Run agent in background thread (non-blocking)
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _run():
        from agent import run_agent
        run_agent()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _run)

    from datetime import datetime as dt
    return {"status": "started", "time": dt.utcnow().isoformat()}


@app.post("/cron/rebuild-clusters")
async def cron_rebuild_clusters(request: Request):
    """Rebuild topic clusters for all digests. Auth via cron secret."""
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {settings.cron_secret}":
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _run():
        from tasks.clustering_tasks import recluster_recent_digests
        results = recluster_recent_digests(days=None)
        total = sum(r.get("clusters_created", 0) for r in results if "error" not in r)
        print(f"🏷️ Cluster rebuild complete: {total} clusters across {len(results)} digests")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _run)

    from datetime import datetime as dt
    return {"status": "started", "time": dt.utcnow().isoformat()}


@app.get("/api/admin/scheduler")
async def get_scheduler_status():
    """Get scheduler status."""
    return {"enabled": _scheduler_enabled}


@app.post("/api/admin/scheduler")
async def toggle_scheduler(request: Request):
    """Toggle scheduler on/off."""
    global _scheduler_enabled
    body = await request.json()
    if "enabled" in body:
        _scheduler_enabled = bool(body["enabled"])
    else:
        _scheduler_enabled = not _scheduler_enabled
    return {"enabled": _scheduler_enabled}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.is_development,
    )
