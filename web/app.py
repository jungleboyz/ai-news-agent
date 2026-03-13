"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager
from datetime import date, datetime as dt, timezone

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from sqlalchemy.orm import Session

from config import settings
from web.database import get_db, init_db
from web.models import Digest
from web.middleware.auth import (
    AuthMiddleware,
    create_session_cookie,
    verify_credentials,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
)
from web.middleware.security import SecurityHeadersMiddleware
from web.routes import digests, search, api, semantic_search, clusters, preferences, sources, chat


# APScheduler instance (created once, started in lifespan)
scheduler = BackgroundScheduler()

DIGEST_JOB_ID = "daily_digest"


def _run_digest_job():
    """Job function called by APScheduler (and /cron/run-digest) to run the digest pipeline."""
    import time as _time
    import traceback as _tb

    start = _time.monotonic()
    started_at = dt.now(timezone.utc)
    print(f"Scheduler: starting digest run at {started_at.isoformat()}")
    error_msg = None
    try:
        from agent import run_agent
        run_agent()
        print("Scheduler: digest run completed")
    except Exception as e:
        error_msg = f"{e}\n{_tb.format_exc()}"
        print(f"Scheduler: digest run failed:\n{_tb.format_exc()}")

    elapsed = _time.monotonic() - start
    _send_digest_status_email(started_at, elapsed, error_msg)


def _send_digest_status_email(started_at: dt, elapsed_seconds: float, error: str | None):
    """Send a status report email with daily brief after the digest pipeline finishes."""
    import resend

    api_key = os.getenv("RESEND_API_KEY")
    from_email = os.getenv("FROM_EMAIL") or os.getenv("EMAIL_FROM", "onboarding@resend.dev")
    to_email = os.getenv("EMAIL_TO") or from_email
    if not api_key or not to_email:
        print("Scheduler: status email skipped (RESEND_API_KEY not configured)")
        return

    # Gather digest stats from DB
    status = "SUCCESS" if error is None else "FAILED"
    stats_lines = []
    brief_text = ""
    brief_html = ""
    digest_date = None
    try:
        from web.database import SessionLocal
        from web.models import Digest, Item

        today_utc = dt.now(timezone.utc).date()
        with SessionLocal() as session:
            digest = session.query(Digest).filter(
                Digest.date == today_utc
            ).first()
            if digest:
                digest_date = digest.date
                items = session.query(Item).filter(Item.digest_id == digest.id).all()
                by_type: dict[str, list] = {}
                for item in items:
                    by_type.setdefault(item.type, []).append(item)

                stats_lines.append(f"Digest date: {digest.date}")
                stats_lines.append(f"News sources: {digest.news_sources_count}")
                stats_lines.append(f"Podcast sources: {digest.podcast_sources_count}")
                stats_lines.append(f"Items considered: {digest.total_items_considered}")
                stats_lines.append(f"Items in digest: {len(items)}")
                stats_lines.append("")
                for item_type, type_items in sorted(by_type.items()):
                    stats_lines.append(f"  {item_type}: {len(type_items)} items")

                # Generate daily brief summary
                try:
                    from services.daily_brief import DailyBriefService
                    brief_service = DailyBriefService()
                    summary = brief_service.get_or_generate_summary(
                        session, digest_date=digest.date
                    )
                    if "error" not in summary:
                        brief_text = brief_service.generate_brief_text(
                            summary, digest.date
                        )
                        brief_html = brief_service.generate_brief_html(
                            summary, digest.date
                        )
                        print("Scheduler: daily brief generated for status email")
                    else:
                        brief_text = f"\n(Brief generation failed: {summary['error']})\n"
                except Exception as e:
                    print(f"Scheduler: brief generation failed: {e}")
                    brief_text = f"\n(Brief generation failed: {e})\n"
            else:
                stats_lines.append("No digest found in database for today.")
    except Exception as e:
        stats_lines.append(f"Could not query digest stats: {e}")

    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)

    status_block = f"""Neural Feed Digest — Status Report
{'=' * 40}

Status:   {status}
Started:  {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
Duration: {minutes}m {seconds}s
"""
    if error:
        status_block += f"\nError:\n  {error}\n"

    status_block += f"\n{chr(10).join(stats_lines)}\n"

    # Plain text: status block + brief
    plain_body = status_block
    if brief_text:
        plain_body += f"\n\n{brief_text}"

    subject = f"Neural Feed {'OK' if error is None else 'FAILED'} — {started_at.strftime('%b %d')}"

    try:
        resend.api_key = api_key

        # Build HTML body if we have the brief
        html_body = None
        if brief_html:
            status_html = status_block.replace("\n", "<br>\n")
            html_body = (
                f'<div style="font-family:monospace;font-size:13px;'
                f'background:#0f172a;color:#94a3b8;padding:20px;'
                f'border-radius:8px;margin-bottom:24px;">'
                f'{status_html}</div>\n'
                f'{brief_html}'
            )

        params = {
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "text": plain_body,
        }
        if html_body:
            params["html"] = html_body

        resend.Emails.send(params)
        print(f"Scheduler: status email sent to {to_email}")
    except Exception as e:
        print(f"Scheduler: failed to send status email: {e}")


def _scheduler_event_listener(event):
    """Log APScheduler job misfire and error events."""
    if event.code == EVENT_JOB_MISSED:
        print(
            f"Scheduler WARNING: job '{event.job_id}' missed its scheduled run time. "
            f"This typically means the process was busy or paused at the scheduled time."
        )
    elif event.code == EVENT_JOB_ERROR:
        print(f"Scheduler ERROR: job '{event.job_id}' raised an exception: {event.exception}")


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
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _init():
        print("Startup: initializing database...")
        init_db()
        print("Startup: importing feeds...")
        _auto_import_feeds()
        print("Startup: ready")

    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _init),
            timeout=20,
        )
    except asyncio.TimeoutError:
        print("Startup warning: DB init timed out after 20s — app will start anyway")
    except Exception as e:
        print(f"Startup warning: {e} — app will start anyway")

    # Start APScheduler
    if settings.scheduler_enabled:
        # Use Australia/Sydney timezone so the schedule automatically
        # adjusts for AEST/AEDT daylight saving transitions.
        # SCHEDULER_HOUR is now interpreted as local Sydney time (default: 6 = 6 AM).
        trigger = CronTrigger(
            hour=settings.scheduler_cron_hour,
            minute=settings.scheduler_cron_minute,
            timezone=settings.scheduler_timezone,
        )
        scheduler.add_job(
            _run_digest_job,
            trigger=trigger,
            id=DIGEST_JOB_ID,
            replace_existing=True,
            misfire_grace_time=900,  # 15 min window to still run if delayed
            max_instances=1,
        )
        scheduler.add_listener(_scheduler_event_listener, EVENT_JOB_MISSED | EVENT_JOB_ERROR)
        scheduler.start()
        job = scheduler.get_job(DIGEST_JOB_ID)
        print(f"Scheduler started, next run at {job.next_run_time}")

        # Catch-up: if it's past today's scheduled time and no digest exists
        # for today, run immediately.  This handles Railway restarts, redeploys,
        # and container sleep that cause the in-memory scheduler to miss its
        # window entirely.
        _maybe_run_catchup(loop, ThreadPoolExecutor(max_workers=1))
    else:
        print("Scheduler disabled via SCHEDULER_ENABLED=false")

    yield

    # Shutdown scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("Scheduler shut down")


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


def _maybe_run_catchup(loop, executor):
    """If today's scheduled time has passed and no digest exists yet, trigger a
    catch-up run in the background.  This makes the system self-healing after
    Railway restarts, redeploys, or container sleep events that cause the
    in-memory APScheduler to miss its daily window."""
    import zoneinfo

    sched_tz = zoneinfo.ZoneInfo(settings.scheduler_timezone)
    now_local = dt.now(sched_tz)
    scheduled_hour = settings.scheduler_cron_hour
    scheduled_minute = settings.scheduler_cron_minute

    # Compare in the scheduler's timezone (e.g. Australia/Sydney) so the
    # catch-up triggers at the right wall-clock time regardless of DST.
    scheduled_today = now_local.replace(
        hour=scheduled_hour, minute=scheduled_minute, second=0, microsecond=0
    )
    if now_local < scheduled_today:
        print(f"Startup catch-up: not yet past {scheduled_hour:02d}:{scheduled_minute:02d} "
              f"{settings.scheduler_timezone}, skipping")
        return

    # Check DB for today's digest.  The digest date is stored as UTC date,
    # so use UTC for the lookup.
    today_utc = dt.now(timezone.utc).date()
    try:
        from web.database import SessionLocal
        from web.models import Digest
        with SessionLocal() as session:
            existing = session.query(Digest.id).filter(
                Digest.date == today_utc
            ).first()
            if existing:
                print(f"Startup catch-up: digest for {today_utc} already exists, skipping")
                return
    except Exception as e:
        print(f"Startup catch-up: DB check failed ({e}), skipping to avoid duplicate runs")
        return

    print(
        f"Startup catch-up: no digest for {today_utc} and it's past "
        f"{scheduled_hour:02d}:{scheduled_minute:02d} {settings.scheduler_timezone} "
        f"— triggering catch-up run"
    )
    loop.run_in_executor(executor, _run_digest_job)


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


# HEAD request middleware — converts HEAD to GET, then strips the body
@app.middleware("http")
async def head_request_middleware(request: Request, call_next):
    if request.method == "HEAD":
        request.scope["method"] = "GET"
        response = await call_next(request)
        return Response(
            content=b"",
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
    return await call_next(request)


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


# --- Audio API ---

@app.get("/api/audio/brief/{digest_date}")
async def audio_brief(digest_date: str, db: Session = Depends(get_db)):
    """Stream TTS audio for a daily brief, generating on first request."""
    from services.daily_brief import DailyBriefService
    from services.voice_service import VoiceService

    try:
        parsed_date = date.fromisoformat(digest_date)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid date format"})

    digest = db.query(Digest).filter(Digest.date == parsed_date).first()
    if not digest:
        return JSONResponse(status_code=404, content={"detail": "Digest not found"})

    summary = DailyBriefService().get_or_generate_summary(db, parsed_date)

    voice = VoiceService()
    audio_bytes = voice.get_or_generate_audio_bytes(summary, parsed_date, db_session=db, digest=digest)
    if not audio_bytes:
        return JSONResponse(status_code=404, content={"detail": "Audio not available"})

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


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


# Privacy policy (unlisted — direct access only, not linked from site)
@app.api_route("/privacy", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
async def privacy_health_intelligence(request: Request):
    return templates.TemplateResponse("privacy_health.html", {"request": request})


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
    import hmac as _hmac
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.cron_secret}"
    if not settings.cron_secret or not _hmac.compare_digest(auth, expected):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    # Check if scheduler is enabled
    if not _scheduler_enabled:
        return {"status": "skipped", "reason": "Scheduler disabled"}

    # Run agent in background thread (non-blocking) — use the same
    # wrapper as the APScheduler job so errors are captured and a
    # status email is always sent.
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _run_digest_job)

    return {"status": "started", "time": dt.now(timezone.utc).isoformat()}


@app.post("/cron/rebuild-clusters")
async def cron_rebuild_clusters(request: Request):
    """Rebuild topic clusters for all digests. Auth via cron secret."""
    import hmac as _hmac
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.cron_secret}"
    if not settings.cron_secret or not _hmac.compare_digest(auth, expected):
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

    return {"status": "started", "time": dt.now(timezone.utc).isoformat()}


@app.get("/api/admin/scheduler")
async def get_scheduler_status():
    """Get scheduler status."""
    job = scheduler.get_job(DIGEST_JOB_ID) if scheduler.running else None
    next_run = None
    if job and job.next_run_time:
        next_run = job.next_run_time.isoformat()
    return {
        "enabled": _scheduler_enabled,
        "running": scheduler.running,
        "next_run": next_run,
    }


@app.post("/api/admin/scheduler")
async def toggle_scheduler(request: Request):
    """Toggle scheduler on/off. Pauses/resumes the APScheduler job."""
    global _scheduler_enabled
    body = await request.json()
    if "enabled" in body:
        _scheduler_enabled = bool(body["enabled"])
    else:
        _scheduler_enabled = not _scheduler_enabled

    if scheduler.running:
        job = scheduler.get_job(DIGEST_JOB_ID)
        if job:
            if _scheduler_enabled:
                job.resume()
            else:
                job.pause()

    return {"enabled": _scheduler_enabled}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.is_development,
    )
