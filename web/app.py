"""FastAPI application entry point."""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.database import init_db
from web.routes import digests, search, api

# Initialize FastAPI app
app = FastAPI(
    title="AI News Agent",
    description="Web interface for browsing AI news digests",
    version="1.0.0"
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


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
