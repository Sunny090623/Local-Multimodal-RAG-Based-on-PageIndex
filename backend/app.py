"""
Application entry point.

Creates the FastAPI app, registers CORS, includes route modules,
manages application lifespan, and mounts static files.
"""

import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Bootstrap: ensure project root is on sys.path when run as a script
# (backend/__init__.py won't auto-load in that case)
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import backend  # noqa: E402 — triggers __init__.py to set up PageIndex path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.settings import router as settings_router, load_settings
from backend.routes.upload import router as upload_router, task_worker
from backend.routes.documents import router as documents_router
from backend.routes.chat import router as chat_router

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    log_file = Path(__file__).parent.parent / "Log.txt"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file handler configured. Logging to {log_file}")
except Exception as e:
    logger.error(f"Failed to setup file log handler: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_settings()
    worker = asyncio.create_task(task_worker())
    yield
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


# FastAPI application
app = FastAPI(title="Local Multi-Modal Vectorless RAG", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(settings_router)
app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(chat_router)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8088)
