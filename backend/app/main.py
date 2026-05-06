import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import create_tables
from app.routers import ws_realtime, jobs_router, upload_router, export_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting VIBS backend...")
    await create_tables()
    logger.info("Database tables ready ✓")

    # Preload all ML models in a background thread (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _preload_models)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down...")


def _preload_models():
    try:
        from app.services.audio_pipeline import preload_all
        preload_all()
    except Exception as e:
        logger.warning(f"Model preload error (non-fatal): {e}")


app = FastAPI(
    title="VIBS — Voice Intelligence Backend System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ws_realtime.router)
app.include_router(upload_router.router)
app.include_router(jobs_router.router)
app.include_router(export_router.router)

@app.get("/")
async def root():
    return {"name": "VIBS", "version": "2.0", "status": "running"}
