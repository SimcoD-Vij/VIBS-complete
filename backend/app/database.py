from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session as _SyncSession
from sqlalchemy import create_engine
from app.config import settings
from app.models.db_models import Base

async_url = str(settings.DATABASE_URL)
if "+asyncpg" not in async_url:
    async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(async_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

sync_url = async_url.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2+psycopg2", "postgresql+psycopg2")
sync_engine = create_engine(sync_url, pool_pre_ping=True)

def SessionLocal() -> _SyncSession:
    """Synchronous session for Celery workers and background threads."""
    return _SyncSession(bind=sync_engine)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
