from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.config import settings
from app.models.db_models import Base

# Async engine for FastAPI routes
async_url = str(settings.DATABASE_URL)
if "+asyncpg" not in async_url:
    async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(async_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Celery workers (uses psycopg2, not asyncpg)
sync_url = async_url.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2+psycopg2", "postgresql+psycopg2")
sync_engine = create_engine(sync_url, pool_pre_ping=True)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
