import asyncio
import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

# Mock settings or import them
from app.config import settings
from app.models.db_models import Session, Segment, Speaker, GraphData, TopicShift
from app.routers.export_router import _build_html

async def debug_export(session_id):
    async_url = str(settings.DATABASE_URL)
    if "+asyncpg" not in async_url:
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    
    engine = create_async_engine(async_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as db:
        # Load all data
        r = await db.execute(select(Session).where(Session.id == session_id))
        session = r.scalar_one_or_none()
        if not session:
            print(f"Session {session_id} not found")
            return

        segs = await db.execute(select(Segment).where(Segment.session_id == session_id).order_by(Segment.start_time))
        spks = await db.execute(select(Speaker).where(Speaker.session_id == session_id))
        shifts = await db.execute(select(TopicShift).where(TopicShift.session_id == session_id).order_by(TopicShift.time_seconds))
        graph_res = await db.execute(select(GraphData).where(GraphData.session_id == session_id))

        segments = segs.scalars().all()
        speakers = spks.scalars().all()
        topic_shifts = shifts.scalars().all()
        graph = graph_res.scalar_one_or_none()

        print(f"Building HTML for session {session_id}...")
        html = _build_html(session, speakers, segments, topic_shifts, graph)
        
        print("Rendering to PDF with WeasyPrint...")
        from weasyprint import HTML
        try:
            pdf_bytes = HTML(string=html).write_pdf()
            with open(f"debug_{session_id[:8]}.pdf", "wb") as f:
                f.write(pdf_bytes)
            print("Successfully rendered PDF!")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "efbfce75-1f86-4435-a205-7ba41443660f"
    asyncio.run(debug_export(sid))
