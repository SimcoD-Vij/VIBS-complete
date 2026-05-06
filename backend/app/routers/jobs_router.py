from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.db_models import Session, Segment, Speaker, GraphData, TopicShift

router = APIRouter(prefix="/api")


@router.get("/session/{session_id}/status")
async def get_status(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {
        "status": session.status,
        "progress_percent": session.progress_percent,
        "speaker_count": session.speaker_count,
        "duration_seconds": session.duration_seconds,
    }


@router.get("/session/{session_id}/result")
async def get_result(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status not in ("complete",):
        return JSONResponse({"status": session.status, "progress": session.progress_percent}, status_code=202)

    segs = await db.execute(select(Segment).where(Segment.session_id == session_id).order_by(Segment.start_time))
    speakers = await db.execute(select(Speaker).where(Speaker.session_id == session_id))
    graph = await db.execute(select(GraphData).where(GraphData.session_id == session_id))
    shifts = await db.execute(select(TopicShift).where(TopicShift.session_id == session_id).order_by(TopicShift.time_seconds))

    import json
    graph_rec = graph.scalar_one_or_none()
    return {
        "session": {
            "id": session.id,
            "duration_seconds": session.duration_seconds,
            "speaker_count": session.speaker_count,
            "created_at": str(session.created_at),
        },
        "speakers": [
            {
                "speaker_label": s.speaker_label,
                "display_name": s.display_name,
                "summary": s.summary,
                "total_seconds": s.total_seconds,
                "talk_share": s.talk_share,
                "color": s.color,
            }
            for s in speakers.scalars().all()
        ],
        "segments": [
            {
                "speaker_label": s.speaker_label,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "is_overlap": s.is_overlap,
            }
            for s in segs.scalars().all()
        ],
        "graph": {
            "nodes": json.loads(graph_rec.nodes_json) if graph_rec else [],
            "edges": json.loads(graph_rec.edges_json) if graph_rec else [],
            "explanation": graph_rec.explanation if graph_rec else "",
        },
        "topic_shifts": [
            {
                "time_seconds": t.time_seconds,
                "from_topic": t.from_topic,
                "to_topic": t.to_topic,
                "speaker_label": t.speaker_label,
            }
            for t in shifts.scalars().all()
        ],
    }


@router.patch("/session/{session_id}/speaker/{speaker_label}")
async def rename_speaker(
    session_id: str,
    speaker_label: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Speaker).where(Speaker.session_id == session_id, Speaker.speaker_label == speaker_label)
    )
    spk = result.scalar_one_or_none()
    if not spk:
        raise HTTPException(404, "Speaker not found")
    spk.display_name = body.get("display_name", spk.display_name)
    await db.commit()
    return {"ok": True, "display_name": spk.display_name}


@router.get("/status/health")
async def health():
    from app.config import settings
    return {
        "status": "ok",
        "device": settings.device,
        "compute_type": settings.compute_type,
        "gpu": settings.gpu_info,
        "model": settings.WHISPER_MODEL,
    }
