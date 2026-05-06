"""
export_router.py
================
PDF export endpoint. Generates a multi-page PDF report from a completed session.
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.db_models import Session, Segment, Speaker, GraphData, TopicShift

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


@router.get("/session/{session_id}/export/pdf")
async def export_pdf(session_id: str, db: AsyncSession = Depends(get_db)):
    # Load all data
    r = await db.execute(select(Session).where(Session.id == session_id))
    session = r.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "complete":
        raise HTTPException(202, "Session still processing")

    segs = await db.execute(select(Segment).where(Segment.session_id == session_id).order_by(Segment.start_time))
    spks = await db.execute(select(Speaker).where(Speaker.session_id == session_id))
    shifts = await db.execute(select(TopicShift).where(TopicShift.session_id == session_id).order_by(TopicShift.time_seconds))

    segments = segs.scalars().all()
    speakers = spks.scalars().all()
    topic_shifts = shifts.scalars().all()

    # Build HTML
    html = _build_html(session, speakers, segments, topic_shifts)

    # Render to PDF
    try:
        from weasyprint import HTML as WP_HTML
        pdf_bytes = WP_HTML(string=html).write_pdf()
    except ImportError:
        raise HTTPException(500, "weasyprint not installed — PDF export unavailable")
    except Exception as e:
        logger.exception(f"PDF render failed: {e}")
        raise HTTPException(500, f"PDF render failed: {e}")

    filename = f"vibs_session_{session_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _build_html(session, speakers, segments, topic_shifts) -> str:
    duration_str = _format_time(session.duration_seconds or 0)

    # Speaker cards HTML
    speaker_cards = ""
    for spk in speakers:
        bullets = ""
        if spk.summary:
            for line in spk.summary.split("\n"):
                line = line.strip()
                if line:
                    bullets += f"<li>{line.lstrip('•').strip()}</li>"
        speaker_cards += f"""
        <div class="speaker-card" style="border-left: 4px solid {spk.color};">
          <div class="speaker-header">
            <span class="speaker-dot" style="background:{spk.color}"></span>
            <span class="speaker-name">{spk.display_name or spk.speaker_label}</span>
            <span class="speaker-stats">{_format_time(spk.total_seconds or 0)} · {int(spk.talk_share or 0)}% of session</span>
          </div>
          <ul class="summary-bullets">{bullets or "<li>No summary available</li>"}</ul>
        </div>"""

    # Topic shifts
    shifts_html = ""
    for shift in topic_shifts:
        shifts_html += f"""
        <div class="shift-row">
          <span class="shift-time">{_format_time(shift.time_seconds)}</span>
          <span class="shift-from">{shift.from_topic}</span>
          <span class="shift-arrow">→</span>
          <span class="shift-to">{shift.to_topic}</span>
          <span class="shift-speaker">({shift.speaker_label})</span>
        </div>"""

    # Transcript
    transcript_html = ""
    spk_map = {s.speaker_label: s for s in speakers}
    for seg in segments:
        spk = spk_map.get(seg.speaker_label)
        color = spk.color if spk else "#888"
        name = spk.display_name if spk else seg.speaker_label
        overlap_style = "background:#fffbe6;" if seg.is_overlap else ""
        transcript_html += f"""
        <div class="transcript-row" style="{overlap_style}">
          <span class="t-time">{_format_time(seg.start_time)}</span>
          <span class="t-spk" style="color:{color}">{name}</span>
          <span class="t-text">{seg.text}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: -apple-system, sans-serif; color: #1a1a1a; font-size: 12px; margin: 0; }}
  .page {{ padding: 40px; page-break-after: always; }}
  .page:last-child {{ page-break-after: avoid; }}
  h1 {{ font-size: 22px; color: #111; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 24px; }}
  .meta {{ color: #888; font-size: 11px; margin-bottom: 24px; }}
  .speaker-card {{ margin-bottom: 20px; padding: 12px 16px; background: #fafafa; border-radius: 6px; }}
  .speaker-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .speaker-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .speaker-name {{ font-weight: 600; font-size: 13px; }}
  .speaker-stats {{ color: #888; font-size: 11px; margin-left: auto; }}
  .summary-bullets {{ margin: 0; padding-left: 18px; color: #333; line-height: 1.7; }}
  .shift-row {{ display: flex; gap: 12px; align-items: center; padding: 6px 0; border-bottom: 1px solid #f0f0f0; font-size: 11px; }}
  .shift-time {{ color: #888; font-family: monospace; min-width: 40px; }}
  .shift-from {{ color: #555; }}
  .shift-arrow {{ color: #888; }}
  .shift-to {{ font-weight: 500; }}
  .shift-speaker {{ color: #aaa; }}
  .transcript-row {{ display: grid; grid-template-columns: 40px 100px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px solid #f5f5f5; font-size: 11px; line-height: 1.5; }}
  .t-time {{ color: #999; font-family: monospace; }}
  .t-spk {{ font-weight: 600; }}
  .t-text {{ color: #333; }}
  @page {{ size: A4; margin: 20mm; }}
</style>
</head>
<body>

<div class="page">
  <h1>Session Report</h1>
  <p class="meta">Session ID: {session.id} &nbsp;·&nbsp; Duration: {duration_str} &nbsp;·&nbsp; Speakers: {session.speaker_count or len(speakers)}</p>
  <h2>Speakers</h2>
  {speaker_cards}
</div>

<div class="page">
  <h2>Topic Shifts</h2>
  {shifts_html or "<p style='color:#aaa'>No topic shifts detected.</p>"}
</div>

<div class="page">
  <h2>Full Transcript</h2>
  {transcript_html}
</div>

</body>
</html>"""
