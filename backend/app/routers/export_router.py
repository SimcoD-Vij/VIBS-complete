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
    graph_res = await db.execute(select(GraphData).where(GraphData.session_id == session_id))

    segments = segs.scalars().all()
    speakers = spks.scalars().all()
    topic_shifts = shifts.scalars().all()
    graph = graph_res.scalar_one_or_none()

    # Build HTML
    html = _build_html(session, speakers, segments, topic_shifts, graph)

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


def _build_html(session, speakers, segments, topic_shifts, graph) -> str:
    duration_str = _format_time(session.duration_seconds or 0)

    # Overview / Explanation
    overview_html = ""
    if graph and graph.explanation:
        overview_html = f"""
        <div class="section-block">
          <h2>Session Overview</h2>
          <div class="explanation-text">{graph.explanation}</div>
        </div>"""

    # Mindmap / Graph
    graph_html = ""
    if graph:
        try:
            nodes = json.loads(graph.nodes_json)
            edges = json.loads(graph.edges_json)
            
            node_list = "".join([f"<li><strong>{n.get('id', 'Node')}</strong>: {n.get('label', '')}</li>" for n in nodes])
            edge_list = "".join([f"<li>{e.get('source')} &rarr; {e.get('target')} ({e.get('label', '')})</li>" for e in edges])
            
            graph_html = f"""
            <div class="section-block">
              <h2>Knowledge Graph (Mindmap)</h2>
              <div class="graph-content">
                <h3>Entities & Concepts</h3>
                <ul>{node_list or "<li>None</li>"}</ul>
                <h3>Relationships</h3>
                <ul>{edge_list or "<li>None</li>"}</ul>
              </div>
            </div>"""
        except Exception:
            graph_html = "<!-- Graph parsing failed -->"

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
            <span class="speaker-stats">{_format_time(spk.total_seconds or 0)} · {int(spk.talk_share or 0)}% share</span>
          </div>
          <ul class="summary-bullets">{bullets or "<li>No summary available</li>"}</ul>
        </div>"""

    # Topic shifts (Timeline)
    shifts_html = ""
    if topic_shifts:
        rows = ""
        for shift in topic_shifts:
            rows += f"""
            <tr>
              <td class="shift-time">{_format_time(shift.time_seconds)}</td>
              <td>{shift.from_topic} &rarr; <strong>{shift.to_topic}</strong></td>
              <td class="shift-speaker">{shift.speaker_label}</td>
            </tr>"""
        shifts_html = f"""
        <table class="shifts-table">
          <thead><tr><th>Time</th><th>Topic Transition</th><th>By</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    # Transcript
    transcript_rows = ""
    spk_map = {s.speaker_label: s for s in speakers}
    for seg in segments:
        spk = spk_map.get(seg.speaker_label)
        color = spk.color if spk else "#888"
        name = spk.display_name if spk else seg.speaker_label
        overlap_style = "background:#fffbe6;" if seg.is_overlap else ""
        transcript_rows += f"""
        <tr style="{overlap_style}">
          <td class="t-time">{_format_time(seg.start_time)}</td>
          <td class="t-spk" style="color:{color}">{name}</td>
          <td class="t-text">{seg.text}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1a1a1a; font-size: 11pt; line-height: 1.5; margin: 0; }}
  .page {{ padding: 20mm; page-break-after: always; }}
  .page:last-child {{ page-break-after: avoid; }}
  h1 {{ font-size: 24pt; color: #111; margin-bottom: 5pt; border-bottom: 2pt solid #eee; padding-bottom: 10pt; }}
  h2 {{ font-size: 16pt; color: #222; border-bottom: 1pt solid #ddd; padding-bottom: 5pt; margin-top: 25pt; margin-bottom: 15pt; }}
  h3 {{ font-size: 13pt; color: #444; margin-bottom: 8pt; }}
  .meta {{ color: #666; font-size: 10pt; margin-bottom: 30pt; }}
  .section-block {{ margin-bottom: 30pt; }}
  .explanation-text {{ background: #f9f9f9; padding: 15pt; border-radius: 8pt; color: #333; font-style: italic; }}
  .speaker-card {{ margin-bottom: 15pt; padding: 10pt 15pt; background: #fafafa; border-radius: 8pt; border: 1pt solid #eee; }}
  .speaker-header {{ margin-bottom: 8pt; }}
  .speaker-dot {{ width: 8pt; height: 8pt; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 5pt; }}
  .speaker-name {{ font-weight: bold; font-size: 12pt; vertical-align: middle; }}
  .speaker-stats {{ color: #888; font-size: 9pt; float: right; }}
  .summary-bullets {{ margin: 0; padding-left: 15pt; color: #444; }}
  .graph-content ul {{ padding-left: 15pt; margin-bottom: 15pt; list-style-type: square; }}
  .graph-content li {{ margin-bottom: 5pt; font-size: 10pt; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20pt; }}
  th {{ text-align: left; background: #f0f0f0; padding: 8pt; font-size: 10pt; color: #555; }}
  td {{ padding: 8pt; border-bottom: 1pt solid #f0f0f0; vertical-align: top; font-size: 10pt; }}
  .shift-time, .t-time {{ color: #888; font-family: monospace; width: 40pt; }}
  .shift-speaker {{ color: #999; text-align: right; }}
  .t-spk {{ font-weight: bold; width: 80pt; }}
  .t-text {{ color: #333; }}
  @page {{ size: A4; margin: 0; }}
</style>
</head>
<body>

<div class="page">
  <h1>Session Analysis Report</h1>
  <p class="meta">Session ID: {session.id} &nbsp;·&nbsp; Duration: {duration_str} &nbsp;·&nbsp; Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}</p>
  
  {overview_html}
  
  <h2>Key Speakers</h2>
  {speaker_cards}
</div>

<div class="page">
  {graph_html}
  
  <h2>Topic Timeline</h2>
  {shifts_html or "<p style='color:#aaa'>No topic shifts detected.</p>"}
</div>

<div class="page">
  <h2>Full Transcript</h2>
  <table>
    <thead><tr><th>Time</th><th>Speaker</th><th>Content</th></tr></thead>
    <tbody>
      {transcript_rows}
    </tbody>
  </table>
</div>

</body>
</html>"""
