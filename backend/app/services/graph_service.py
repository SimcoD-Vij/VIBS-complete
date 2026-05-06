"""
graph_service.py
================
Builds a knowledge graph from speaker summaries and topics.
Uses an LLM prompt to determine node/edge structure, then:
  - Validates the JSON output
  - Builds a NetworkX DiGraph
  - Evaluates graph quality (1-5 score)
  - Detects topic shift sequence
  - Saves everything to DB
  - Runs prompt improvement check every 50 sessions
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import networkx as nx

from app.config import settings
from app.services.nlp_service import get_llm_client

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

EVAL_PROMPT = """
Score this knowledge graph from a debate/meeting session on a scale of 1–5 for each criterion.
Return ONLY valid JSON. No markdown. No text before or after.

{{
  "completeness": <1-5>,
  "clarity": <1-5>,
  "connectivity": <1-5>,
  "hierarchy": <1-5>,
  "insight": <1-5>,
  "average": <float 1-5>,
  "reasoning": "one sentence"
}}

Criteria:
- completeness: are all speakers represented? are all major topics covered?
- clarity: do edge labels describe relationships clearly and concisely?
- connectivity: are there orphan nodes with no connections?
- hierarchy: do important topics have higher weight than minor ones?
- insight: does the graph reveal something non-obvious about how the session flowed?

Graph JSON:
{graph_json}
"""

IMPROVE_PROMPT = """
You are improving a prompt that generates knowledge graphs from debate/meeting transcripts.
Analyze what the top-performing graphs do well vs the worst-performing ones.
Rewrite the prompt to reliably produce graphs more like the top examples.
Keep the JSON schema IDENTICAL. Return only the new prompt text, nothing else.

Current prompt:
{current_prompt}

Top 5 highest-scoring graph examples (score >= 4.0):
{top_examples}

Bottom 5 lowest-scoring graph examples (score <= 2.5):
{worst_examples}
"""


# ── Load prompt ────────────────────────────────────────────────────────────────

def load_prompt(version: str = "v1") -> str:
    path = PROMPTS_DIR / f"graph_prompt_{version}.txt"
    if path.exists():
        return path.read_text()
    # Fallback minimal prompt
    return """Return a JSON knowledge graph with nodes and edges from these speaker summaries.
Speaker summaries: {speakers_json}
Topics: {topics_list}
Schema: {{"nodes":[{"id":"","label":"","type":"speaker|topic|claim","weight":1-10}],
         "edges":[{"source":"","target":"","relation":"argues|opposes|supports","strength":1-5,"label":""}]}}"""


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph_from_summaries(
    speaker_summaries: dict[str, str],
    topics: list[str],
    llm=None,
    retries: int = 2,
) -> Optional[dict]:
    """
    Call the LLM to generate a knowledge graph JSON.
    Returns {"nodes": [...], "edges": [...]} or None on failure.
    """
    if llm is None:
        llm = get_llm_client()

    if llm is None:
        logger.warning("No LLM — skipping graph generation")
        return _build_minimal_graph(speaker_summaries, topics)

    prompt_template = load_prompt(settings.PROMPT_VERSION if hasattr(settings, "PROMPT_VERSION") else "v1")
    prompt = prompt_template.format(
        speakers_json=json.dumps(speaker_summaries, indent=2),
        topics_list=", ".join(topics),
        max_nodes=settings.MAX_SPEAKERS * 3 if hasattr(settings, "MAX_SPEAKERS") else 25,
    )

    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            raw = re.sub(r"```json|```", "", raw).strip()

            graph_json = json.loads(raw)
            if _validate_graph(graph_json):
                return graph_json

        except json.JSONDecodeError:
            if attempt == 0:
                logger.warning("LLM returned invalid JSON, retrying with stricter prompt")
                prompt = prompt + "\n\nIMPORTANT: Return ONLY the JSON object. No markdown. No extra text."
            else:
                logger.error("LLM returned invalid JSON after retry")
        except Exception as e:
            logger.warning(f"Graph LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return _build_minimal_graph(speaker_summaries, topics)


def _validate_graph(g: dict) -> bool:
    """Check graph has required structure."""
    if not isinstance(g, dict):
        return False
    if "nodes" not in g or "edges" not in g:
        return False
    if not isinstance(g["nodes"], list) or not isinstance(g["edges"], list):
        return False
    return True


def _build_minimal_graph(speaker_summaries: dict, topics: list) -> dict:
    """Fallback graph when LLM is unavailable — just speakers and topics."""
    nodes = []
    edges = []
    for i, (spk, _) in enumerate(speaker_summaries.items()):
        nodes.append({"id": spk, "label": spk.replace("_", " ").title(), "type": "speaker", "weight": 7})

    for j, topic in enumerate(topics[:15]):
        tid = f"topic_{j}"
        nodes.append({"id": tid, "label": topic[:30], "type": "topic", "weight": 5})
        # Connect first speaker to all topics (minimal graph)
        if speaker_summaries:
            first_spk = list(speaker_summaries.keys())[0]
            edges.append({"source": first_spk, "target": tid, "relation": "introduces", "strength": 2, "label": "discussed"})

    return {"nodes": nodes, "edges": edges}


# ── NetworkX helper ────────────────────────────────────────────────────────────

def build_networkx_graph(graph_json: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph_json.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in graph_json.get("edges", []):
        G.add_edge(edge["source"], edge["target"], **{k: v for k, v in edge.items() if k not in ("source", "target")})
    return G


# ── Topic shift detection ──────────────────────────────────────────────────────

def detect_topic_shifts(graph_json: dict, segments: list[dict]) -> list[dict]:
    """
    Find topic shift events by matching shift edges to segment timestamps.
    Returns list of {time_seconds, from_topic, to_topic, speaker_label}
    """
    shifts = []
    node_map = {n["id"]: n for n in graph_json.get("nodes", [])}

    shift_edges = [
        e for e in graph_json.get("edges", [])
        if e.get("relation") == "shifts_to"
    ]

    for edge in shift_edges:
        from_node = node_map.get(edge["source"], {})
        to_node = node_map.get(edge["target"], {})
        from_label = from_node.get("label", edge["source"])
        to_label = to_node.get("label", edge["target"])

        # Find timestamp: first segment mentioning keywords from the to_node label
        to_keywords = set(to_label.lower().split())
        best_time = 0.0
        best_speaker = ""

        for seg in segments:
            text_words = set(seg.get("text", "").lower().split())
            if to_keywords & text_words:
                best_time = seg.get("start", 0.0)
                best_speaker = seg.get("speaker_label", seg.get("speaker", ""))
                break

        shifts.append({
            "time_seconds": best_time,
            "from_topic": from_label,
            "to_topic": to_label,
            "speaker_label": best_speaker,
        })

    return sorted(shifts, key=lambda x: x["time_seconds"])


# ── Graph evaluation ───────────────────────────────────────────────────────────

def evaluate_graph(graph_json: dict, llm=None) -> float:
    """Score the graph quality 1–5. Returns average score."""
    if llm is None:
        llm = get_llm_client()
    if llm is None:
        return 3.0  # neutral default

    try:
        prompt = EVAL_PROMPT.format(graph_json=json.dumps(graph_json)[:3000])
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return float(result.get("average", 3.0))
    except Exception as e:
        logger.warning(f"Graph eval failed: {e}")
        return 3.0


# ── Graph explanation ──────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """
You are analyzing a meeting/debate knowledge graph.
Write a 3–4 sentence explanation of what this graph reveals about:
1. How the discussion flowed between speakers
2. Which topics were central vs peripheral
3. Whether there was consensus or conflict

Be specific — name actual speakers and topics from the data.
Write in plain language, no jargon.

Speakers: {speakers}
Graph summary (nodes and edges): {graph_summary}
"""


def explain_graph(graph_json: dict, speaker_names: list[str], llm=None) -> str:
    if llm is None:
        llm = get_llm_client()
    if llm is None:
        return ""

    try:
        node_summary = [f"{n['label']} ({n['type']})" for n in graph_json.get("nodes", [])[:15]]
        edge_summary = [f"{e['source']} —[{e['relation']}]→ {e['target']}" for e in graph_json.get("edges", [])[:20]]
        graph_summary = "Nodes: " + ", ".join(node_summary) + "\nEdges: " + "; ".join(edge_summary)

        prompt = EXPLAIN_PROMPT.format(
            speakers=", ".join(speaker_names),
            graph_summary=graph_summary,
        )
        response = llm.invoke(prompt)
        return (response.content if hasattr(response, "content") else str(response)).strip()
    except Exception as e:
        logger.warning(f"Graph explain failed: {e}")
        return ""


# ── Save to DB ────────────────────────────────────────────────────────────────

def save_graph_to_db(session_id: str, graph_json: dict, explanation: str, eval_score: float, db) -> None:
    """Insert or update GraphData record (sync — for Celery workers)."""
    from sqlalchemy.orm import Session
    from app.models.db_models import GraphData

    existing = db.query(GraphData).filter(GraphData.session_id == session_id).first()
    version = "v1"

    if existing:
        existing.nodes_json = json.dumps(graph_json.get("nodes", []))
        existing.edges_json = json.dumps(graph_json.get("edges", []))
        existing.explanation = explanation
        existing.eval_score = eval_score
        existing.prompt_version = version
    else:
        record = GraphData(
            session_id=session_id,
            nodes_json=json.dumps(graph_json.get("nodes", [])),
            edges_json=json.dumps(graph_json.get("edges", [])),
            explanation=explanation,
            eval_score=eval_score,
            prompt_version=version,
        )
        db.add(record)
    db.commit()


# ── Prompt improvement loop ────────────────────────────────────────────────────

def run_improvement_check(db) -> None:
    """
    Every 50 completed sessions, rewrite the graph prompt using the
    best and worst examples collected so far.
    """
    from app.models.db_models import Session as SessionModel, GraphData

    try:
        count = db.query(SessionModel).filter(SessionModel.status == "complete").count()
        if count == 0 or count % 50 != 0:
            return

        llm = get_llm_client()
        if llm is None:
            return

        # Fetch top 5 and bottom 5
        top = db.query(GraphData).order_by(GraphData.eval_score.desc()).limit(5).all()
        bottom = db.query(GraphData).order_by(GraphData.eval_score.asc()).limit(5).all()

        if not top or not bottom:
            return

        current_prompt = load_prompt("v1")

        top_examples = "\n---\n".join(
            f"Score {r.eval_score:.1f}: nodes={r.nodes_json[:300]}"
            for r in top if r.eval_score
        )
        worst_examples = "\n---\n".join(
            f"Score {r.eval_score:.1f}: nodes={r.nodes_json[:300]}"
            for r in bottom if r.eval_score
        )

        prompt = IMPROVE_PROMPT.format(
            current_prompt=current_prompt,
            top_examples=top_examples,
            worst_examples=worst_examples,
        )

        response = llm.invoke(prompt)
        new_prompt_text = (response.content if hasattr(response, "content") else str(response)).strip()

        # Find next version number
        existing = sorted(PROMPTS_DIR.glob("graph_prompt_v*.txt"))
        next_n = len(existing) + 1
        new_path = PROMPTS_DIR / f"graph_prompt_v{next_n}.txt"
        new_path.write_text(new_prompt_text)
        logger.info(f"Prompt improved: saved to {new_path.name} after {count} sessions")

    except Exception as e:
        logger.warning(f"Prompt improvement check failed: {e}")
