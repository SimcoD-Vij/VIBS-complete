"""
nlp_service.py
==============
LLM-powered NLP operations:
  - Per-speaker bullet-point summarization
  - Topic/entity extraction via spaCy
  - LLM client factory (Groq / OpenAI / Anthropic / Ollama)
"""
import logging
import time
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)


# ── LLM client factory ────────────────────────────────────────────────────────

def get_llm_client():
    """
    Return a LangChain LLM based on settings.LLM_PROVIDER.
    Returns None if no provider is configured.
    """
    provider = settings.LLM_PROVIDER.lower()

    try:
        if provider == "groq" and settings.GROQ_API_KEY:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
                temperature=0.2,
                max_tokens=2048,
            )

        elif provider == "openai" and settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
                temperature=0.2,
                max_tokens=2048,
            )

        elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.2,
                max_tokens=2048,
            )

        elif provider == "ollama":
            from langchain_community.llms import Ollama
            base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
            model = getattr(settings, "OLLAMA_MODEL", "llama3.2")
            return Ollama(model=model, base_url=base_url, temperature=0.2)

    except ImportError as e:
        logger.warning(f"LLM provider {provider} unavailable (missing package): {e}")
    except Exception as e:
        logger.warning(f"LLM client init failed: {e}")

    logger.warning("No LLM configured — summarization will be skipped")
    return None


# ── Summarization ─────────────────────────────────────────────────────────────

SUMMARIZE_PROMPT = """You are analyzing a debate or meeting transcript.
Below is everything one speaker said throughout the session.
Extract their 3–5 key arguments, opinions, or points as bullet points.
Each bullet must be one concise sentence. No padding, no filler.
If they said very little, write fewer bullets.

SPEAKER TEXT:
{speaker_text}

Return ONLY the bullet points, one per line, starting with "•".
"""


def summarize_speaker(speaker_text: str, llm=None, retries: int = 2) -> str:
    """
    Generate a bullet-point summary of what a speaker said.
    Returns a string of bullet points.
    Falls back to a truncated excerpt if LLM fails or isn't configured.
    """
    if not speaker_text or len(speaker_text.strip()) < 30:
        return "• (Speaker said very little)"

    if llm is None:
        llm = get_llm_client()

    if llm is None:
        # No LLM — return first 300 chars as fallback
        excerpt = speaker_text[:300].strip()
        if len(speaker_text) > 300:
            excerpt += "..."
        return f"• {excerpt}"

    prompt = SUMMARIZE_PROMPT.format(speaker_text=speaker_text[:4000])  # cap token input

    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip()
        except Exception as e:
            logger.warning(f"Summarization attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return f"• {speaker_text[:200].strip()}..."


# ── Topic extraction (spaCy) ──────────────────────────────────────────────────

# Generic noun chunks to exclude from topic list
TOPIC_STOPLIST = {
    "i", "we", "you", "he", "she", "they", "it", "that", "this",
    "thing", "something", "everything", "anything", "way", "time",
    "lot", "point", "fact", "kind", "part", "side", "end", "case",
    "question", "answer", "problem", "issue", "people", "person",
    "word", "comment", "moment", "sense",
}


def extract_topics(full_text: str, max_topics: int = 25) -> list[str]:
    """
    Extract named entities and significant noun phrases from the full transcript.
    Uses spaCy. Falls back to simple word frequency if spaCy not available.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(full_text[:50000])  # cap for speed

        # Named entities
        entities = list({
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in ("ORG", "PERSON", "GPE", "LAW", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART")
            and len(ent.text.strip()) > 2
        })

        # Meaningful noun chunks (2+ words, not in stoplist)
        noun_phrases = list({
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) >= 2
            and chunk.root.pos_ == "NOUN"
            and chunk.text.lower().strip() not in TOPIC_STOPLIST
            and len(chunk.text.strip()) > 4
        })

        combined = list(dict.fromkeys(entities + noun_phrases))
        return combined[:max_topics]

    except Exception as e:
        logger.warning(f"spaCy topic extraction failed: {e}. Using fallback.")
        return _simple_topic_extraction(full_text, max_topics)


def _simple_topic_extraction(text: str, max_topics: int) -> list[str]:
    """Word frequency fallback when spaCy is unavailable."""
    import re
    from collections import Counter

    words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)  # capitalized words
    freq = Counter(words)
    stopwords = {"The", "This", "That", "When", "What", "Where", "Which", "With", "From"}
    topics = [w for w, _ in freq.most_common(50) if w not in stopwords]
    return topics[:max_topics]
