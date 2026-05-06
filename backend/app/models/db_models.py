import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime, Enum, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum


class Base(DeclarativeBase):
    pass


class SessionStatus(str, enum.Enum):
    recording = "recording"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String, default="recording")
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    wav_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    speaker_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    topics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    segments: Mapped[list["Segment"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    speakers: Mapped[list["Speaker"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    speaker_label: Mapped[str] = mapped_column(String)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    is_overlap: Mapped[bool] = mapped_column(Boolean, default=False)
    session: Mapped["Session"] = relationship(back_populates="segments")


class Speaker(Base):
    __tablename__ = "speakers"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    speaker_label: Mapped[str] = mapped_column(String)
    display_name: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    total_seconds: Mapped[float] = mapped_column(Float, default=0)
    talk_share: Mapped[float] = mapped_column(Float, default=0)
    color: Mapped[str] = mapped_column(String, default="#888780")
    session: Mapped["Session"] = relationship(back_populates="speakers")


class GraphData(Base):
    __tablename__ = "graph_data"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), unique=True)
    nodes_json: Mapped[str] = mapped_column(Text)
    edges_json: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str] = mapped_column(String, default="v1")
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class TopicShift(Base):
    __tablename__ = "topic_shifts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    time_seconds: Mapped[float] = mapped_column(Float)
    from_topic: Mapped[str] = mapped_column(String)
    to_topic: Mapped[str] = mapped_column(String)
    speaker_label: Mapped[str] = mapped_column(String)
