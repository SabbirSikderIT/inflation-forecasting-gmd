"""SQLite persistence for prediction history."""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DB_PATH = Path(__file__).resolve().parent / "inflation.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_used: Mapped[str] = mapped_column(String)
    predicted_inflation: Mapped[float] = mapped_column(Float)
    all_predictions: Mapped[dict] = mapped_column(JSON)
    filename: Mapped[str] = mapped_column(String, default="")
    last_year: Mapped[int] = mapped_column(Integer)
    input_data: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
