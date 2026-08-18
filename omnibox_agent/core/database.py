"""MySQL read-only database connection via SQLAlchemy."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = get_config().mysql
        url = cfg.url
        log.info("MySQL connecting: %s:%s/%s", cfg.host, cfg.port, cfg.database)

        _engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        with _engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS ok"))
            row = result.fetchone()
            log.info("MySQL connection verified: %s", "ok" if row else "FAILED")
    return _engine


def get_session() -> Session:
    """Create a new SQLAlchemy session. Caller must close it."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager that provides a session with auto-commit/rollback/close.

    Issue #6: Ensures proper session lifecycle management.

    Usage:
        with session_scope() as session:
            result = session.execute(...)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
