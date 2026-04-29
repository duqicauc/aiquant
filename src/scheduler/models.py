"""
调度模块 ORM 模型
SQLAlchemy 2.0+ 风格
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, Integer, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

from config.database import DatabaseConfig


class Base(DeclarativeBase):
    pass


class JobHistory(Base):
    """任务执行历史记录"""

    __tablename__ = "scheduler_job_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    job_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    trigger_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    run_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running / success / failed
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stdout: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stderr: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exception: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class JobLog(Base):
    """任务执行逐行日志"""

    __tablename__ = "scheduler_job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    history_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    level: Mapped[str] = mapped_column(String(20), default="INFO")  # INFO / WARNING / ERROR
    message: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine():
    """获取 SQLAlchemy engine（复用现有数据库配置）"""
    db_url = DatabaseConfig.get_database_url()
    return create_engine(db_url, echo=False, pool_pre_ping=True)


def get_session_factory():
    """获取 session maker"""
    return sessionmaker(bind=get_engine())


def init_db():
    """初始化调度器相关表（如果不存在则创建）"""
    engine = get_engine()
    Base.metadata.create_all(engine)
