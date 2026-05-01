"""
调度模块 ORM 模型
SQLAlchemy 2.0+ 风格
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, Integer, DateTime, Boolean, create_engine, ForeignKey, REAL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

from config.database import DatabaseConfig


class Base(DeclarativeBase):
    pass


# ============================================================================
# 调度器相关表
# ============================================================================

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


# ============================================================================
# 用户管理相关表
# ============================================================================

class User(Base):
    """用户表"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # admin / user
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserSetting(Base):
    """用户设置表"""

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)
    setting_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserStockNote(Base):
    """用户股票标记表"""

    __tablename__ = "user_stock_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    ts_code: Mapped[str] = mapped_column(String(20), nullable=False)
    note_type: Mapped[str] = mapped_column(String(20), nullable=False)  # watched / researched / excluded
    prediction_date: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPosition(Base):
    """用户模拟持仓表"""

    __tablename__ = "user_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    ts_code: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    buy_price: Mapped[float] = mapped_column(REAL, nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    buy_date: Mapped[str] = mapped_column(String(8), nullable=False)
    stop_loss_price: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    strategy_tag: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="holding")  # holding / sold
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPositionHistory(Base):
    """用户历史交易记录表"""

    __tablename__ = "user_position_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    position_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_code: Mapped[str] = mapped_column(String(20), nullable=False)
    buy_price: Mapped[float] = mapped_column(REAL, nullable=False)
    sell_price: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    sell_date: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    pnl_amount: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    pnl_pct: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine():
    """获取 SQLAlchemy engine（复用现有数据库配置）"""
    db_url = DatabaseConfig.get_database_url()
    return create_engine(db_url, echo=False, pool_pre_ping=True)


def get_session_factory():
    """获取 session maker"""
    return sessionmaker(bind=get_engine())


def init_db():
    """初始化所有数据库表（如果不存在则创建）"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _init_admin_user()


def _init_admin_user():
    """初始化 admin 账号（如果不存在）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        from sqlalchemy import select
        stmt = select(User).where(User.username == "admin")
        existing = session.execute(stmt).scalar_one_or_none()
        if not existing:
            import bcrypt
            password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt(rounds=12)).decode()
            admin = User(
                username="admin",
                password_hash=password_hash,
                display_name="管理员",
                role="admin",
                is_active=True,
            )
            session.add(admin)
            session.commit()
            print("[Init] admin 账号已创建（密码: admin123）")
