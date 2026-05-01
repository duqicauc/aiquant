"""
管理员模块 API
提供用户列表、创建用户、修改用户、删除用户
"""
import bcrypt
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.routers.auth import get_current_admin
from src.scheduler.models import get_session_factory, User

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    role: str = "user"


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserItem(BaseModel):
    id: int
    username: str
    display_name: Optional[str]
    role: str
    is_active: bool
    created_at: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/users", response_model=List[UserItem])
async def list_users(admin: User = Depends(get_current_admin)):
    """获取用户列表"""
    session_factory = get_session_factory()
    with session_factory() as session:
        users = session.query(User).all()
        return [
            UserItem(
                id=u.id,
                username=u.username,
                display_name=u.display_name,
                role=u.role,
                is_active=u.is_active,
                created_at=u.created_at.isoformat() if u.created_at else None,
            )
            for u in users
        ]


@router.post("/users")
async def create_user(req: UserCreate, admin: User = Depends(get_current_admin)):
    """创建用户"""
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少需要6位")

    session_factory = get_session_factory()
    with session_factory() as session:
        existing = session.query(User).filter(User.username == req.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="用户名已存在")

        password_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt(rounds=12)).decode()
        user = User(
            username=req.username,
            password_hash=password_hash,
            display_name=req.display_name or req.username,
            role=req.role,
            is_active=True,
        )
        session.add(user)
        session.commit()
        return {"message": "用户创建成功", "user_id": user.id}


@router.put("/users/{user_id}")
async def update_user(user_id: int, req: UserUpdate, admin: User = Depends(get_current_admin)):
    """修改用户信息"""
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        if req.display_name is not None:
            user.display_name = req.display_name
        if req.role is not None:
            user.role = req.role
        if req.is_active is not None:
            user.is_active = req.is_active
        user.updated_at = datetime.utcnow()
        session.commit()
        return {"message": "用户更新成功"}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, admin: User = Depends(get_current_admin)):
    """删除用户（软删除：禁用账号）"""
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        if user.id == admin.id:
            raise HTTPException(status_code=400, detail="不能删除自己")

        user.is_active = False
        user.updated_at = datetime.utcnow()
        session.commit()
        return {"message": "用户已禁用"}
