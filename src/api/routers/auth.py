"""
认证模块 API
提供登录、登出、获取当前用户、修改密码
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

from src.scheduler.models import get_session_factory, User

router = APIRouter()

# JWT 配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "aiquant-dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


class UserInfo(BaseModel):
    id: int
    username: str
    display_name: Optional[str]
    role: str


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(user_id: int, username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """从 JWT Token 中解析当前用户"""
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证信息")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="认证格式错误")

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的认证信息")

    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
        return user


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[User]:
    """可选认证：解析当前用户，未登录返回 None"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
    except Exception:
        return None
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            return None
        return user


def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """要求当前用户为管理员"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """用户登录"""
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.query(User).filter(User.username == req.username).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="账号或密码错误")
        if not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="账号或密码错误")

        token = create_access_token(user.id, user.username, user.role)
        return LoginResponse(
            token=token,
            user={
                "id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "role": user.role,
            },
        )


@router.post("/logout")
async def logout():
    """登出（前端清除token即可，服务端无状态）"""
    return {"message": "登出成功"}


@router.get("/me", response_model=UserInfo)
async def get_me(user: User = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return UserInfo(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
    )


@router.put("/password")
async def change_password(req: PasswordChangeRequest, user: User = Depends(get_current_user)):
    """修改密码"""
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少需要6位")

    new_hash = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt(rounds=12)).decode()
    session_factory = get_session_factory()
    with session_factory() as session:
        db_user = session.query(User).filter(User.id == user.id).first()
        db_user.password_hash = new_hash
        db_user.updated_at = datetime.utcnow()
        session.commit()

    return {"message": "密码修改成功"}
