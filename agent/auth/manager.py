"""
MASS 用户认证管理器
支持 session-based 认证，管理员一键登录
"""
import functools
from typing import Optional, Dict, Any
from datetime import datetime

from flask import session, redirect, url_for, request


class AuthManager:
    """
    用户认证管理器
    
    默认管理员账号:
    - username: admin
    - password: ********
    """
    
    DEFAULT_USERS = {
        "admin": {
            "username": "admin",
            "password": "********",
            "display_name": "管理员",
            "role": "admin",
            "avatar": "admin",
            "created_at": "2024-01-01",
        },
        "demo": {
            "username": "demo",
            "password": "********",
            "display_name": "演示用户",
            "role": "user",
            "avatar": "demo",
            "created_at": "2024-01-01",
        },
    }
    
    SESSION_KEY = "mass_user"

    @classmethod
    def authenticate(cls, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户名密码"""
        user = cls.DEFAULT_USERS.get(username)
        if user and user["password"] == password:
            return {k: v for k, v in user.items() if k != "password"}
        return None

    @classmethod
    def login(cls, username: str) -> None:
        """创建session登录状态"""
        user = cls.DEFAULT_USERS.get(username)
        if user:
            session[cls.SESSION_KEY] = {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": user["role"],
                "avatar": user["avatar"],
                "login_at": datetime.now().isoformat(),
            }
            session.permanent = True

    @classmethod
    def logout(cls) -> None:
        """清除session"""
        session.pop(cls.SESSION_KEY, None)

    @classmethod
    def get_current_user(cls) -> Optional[Dict[str, Any]]:
        """获取当前登录用户"""
        return session.get(cls.SESSION_KEY)

    @classmethod
    def is_authenticated(cls) -> bool:
        """检查是否已登录"""
        return cls.SESSION_KEY in session and session[cls.SESSION_KEY] is not None

    @classmethod
    def is_admin(cls) -> bool:
        """检查是否是管理员"""
        user = cls.get_current_user()
        return user is not None and user.get("role") == "admin"


def require_login(f):
    """登录保护装饰器"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not AuthManager.is_authenticated():
            if request.is_json or request.path.startswith("/api/"):
                from flask import jsonify
                return jsonify({"error": "未登录", "code": "UNAUTHORIZED"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function
