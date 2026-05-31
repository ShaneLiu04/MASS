"""
MASS 用户认证模块
"""
from .manager import AuthManager, require_login

__all__ = ["AuthManager", "require_login"]
