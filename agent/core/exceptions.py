"""
MASS 统一异常体系
"""


class MASSException(Exception):
    """MASS 基础异常"""
    code = "INTERNAL_ERROR"
    status_code = 500
    
    def __init__(self, message: str = "", detail: dict = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class AgentError(MASSException):
    """Agent 分析异常"""
    code = "AGENT_ERROR"
    status_code = 500


class DataError(MASSException):
    """数据获取/处理异常"""
    code = "DATA_ERROR"
    status_code = 500


class LLMError(MASSException):
    """LLM 调用异常"""
    code = "LLM_ERROR"
    status_code = 503


class ValidationError(MASSException):
    """输入校验异常"""
    code = "VALIDATION_ERROR"
    status_code = 400


class NotFoundError(MASSException):
    """资源不存在"""
    code = "NOT_FOUND"
    status_code = 404


class RateLimitError(MASSException):
    """请求限流"""
    code = "RATE_LIMIT"
    status_code = 429
