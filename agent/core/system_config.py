"""
MASS 系统配置管理器
支持从数据库读取配置，覆盖环境变量默认值
解决硬编码 API Key 问题，提供 Web UI 配置渠道
"""
import os
import json
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from loguru import logger


@dataclass
class LLMRuntimeConfig:
    """LLM 运行时配置 — 可从数据库热更新"""
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    use_mock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMRuntimeConfig":
        # 过滤掉不支持的字段
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def mask_api_key(self) -> str:
        """返回脱敏的 API Key 用于前端展示"""
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "****"
        # 保留原始长度，避免用户误以为被截断
        return self.api_key[:4] + "•" * (len(self.api_key) - 8) + self.api_key[-4:]


class SystemConfigManager:
    """
    系统配置管理器 — 线程安全单例
    
    配置优先级：数据库 > 环境变量 > 代码默认值
    
    使用方式:
        from agent.core.system_config import SystemConfigManager
        cfg = SystemConfigManager().get_llm_config()
        # cfg.api_key 将自动从数据库读取（如果已设置）
    """

    _instance = None
    _lock = threading.Lock()

    # 配置项的 key 常量
    KEY_LLM_CONFIG = "llm_config"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._db = None
        self._load_defaults()

    def _get_db(self):
        """延迟初始化数据库连接"""
        if self._db is None:
            from agent.models.database import Database
            self._db = Database()
        return self._db

    def _load_defaults(self):
        """从环境变量加载默认配置到缓存"""
        # 读取环境变量，不再使用硬编码密钥
        self._cache[self.KEY_LLM_CONFIG] = LLMRuntimeConfig(
            provider=os.getenv("LLM_PROVIDER", "deepseek"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEFAULT_MODEL", "deepseek-v4-pro"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")),
            presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", "0.0")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            use_mock=os.getenv("USE_MOCK_LLM", "True").lower() == "true",
        )

    def _load_from_db(self) -> Optional[Dict[str, Any]]:
        """从数据库加载配置"""
        try:
            conn = self._get_db()._get_connection()
            row = conn.execute(
                "SELECT config_value FROM system_settings WHERE config_key = ?",
                (self.KEY_LLM_CONFIG,),
            ).fetchone()
            if row and row["config_value"]:
                return json.loads(row["config_value"])
        except Exception as e:
            logger.debug(f"从数据库加载配置失败（可能表尚未创建）: {e}")
        return None

    def get_llm_config(self) -> LLMRuntimeConfig:
        """
        获取 LLM 运行时配置
        
        优先级：数据库 > 环境变量 > 默认值
        """
        with self._cache_lock:
            # 尝试从数据库加载（支持热更新）
            db_config = self._load_from_db()
            if db_config:
                try:
                    return LLMRuntimeConfig.from_dict(db_config)
                except Exception as e:
                    logger.warning(f"数据库配置解析失败，使用环境变量: {e}")
            # 回退到缓存（环境变量/默认值）
            return self._cache.get(self.KEY_LLM_CONFIG, LLMRuntimeConfig())

    def save_llm_config(self, config: LLMRuntimeConfig) -> bool:
        """
        保存 LLM 配置到数据库
        
        Args:
            config: 新的 LLM 配置
            
        Returns:
            是否保存成功
        """
        try:
            conn = self._get_db()._get_connection()
            config_dict = config.to_dict()
            conn.execute(
                """
                INSERT INTO system_settings (config_key, config_value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(config_key) DO UPDATE SET
                    config_value = excluded.config_value,
                    updated_at = excluded.updated_at
                """,
                (self.KEY_LLM_CONFIG, json.dumps(config_dict, ensure_ascii=False)),
            )
            conn.commit()

            # 更新缓存
            with self._cache_lock:
                self._cache[self.KEY_LLM_CONFIG] = config

            logger.info(f"LLM 配置已保存: provider={config.provider}, model={config.model}, use_mock={config.use_mock}")
            return True
        except Exception as e:
            logger.error(f"保存 LLM 配置失败: {e}")
            return False

    def get_config_for_display(self) -> Dict[str, Any]:
        """
        获取用于前端展示的配置（API Key 脱敏）
        """
        cfg = self.get_llm_config()
        return {
            "provider": cfg.provider,
            "api_key": cfg.mask_api_key(),
            "base_url": cfg.base_url,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "frequency_penalty": cfg.frequency_penalty,
            "presence_penalty": cfg.presence_penalty,
            "timeout": cfg.timeout,
            "max_retries": cfg.max_retries,
            "use_mock": cfg.use_mock,
            "has_api_key": bool(cfg.api_key),
        }

    def validate_config(self, config: LLMRuntimeConfig) -> tuple[bool, str]:
        """
        验证配置是否有效
        
        Returns:
            (是否有效, 错误信息)
        """
        if config.use_mock:
            return True, ""

        if not config.api_key:
            return False, "非 Mock 模式下 API Key 不能为空"

        if not config.base_url:
            return False, "Base URL 不能为空"

        if not config.model:
            return False, "模型名称不能为空"

        if config.provider not in ("deepseek", "openai", "claude", "ollama"):
            return False, f"不支持的提供商: {config.provider}"

        return True, ""


# 全局便捷函数
def get_system_config() -> SystemConfigManager:
    """获取系统配置管理器实例"""
    return SystemConfigManager()
