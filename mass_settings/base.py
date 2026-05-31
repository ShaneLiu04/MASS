"""
MASS 基础配置 - Pydantic Settings v2
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """MASS 全局配置"""
    
    VERSION: str = "2.0.0"
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()
    
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = True
    
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gpt-4o"
    LLM_TIMEOUT: int = 60
    
    USE_MOCK_LLM: bool = Field(default_factory=lambda: os.getenv('USE_MOCK_LLM', 'True').lower() == 'true')
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
