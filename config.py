"""
MASS: Multi-Agent Stock System - 全局配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()

# 数据目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 日志目录
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# SQLite 数据库路径
DATABASE_PATH = DATA_DIR / "mass.db"

# Flask 配置
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

# ───────────────────────────────────────────────
# LLM API 配置 — DeepSeek 为默认提供商
# ───────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek / openai / claude / ollama

# DeepSeek 配置（默认主源）— 密钥必须通过环境变量或 Web UI 配置，不再硬编码
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# OpenAI 配置（备选）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Claude 配置（备选）
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")

# Ollama 配置（本地）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 默认模型 — DeepSeek-V4-Pro
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-v4-pro")
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "deepseek-v4-pro")
PREDICTION_MODEL = os.getenv("PREDICTION_MODEL", "deepseek-v4-pro")
FALLBACK_PREDICTION_MODEL = os.getenv("FALLBACK_PREDICTION_MODEL", "")  # 备模型，为空则不启用

# 预测全局默认参数
PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.6"))
PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "300"))

# LLM 调用参数（全部可调）
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

# 缓存配置
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5分钟

# Agent 配置
AGENT_PARALLEL = os.getenv("AGENT_PARALLEL", "True").lower() == "true"
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "6"))

# Agent 间通信（两轮分析）配置
AGENT_INTER_COMMUNICATION = os.getenv("AGENT_INTER_COMMUNICATION", "True").lower() == "true"
AGENT_REVISION_THRESHOLD = float(os.getenv("AGENT_REVISION_THRESHOLD", "0.50"))
AGENT_REVISION_MAX = int(os.getenv("AGENT_REVISION_MAX", "2"))

# 自适应并发控制配置
ADAPTIVE_CONCURRENCY = os.getenv("ADAPTIVE_CONCURRENCY", "True").lower() == "true"
CONCURRENCY_MIN = int(os.getenv("CONCURRENCY_MIN", "2"))
CONCURRENCY_MAX = int(os.getenv("CONCURRENCY_MAX", "10"))
CONCURRENCY_SLOW_THRESHOLD = float(os.getenv("CONCURRENCY_SLOW_THRESHOLD", "30.0"))  # 响应时间>此值降级
CONCURRENCY_FAST_THRESHOLD = float(os.getenv("CONCURRENCY_FAST_THRESHOLD", "10.0"))  # 响应时间<此值允许升级
CONCURRENCY_ERROR_HIGH = float(os.getenv("CONCURRENCY_ERROR_HIGH", "0.20"))         # 错误率>此值大幅降级
CONCURRENCY_ERROR_LOW = float(os.getenv("CONCURRENCY_ERROR_LOW", "0.05"))           # 错误率<此值才允许升级
CONCURRENCY_WINDOW_SIZE = int(os.getenv("CONCURRENCY_WINDOW_SIZE", "50"))           # 滑动窗口大小

# 动态权重映射
WEIGHT_MAP = {
    "bull_trend":   {"TA": 0.30, "FA": 0.15, "CA": 0.25, "SA": 0.15, "MA": 0.10, "RA": 0.05},
    "bull_value":   {"TA": 0.15, "FA": 0.35, "CA": 0.20, "SA": 0.10, "MA": 0.10, "RA": 0.10},
    "bear_defense": {"TA": 0.10, "FA": 0.20, "CA": 0.15, "SA": 0.15, "MA": 0.10, "RA": 0.30},
    "oscillation":  {"TA": 0.25, "FA": 0.15, "CA": 0.25, "SA": 0.20, "MA": 0.05, "RA": 0.10},
}

# 市场周期映射到权重方案
CYCLE_WEIGHT_MAP = {
    "复苏早期": "bull_trend",
    "复苏晚期": "bull_value",
    "过热": "bull_trend",
    "滞胀": "bear_defense",
    "衰退早期": "bear_defense",
    "衰退晚期": "bull_value",
}

# 回测配置
VALIDATION_LOOKAHEAD_DAYS = int(os.getenv("VALIDATION_LOOKAHEAD_DAYS", "20"))

# 动态权重学习配置
DYNAMIC_WEIGHT_ENABLED = os.getenv("DYNAMIC_WEIGHT_ENABLED", "True").lower() == "true"
ACCURACY_HISTORY_MAXLEN = int(os.getenv("ACCURACY_HISTORY_MAXLEN", "100"))
ACCURACY_MIN_SAMPLES = int(os.getenv("ACCURACY_MIN_SAMPLES", "20"))

# 非线性置信度加权配置
NONLINEAR_CONFIDENCE_ENABLED = os.getenv("NONLINEAR_CONFIDENCE_ENABLED", "True").lower() == "true"
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.8"))
CONFIDENCE_LOW_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_THRESHOLD", "0.5"))
CONFIDENCE_HIGH_EXPONENT = float(os.getenv("CONFIDENCE_HIGH_EXPONENT", "0.5"))
CONFIDENCE_LOW_EXPONENT = float(os.getenv("CONFIDENCE_LOW_EXPONENT", "2.0"))

# ══════════════════════════════════════════════════════════════════════
# LLM 调用优化配置 — Agent 缓存 / Prompt 压缩 / 模型分层 / 批量推理
# ══════════════════════════════════════════════════════════════════════

# Agent 结论缓存：同股票同参数 30s 内复用
AGENT_CACHE_ENABLED = os.getenv("AGENT_CACHE_ENABLED", "True").lower() == "true"
AGENT_CACHE_TTL = int(os.getenv("AGENT_CACHE_TTL", "30"))

# Prompt 压缩：仅保留每个 Agent 关注的核心指标
PROMPT_COMPRESSION_ENABLED = os.getenv("PROMPT_COMPRESSION_ENABLED", "True").lower() == "true"

# 模型分层：Agent 用轻量模型，Chairman 用重型模型
AGENT_LIGHT_MODEL = os.getenv("AGENT_LIGHT_MODEL", "deepseek-chat")
AGENT_HEAVY_MODEL = os.getenv("AGENT_HEAVY_MODEL", "deepseek-v4-pro")

AGENT_MODEL_MAP = {
    "TA-Agent": AGENT_LIGHT_MODEL,
    "FA-Agent": AGENT_HEAVY_MODEL,
    "CA-Agent": AGENT_LIGHT_MODEL,
    "SA-Agent": AGENT_LIGHT_MODEL,
    "MA-Agent": AGENT_HEAVY_MODEL,
    "RA-Agent": AGENT_LIGHT_MODEL,
}

# 批量推理：6 次独立调用 → 2 次批量调用（每批 3 个 Agent）
BATCH_INFERENCE_ENABLED = os.getenv("BATCH_INFERENCE_ENABLED", "False").lower() == "true"
BATCH_INFERENCE_SIZE = int(os.getenv("BATCH_INFERENCE_SIZE", "3"))

# TuShare 配置
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# Redis 黑板配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS_BLACKBOARD = os.getenv("USE_REDIS_BLACKBOARD", "False").lower() == "true"

# 爬虫配置
CRAWLER_REQUEST_INTERVAL = float(os.getenv("CRAWLER_REQUEST_INTERVAL", "0.5"))
CRAWLER_MAX_RETRIES = int(os.getenv("CRAWLER_MAX_RETRIES", "3"))
CRAWLER_TIMEOUT = int(os.getenv("CRAWLER_TIMEOUT", "30"))
CRAWLER_ENABLE_EASTMONEY = os.getenv("CRAWLER_ENABLE_EASTMONEY", "True").lower() == "true"
CRAWLER_ENABLE_THS = os.getenv("CRAWLER_ENABLE_THS", "True").lower() == "true"

# ══════════════════════════════════════════════════════════════════════
# 运行时配置校验 — 确保非 Mock 模式下必须配置 API Key
# ══════════════════════════════════════════════════════════════════════
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "True").lower() == "true"

if not USE_MOCK_LLM and not DEEPSEEK_API_KEY and not OPENAI_API_KEY and not CLAUDE_API_KEY:
    import warnings
    warnings.warn(
        "未配置任何 LLM API 密钥，系统将自动启用 Mock 模式。"
        "请在 .env 文件中配置密钥，或通过 Web 界面设置。",
        RuntimeWarning,
        stacklevel=2,
    )

# 免责声明
DISCLAIMER = """
免责声明：本系统所有Agent输出仅供参考，不构成投资建议。
股市有风险，投资需谨慎。请根据自身风险承受能力做出决策。
"""
