# MASS v2.3 技术白皮书

> **Multi-Agent Stock System — 多智能体协同股票投研与决策系统**
> 版本：v2.3 | 日期：2026-05-27 | 测试覆盖：141 项 | 通过率：100%

---

## 目录

1. [项目概览](#1-项目概览)
2. [系统架构深度剖析](#2-系统架构深度剖析)
3. [多智能体协作原理](#3-多智能体协作原理)
4. [LLM 集成与 Prompt 工程](#4-llm-集成与-prompt-工程)
5. [数据层设计详解](#5-数据层设计详解)
6. [决策引擎数学原理](#6-决策引擎数学原理)
7. [性能优化全景](#7-性能优化全景)
8. [前端架构与流式渲染](#8-前端架构与流式渲染)
9. [工程实践与设计模式](#9-工程实践与设计模式)
10. [API 协议规范](#10-api-协议规范)
11. [测试策略与质量保障](#11-测试策略与质量保障)
12. [部署运维手册](#12-部署运维手册)
13. [附录](#13-附录)

---

## 1. 项目概览

### 1.1 项目定位

MASS 是一个**以大语言模型为推理引擎、多智能体协同为决策架构的股票投研系统**。它模拟真实投资委员会的运作机制：6 位具备不同专业能力的分析师（Agent）独立研判后，由投资委员会主席（Chairman）综合各方观点、仲裁冲突、形成最终投资决策。

```
真实投资委员会                  MASS 系统映射
┌──────────────────┐          ┌──────────────────┐
│ 技术分析师        │   ←→    │ TA-Agent          │  K线/均线/指标
│ 基本面研究员      │   ←→    │ FA-Agent          │  PE/PB/ROE/估值
│ 资金流向分析师    │   ←→    │ CA-Agent          │  主力/北向/筹码
│ 市场情绪观察员    │   ←→    │ SA-Agent          │  舆情/crowd情绪
│ 宏观策略师        │   ←→    │ MA-Agent          │  周期/政策/利率
│ 风控官            │   ←→    │ RA-Agent          │  波动率/VaR/仓位
│ 投资委员会主席    │   ←→    │ Chairman          │  综合决策/仲裁
└──────────────────┘          └──────────────────┘
```

### 1.2 解决的行业痛点

| 痛点 | 严重度 | MASS 解决方案 | 技术手段 |
|------|--------|-------------|---------|
| 人工投研覆盖维度有限，难以同时关注 6 个维度 | ***** | 6 Agent 并行分析，每人专精一个维度 | `ThreadPoolExecutor(6)` + LLM role specialization |
| 单一 LLM 输出不可靠，缺乏交叉验证 | **** | 加权投票 + 辩论引擎 + Chairman 仲裁 | DecisionEngine + DebateEngine |
| 投资决策缺乏概率化表达 | **** | 三维概率分布 + 情景分析 | probability_up/down/sideways + scenario_analysis |
| 财经数据源不稳定，字段不完整 | **** | 5 源并发 + 字段级合并 + 指数退避重试 | CrawlerRegistry.fetch_merge() + _request_text() |
| 传统工具页面切换丢失分析进度 | *** | 跨页面持久化 + 事件缓冲 + 自动重连 | TaskTracker + sessionStorage + SSE reconnect |
| API 调用成本高（Token 消耗大） | *** | 分层 Prompt 按需裁剪 | Token 减少 72%（4,500 → 1,250） |
| 实时性差，用户需等待完整结果 | *** | SSE 流式推送，Agent 完成一个显示一个 | fetch + ReadableStream + EventSource pattern |

### 1.3 技术创新点

1. **六维协同 + 动态权重**：根据 MA-Agent 判定的市场周期（牛市趋势/牛市价值/熊市防御/震荡市）自动调整各 Agent 决策权重，RA-Agent 在熊市中权重从 5% 跃升至 30%
2. **分层 LLM Prompt 策略**：short 预测只发 K线+技术+资金（~1,250 tokens），medium 加基本面，long 加宏观，相比全量发送减少 72% Token 消耗
3. **数据质量驱动的置信度校准**：`calibrated_confidence = raw_confidence × data_quality_factor`，缺失数据自动降权
4. **Zero Mock Policy**：全链路数据来自 5 大真实财经平台，获取失败返回 HTTP 503，永不编造虚假数据
5. **字段级多源合并**：不同爬虫返回不同字段时按优先级缝合，而非整份替换

### 1.4 技术栈全景

```
┌─────────────────────────────────────────────────────────────────┐
│  层级          │  技术选型                      │  版本          │
├─────────────────────────────────────────────────────────────────┤
│  运行时        │  Python 3.11                   │  3.11.7        │
│  Web 框架      │  Flask                         │  2.2.5         │
│  WSGI 服务器   │  Waitress (多线程)              │  2.1.0         │
│  LLM 适配      │  DeepSeek / OpenAI / Claude    │  多厂商统一 API│
│  数据采集      │  requests + akshare            │  5 源并发 fan-out│
│  数值计算      │  NumPy / Pandas                │  2.0+          │
│  数据校验      │  Pydantic v2                   │  2.0+          │
│  序列化加速    │  orjson (Rust)                 │  3.9+          │
│  数据库        │  SQLite (WAL + 线程本地连接池) │  3.x           │
│  前端模板      │  Jinja2                        │  3.x           │
│  图表引擎      │  ECharts 5.4 + echarts-gl      │  5.4.3         │
│  图标          │  Bootstrap Icons               │  1.11.2        │
│  测试          │  pytest                        │  9.x           │
│  日志          │  loguru                        │  0.7+          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构深度剖析

### 2.1 五层架构详解

MASS 采用分层架构，每层职责清晰、接口明确、可独立替换。

```
┌──────────────────────────────────────────────────────────────────┐
│  L5 — 前端展示层 (Presentation)                                   │
│  ┌────────────┬────────────┬────────────┬────────────┬─────────┐ │
│  │ Jinja2 模板 │ ECharts 图表│ SSE 流处理器│ sessionStorage│ Logo  │ │
│  │ 10 HTML 文件│ 8 种图表类型│ ReadableStream│ 跨页持久化  │ SVG    │ │
│  └────────────┴────────────┴────────────┴────────────┴─────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  L4 — API 网关层 (Gateway)                                        │
│  ┌────────────┬────────────┬────────────┬──────────────────────┐ │
│  │ 32 端点     │ 中间件链    │ TaskTracker│ 认证 (Session)       │ │
│  │ REST + SSE │ 限流/日志/CORS│ 事件缓冲   │ admin/demo          │ │
│  └────────────┴────────────┴────────────┴──────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  L3 — 编排调度层 (Orchestration)                                   │
│  ┌──────────────┬──────────────┬──────────────┬────────────────┐ │
│  │Orchestrator  │DecisionEngine│PredictionEng │DebateEngine    │ │
│  │5 阶段管线     │加权投票+风险  │分层Prompt+   │1v1 质询+      │ │
│  │线程池调度     │过滤          │置信度校准    │交叉验证        │ │
│  └──────────────┴──────────────┴──────────────┴────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  L2 — Agent 层 (Multi-Agent)                                      │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐        │
│  │TA-Agent│FA-Agent│CA-Agent│SA-Agent│MA-Agent│RA-Agent│        │
│  │技术面   │基本面   │资金面   │情绪面   │宏观策略 │风险控制 │        │
│  │#06b6d4 │#3b82f6 │#a855f7 │#f97316 │#eab308 │#ef4444 │        │
│  └────────┴────────┴────────┴────────┴────────┴────────┘        │
│                     ↕ Blackboard (共享黑板)                        │
├──────────────────────────────────────────────────────────────────┤
│  L1 — 数据与基础层 (Infrastructure)                                │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────────┐ │
│  │CrawlerReg│LLMClient │CacheMgr  │Database  │IndicatorTool   │ │
│  │5源并发    │4厂商适配 │TTL+容量  │WAL+连接池│MA/BOLL/KDJ/MACD│ │
│  └──────────┴──────────┴──────────┴──────────┴────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

用户请求一只股票诊断的完整数据流：

```
GET /api/agent/diagnose  {"stock_code": "000001"}
  │
  ├─[1] Middleware 链
  │   ├─ RequestLogger.before_request()   → 记录 request_id + start_time
  │   ├─ RateLimiter.is_allowed()          → 滑动窗口 O(1) 检查
  │   └─ CORSHeaders.handle_options()     → OPTIONS 预检
  │
  ├─[2] 缓存检查
  │   └─ CacheManager.get("diagnose:000001:2026052719")
  │       ├─ 命中 → 直接返回 (延迟 < 30ms)
  │       └─ 未命中 → 继续
  │
  ├─[3] 数据采集 (并行 fan-out)
  │   ┌─────────────────────────────────────────────┐
  │   │ ThreadPoolExecutor(5)._DATA_FETCH_EXECUTOR  │
  │   ├─────────────────────────────────────────────┤
  │   │ _fetch_kline()          → akshare → sina   │ ← 关键路径
  │   │ _fetch_fundamentals()   → eastmoney+sina+tx│
  │   │ _fetch_fund_flow()      → eastmoney         │
  │   │ _fetch_sentiment_bundle() → eastmoney+sina  │
  │   │ _fetch_market_context() → eastmoney+sina    │
  │   │ _fetch_macro()          → eastmoney+ths     │
  │   └─────────────────────────────────────────────┘
  │   kline 返回后 → 立即 submit compute_all() + compute_risk_metrics()
  │
  ├─[4] 6 Agent 并行分析
  │   ┌─────────────────────────────────────────────┐
  │   │ ThreadPoolExecutor(6)._AGENT_EXECUTOR       │
  │   ├─────────────────────────────────────────────┤
  │   │ TA-Agent.analyze(snapshot) → LLM call      │
  │   │ FA-Agent.analyze(snapshot) → LLM call      │  并发
  │   │ CA-Agent.analyze(snapshot) → LLM call      │
  │   │ SA-Agent.analyze(snapshot) → LLM call      │
  │   │ MA-Agent.analyze(snapshot) → LLM call      │
  │   │ RA-Agent.analyze(snapshot) → LLM call      │
  │   └─────────────────────────────────────────────┘
  │   每个 Agent: BaseAgent._build_user_prompt() → LLMClient.chat() → _safe_parse()
  │
  ├─[5] 决策引擎
  │   ├─ MA-Agent → market_cycle → 选择权重方案 (WEIGHT_MAP)
  │   ├─ 加权投票: Σ(signal × confidence × weight) → preliminary_signal
  │   └─ RA-Agent → risk_level → apply_risk_filter()
  │
  ├─[6] Chairman 综合决策
  │   ├─ 构建上下文 Prompt (6 Agent 观点 + 决策引擎输出)
  │   ├─ LLM 调用 (Chairman 专用模型)
  │   ├─ 校验: confidence < 0.6 → 强制观望
  │   └─ 失败降级: _chairman_rule_decide() 规则引擎
  │
  ├─[7] 响应组装
  │   ├─ DecisionPackage model_dump()
  │   ├─ DB fire-and-forget 保存 (_DB_SAVE_EXECUTOR)
  │   └─ Cache.set(TTL=300)
  │
  └─[8] Response + Middleware 后置
      ├─ RequestLogger.after_request() → 记录响应时间
      └─ CORSHeaders.add_cors_headers() → Access-Control-*
```

### 2.3 关键设计决策

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| LLM 调用方式 | 同步 vs 异步 | 同步 + 线程池 | 简单可靠，Python GIL 下线程池是实际最优解 |
| 爬虫并发模型 | 串行 vs 线程并发 | 线程并发 (ThreadPool) | 5 个独立 HTTP 请求，I/O 密集，线程模型最合适 |
| 数据库 | SQLite vs PostgreSQL | SQLite (WAL 模式) | 单机部署，WAL 模式提供足够并发，零运维成本 |
| JSON 序列化 | stdlib json vs orjson | orjson (Rust) | 5.5× 加速，透明替换，失败自动降级 |
| 前端交互 | SPA (React/Vue) vs MPA (Jinja2) | MPA + SSE | 快速迭代，无需构建工具链，SSE 提供实时能力 |
| WSGI 服务器 | Gunicorn vs Waitress | Waitress | Windows 兼容，纯 Python，多线程模型，零配置 |

---

## 3. 多智能体协作原理

### 3.1 Agent 抽象基类设计

```python
class BaseAgent(ABC):
    """
    所有 Agent 的抽象基类，定义统一接口。

    子类必须实现:
    1. analyze(snapshot, user_position) → AgentOpinion
    2. _default_prompt() → str
    """

    def __init__(self, agent_id: str, llm_client: LLMClient, model_params=None):
        self.agent_id = agent_id          # "TA-Agent", "FA-Agent", ...
        self.llm = llm_client             # 共享的 LLM 客户端
        self.model_params = model_params  # temperature/top_p 等可覆盖
        self.system_prompt = self._load_prompt()  # 从文件动态加载

    @abstractmethod
    def analyze(self, snapshot: StockSnapshot, user_position=None) -> AgentOpinion:
        """核心分析接口"""

    def _call_llm(self, user_prompt: str, json_mode=True) -> dict:
        """调用 LLM 统一入口"""
        return self.llm.chat(
            system=self.system_prompt,
            user=user_prompt,
            json_mode=json_mode,
            override_config=self.model_params,
        )

    def _safe_parse_llm_response(self, response: dict) -> dict:
        """安全解析 LLM 输出，确保关键字段存在并校验类型"""
        defaults = {"signal": 0, "confidence": 0.5, "reasoning": "...",
                    "key_factors": [], "risk_flags": []}
        result = {**defaults, **response}
        # 类型校验 + 边界钳制
        result["signal"] = int(result["signal"])
        if result["signal"] not in (-1, 0, 1): result["signal"] = 0
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        return result
```

**设计要点**：
- **模板方法模式**：`analyze()` 由子类实现，`_call_llm()` 和 `_safe_parse()` 在基类提供
- **防御式编程**：`_safe_parse_llm_response()` 确保即使 LLM 返回异常格式，系统也不会崩溃
- **提示词热加载**：`_load_prompt()` 每次调用时从文件读取，修改 prompt 文件无需重启

### 3.2 各 Agent 分析方法论

#### TA-Agent（技术面分析）

```
输入: StockSnapshot (120日K线 + 44项技术指标)
分析维度:
  - K线形态识别: 双底/头肩/三角形/旗形 (LLM 模式识别)
  - 均线系统: MA5/MA20/MA60 排列 (多头/空头/缠绕)
  - 动量指标: MACD 金叉/死叉、RSI 超买/超卖、KDJ 位置
  - 波动指标: BOLL 带位置、ATR 波动幅度
  - 量价关系: 放量/缩量、量价背离检测
输出: signal + target_price_range + stop_loss + chart_patterns + trend_direction
降级: 均线排列 + MACD 金叉死叉 规则判断
```

#### FA-Agent（基本面分析）

```
输入: StockSnapshot.fundamentals (PE/PB/ROE/营收增速/毛利率/净利率)
分析维度:
  - 盈利能力: ROE、毛利率、净利率 → profitability 子评分 (0-25)
  - 成长能力: 营收YoY、利润YoY → growth 子评分 (0-25)
  - 安全性: 资产负债率、流动比率 → safety 子评分 (0-20)
  - 估值水平: PE分位、PB分位、行业对比 → valuation 子评分 (0-20)
  - 护城河: 品牌/专利/网络效应 → moat 子评分 (0-10)
输出: fundamental_score (0-100) + sub_scores + valuation_gap
局限性: 免费数据源字段有限，PE/PB/行业均值依赖 eastmoney
```

#### CA-Agent（资金面分析）

```
输入: StockSnapshot.fund_flow (主力净流入、大单动向、北向资金)
分析周期: 5日/10日/20日 资金趋势
指标:
  - 主力净流入: 大单买入 - 大单卖出，正值表示吸筹
  - 北向资金: 沪深港通净买入额 (仅沪港通标的)
  - 融资融券: 融资余额变化趋势
  - 大宗交易: 折溢价率、成交量
输出: capital_score + smart_money_direction (建仓期/出货期/观望)
局限性: 东方财富反爬严格，fund_flow 数据获取成功率 ~60%
```

#### SA-Agent（情绪面分析）

```
输入: 新闻列表 + 社交媒体文本
分析流程:
  1. 爬取新闻标题 (sina feed API)
  2. 词典规则打分: POSITIVE_WORDS(30个) vs NEGATIVE_WORDS(30个)
  3. 情感极性: (pos - neg) / (pos + neg) → [-1, 1]
  4. crow_behavior 推断: 恐慌/贪婪/正常/极端
输出: sentiment_index + sentiment_percentile + crowd_behavior
局限性: 无社交媒体 API，依赖新浪财经新闻 (覆盖有限)
```

#### MA-Agent（宏观策略）

```
输入: 10Y国债收益率、PMI、政策信号 (eastmoney 宏观接口)
分析框架:
  - 美林时钟定位: 复苏/过热/滞胀/衰退
  - 货币政策: 利率趋势、准备金率
  - 行业景气: 行业 PMI、政策导向 (利好/中性/利空)
  - 风格匹配: 当前周期下成长/价值/防御的风格适配度
输出: market_cycle + sector_outlook + recommended_weight_adjustment
```

#### RA-Agent（风险控制）

```
输入: StockSnapshot.risk_metrics (波动率/VaR/最大回撤/Beta)
分析方法:
  1. 年化波动率: std(daily_returns) × sqrt(252)
  2. VaR(95%): 历史模拟法 5% 分位数
  3. 最大回撤: cummax - current / cummax
  4. Beta: 相对大盘的回归系数 (简化计算)
  5. 下行标准差: 仅负收益的标准差
风险等级: 1(极低)→5(极高)
仓位公式: 凯利公式修正版
  f = (p × b - q) / b  × risk_multiplier
  其中 p=胜率, b=赔率, q=1-p, risk_multiplier=1/risk_level
输出: risk_level + max_position_pct + recommended_stop_loss + black_scenarios
```

### 3.3 协作流程：从分歧到共识

```
Phase 1: 独立分析
  TA: 买入(85%)  FA: 买入(70%)  CA: 买入(65%)
  SA: 观望(55%)  MA: 观望(60%)  RA: 观望(80%)

Phase 2: 冲突检测
  DebateEngine.detect_conflicts()
  检测到: TA vs SA (signal_diff=1, conf_gap=0.3) ← 触发辩论
          CA vs RA (signal_diff=1, risk认知分歧)    ← 触发辩论

Phase 3: 辩论 (可选)
  SA(challenger) → TA(responder)
  Round 1: "你的技术面买入信号是否忽略了情绪面极度悲观的逆向风险？"
           → "情绪极度悲观往往是技术面见底的确认信号，历史上..."
  CA(challenger) → RA(responder)
  Round 1: "资金持续流入是否说明市场已price in风险？"
           → "资金面只是短期信号，需要关注中期波动率上升..."

Phase 4: 加权投票
  MA-Agent 判定 market_cycle = "复苏早期" → "bull_trend" 权重:
  TA:0.30  FA:0.15  CA:0.25  SA:0.15  MA:0.10  RA:0.05
  weighted_sum = 1×0.85×0.30 + 1×0.70×0.15 + 1×0.65×0.25
               + 0×0.55×0.15 + 0×0.60×0.10 + 0×0.80×0.05
               = 0.523
  threshold: > 0.15 → 买入信号

Phase 5: 风险过滤
  RA-Agent: risk_level=3 (中等), max_position=15%
  不在高风险拦截范围 (level<4)，通过
  confidence_check: overall_confidence=0.72 > 0.55 → 通过

Phase 6: Chairman 综合决策
  final_decision: 买入 (基于 4/6 买入或观望)
  confidence: 0.72 (加权平均)
  position_pct: min(0.10, RA.max_position_pct × 0.8) = 0.10
```

### 3.4 共享黑板 (Blackboard) 设计

```python
@dataclass
class StockSnapshot:
    """所有 Agent 共享的输入数据"""
    stock_code: str
    stock_name: str
    current_price: float
    kline_df: Optional[pd.DataFrame]          # OHLCV 120日
    indicators: Dict[str, Any]                 # 44 项技术指标
    fundamentals: Dict[str, Any]               # PE/PB/ROE/...
    fund_flow: Dict[str, Any]                  # 主力/北向/融资
    market_context: Dict[str, Any]             # 大盘指数/板块
    sentiment_data: Dict[str, Any]             # 新闻/情绪
    macro_data: Dict[str, Any]                 # PMI/利率/政策
    risk_metrics: Dict[str, Any]               # 波动率/VaR/回撤
    data_quality: Dict[str, Any]               # 数据质量报告

@dataclass
class AgentOpinion:
    """每个 Agent 的输出"""
    agent_id: str          # "TA-Agent"
    signal: int            # -1 / 0 / 1
    confidence: float      # 0.0 ~ 1.0
    reasoning: str         # 自然语言推理
    key_factors: List[str] # 关键因子
    risk_flags: List[str]  # 风险标记
    raw_data: Dict         # 完整 LLM 输出
```

**并发安全**：按股票代码分片锁 `Dict[str, Lock]`，不同股票间无锁竞争。

```python
def _get_lock(self, stock_code: str) -> threading.Lock:
    with self._global_lock:
        if stock_code not in self._locks:
            self._locks[stock_code] = threading.Lock()
        return self._locks[stock_code]
```

---

## 4. LLM 集成与 Prompt 工程

### 4.1 多厂商统一适配

```python
@dataclass
class LLMConfig:
    provider: str = "deepseek"     # deepseek / openai / claude / ollama
    api_key: str = ""
    base_url: str = ""
    model: str = "deepseek-v4-pro"
    temperature: float = 0.2       # 低温度保证分析稳定性
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3           # 指数退避重试
```

**调用流程**：
```
LLMClient.chat(system, user)
  ├─ 构建 messages: [{"role":"system","content":system}, {"role":"user","content":user}]
  ├─ json_mode → response_format={"type":"json_object"}
  ├─ retry loop (3次, 指数退避 1s→2s→4s)
  ├─ JSON 解析失败 → _fallback_parse() 正则提取
  └─ 全部失败 → 返回默认值 (signal=0, confidence=0.3)
```

**MockLLMClient**：用于测试，根据 system prompt 内容识别 Agent 类型，返回合理模拟数据，不产生 API 费用。

### 4.2 Prompt 模板体系

MASS 使用三层 Prompt 架构：

```
Layer 1: System Prompt (角色定义 + 分析框架 + 输出约束)
  ├─ agent/prompts/system/ta_agent.md   → 技术分析专家角色
  ├─ agent/prompts/system/fa_agent.md   → 基本面研究员角色
  ├─ agent/prompts/system/ca_agent.md   → 资金流向分析师角色
  ├─ agent/prompts/system/sa_agent.md   → 市场情绪观察员角色
  ├─ agent/prompts/system/ma_agent.md   → 宏观策略师角色
  ├─ agent/prompts/system/ra_agent.md   → 风控官角色
  ├─ agent/prompts/system/chairman.md   → 投资委员会主席角色
  └─ agent/prompts/system/prediction.md → 量化预测师 (含动态注入段)

Layer 2: User Prompt (数据上下文 + 具体任务)
  ├─ StockSnapshot.to_prompt_context() → 结构化的 7 段 JSON 数据
  └─ 各 Agent._build_*_prompt() → 按需裁剪

Layer 3: Dynamic Injection (动态注入)
  ├─ {RISK_TOLERANCE_SECTION}  → conservative/moderate/aggressive
  ├─ {INVESTMENT_STYLE_SECTION} → swing/trend/value
  └─ {CONFIDENCE_SECTION}       → strict/standard/relaxed
```

### 4.3 分层 Prompt 策略（Token 优化核心）

```
预测周期         包含的数据段          Token 估算    相比全量
───────────────────────────────────────────────────────────
short (1-5天)    K线(10日)+技术+资金+情绪    ~1,250      -72%
medium (1-4周)   K线(30日)+技术+资金+情绪+基本面  ~1,284  -71%
long (1-3月)     K线(120日)+技术+基本面+宏观    ~1,308    -71%
旧版全量         全部 7 段数据              ~4,500      baseline
```

**实现**：
```python
def _build_prediction_prompt(self, stock_code, stock_name, snapshot, horizon):
    # 所有周期: K线 + 技术指标 + 风险指标
    # short/medium: + 资金流向 + 市场情绪
    # medium/long:  + 基本面
    # long only:    + 宏观环境 + 大盘指数
```

### 4.4 热更新机制

```python
def _load_prompt(self) -> str:
    """每次 analyze() 调用时从磁盘读取，支持运行时热更新"""
    path = Path(f"agent/prompts/system/{self.agent_id.lower()}.md")
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return self._default_prompt()  # 文件缺失时的内嵌默认值
```

---

## 5. 数据层设计详解

### 5.1 爬虫架构

```
CrawlerRegistry (单例)
  │
  ├─ AkshareCrawler (priority=95)
  │   └─ akshare.stock_zh_a_hist() → K线
  │   └─ akshare.stock_individual_info() → 基本面
  │
  ├─ EastMoneyCrawler (priority=100)
  │   ├─ push2.eastmoney.com/api/qt/stock/get → 基本面
  │   ├─ push2his.../fflow/daykline/get → 资金流向
  │   ├─ push2.../api/qt/clist/get → 市场环境(板块)
  │   └─ searchapi.../api/suggest/get → 情绪(涨跌停)
  │
  ├─ SinaCrawler (priority=90)
  │   ├─ hq.sinajs.cn/list= → 实时行情 (GBK文本)
  │   ├─ money.finance.sina.../getKLineData → K线 (JSON)
  │   └─ feed.mix.sina.../api/roll/get → 新闻
  │
  ├─ TxCrawler (priority=80)
  │   └─ qt.gtimg.cn/q= → 个股信息 (GBK文本)
  │
  └─ THSCrawler (priority=50)
      └─ basic.10jqka.com.cn/api/stockphb/ → 基本面备用
```

### 5.2 统一重试层

```python
def _request_text(self, url, method="GET", params=None, headers=None,
                  encoding=None, **kwargs) -> Optional[str]:
    """
    统一请求方法 — 原始文本版
    特性: 自动频率控制 + UA轮换 + 指数退避重试 + 403/429 长等待
    """
    self._rate_limit()                              # 频率控制
    req_headers = {"User-Agent": self._ua_rotator.get()}  # UA轮换
    if headers: req_headers.update(headers)

    for attempt in range(1, self.retries + 1):      # 最多 3 次
        response = self._session.request(...)
        response.raise_for_status()
        if encoding: response.encoding = encoding   # GBK 适配
        return response.text.strip()

        # 超时 → 重试
        # HTTPError(403/429) → 额外 1-3s 随机长等待 → 重试
        # 其他异常 → 指数退避 sleep_time = 1s × 2^(n-1) + random(0,1)
    return None
```

### 5.3 字段级合并算法

```python
def fetch_merge(self, stock_code, data_type):
    """
    并发 fan-out → 按优先级字段合并
    """
    # Step 1: 并发提交所有爬虫
    futures = {executor.submit(c.fetch, ...): c for c in self._crawlers}

    # Step 2: 收集结果 (按优先级排序)
    source_results = [(c.priority, c.name, result) for ...]
    source_results.sort(key=lambda x: x[0], reverse=True)

    # Step 3: 字段级合并 — 高优先级字段优先采用
    for priority, name, result in source_results:
        for key, value in result.items():
            if value not in (None, "", 0) and key not in all_fields_found:
                merged[key] = value           # 首次出现 → 采用
                all_fields_found.add(key)     # 标记已填充
                fields_from[key] = name       # 记录来源

    # Step 4: 质量报告
    merged["_meta"] = {
        "sources_succeeded": [...],
        "fields_coverage": {...},
        "data_completeness": 0.75,
        "missing_fields": [...]
    }
    return merged
```

**示例**：
```
EastMoney 返回: {pe_ttm:4.41, pb:0.42, company_name:"浦发银行"}
Sina 返回:      {bid_ask:{...}, company_name:"浦发银行", latest_price:10.23}
Tx 返回:        {pe_ttm:4.40, market_cap:2800亿}

合并结果: {pe_ttm:4.41(E), pb:0.42(E), company_name:"浦发银行"(E),
           bid_ask:{...}(S), latest_price:10.23(S), market_cap:2800亿(T)}
```

### 5.4 数据质量评估

```python
def _compute_quality_factor(self, snapshot: StockSnapshot) -> float:
    weights = {"kline": 0.30, "fundamentals": 0.25, "fund_flow": 0.15,
               "sentiment": 0.10, "market_context": 0.10, "macro": 0.10}
    total = 0.0
    for dtype, w in weights.items():
        info = data_types.get(dtype, {})
        if info["status"] == "ok":        total += w
        elif info["status"] == "partial": total += w * info["completeness"]
        # unavailable → 0
    return max(0.3, total)  # 最低 0.3 防止过度惩罚
```

### 5.5 技术指标计算

```
IndicatorTool.compute_all(df):
  ├─ 价格位置: current_price, high_20d, low_20d, position_20d, position_60d
  ├─ 移动平均线: MA5, MA10, MA20, MA60 + 排列判断 (多头/空头/缠绕)
  ├─ BOLL带 (20,2): upper, mid, lower, width, position
  ├─ KDJ (9,3,3): K, D, J + 金叉/死叉 + 状态(超买/超卖/强势/弱势)
  ├─ MACD (12,26,9): DIF, DEA, HIST + 金叉/死叉 + 零轴判断
  ├─ RSI (14): 数值 + 状态(严重超买→严重超卖)
  ├─ 成交量: vol_ma5, vol_ma20, volume_ratio, volume_trend
  ├─ ATR (14): 数值 + % 表达
  ├─ 波动率: volatility_20d, volatility_60d (年化)
  ├─ 最大回撤: max_drawdown_60d
  └─ 量价关系: price_volume_divergence (顶背离/底背离/量价齐升/量价齐跌)
```

**向量化优化**：RSV 计算从 Python for 循环改为 `pd.Series.rolling()`，3.2× 加速。

---

## 6. 决策引擎数学原理

### 6.1 动态权重系统

```python
# 4 种市场周期的权重方案 (config.py)
WEIGHT_MAP = {
    "bull_trend":   {"TA":0.30, "FA":0.15, "CA":0.25, "SA":0.15, "MA":0.10, "RA":0.05},
    "bull_value":   {"TA":0.15, "FA":0.35, "CA":0.20, "SA":0.10, "MA":0.10, "RA":0.10},
    "bear_defense": {"TA":0.10, "FA":0.20, "CA":0.15, "SA":0.15, "MA":0.10, "RA":0.30},
    "oscillation":  {"TA":0.25, "FA":0.15, "CA":0.25, "SA":0.20, "MA":0.05, "RA":0.10},
}
```

**周期-权重映射逻辑**：

| 市场周期 | 权重方案 | 特征 | TA权重 | RA权重 |
|---------|---------|------|--------|--------|
| 复苏早期 | bull_trend | 趋势明确，技术信号可信 | 30% | 5% |
| 复苏晚期 | bull_value | 关注基本面验证 | 15% | 10% |
| 过热 | bull_trend | 趋势仍在，风险上升 | 30% | 5% |
| 滞胀 | bear_defense | **转向防御** | 10% | **30%** |
| 衰退早期 | bear_defense | **风控优先** | 10% | **30%** |
| 衰退晚期 | bull_value | 底部布局，价值优先 | 15% | 10% |

### 6.2 加权投票公式

```
weighted_sum = Σ(signal_i × confidence_i × weight_i)  for i in {TA,FA,CA,SA,MA}
               (RA-Agent 不参与投票，仅用于风险过滤)

decision_threshold:
  weighted_sum >  0.15 → signal = 1  (买入)
  weighted_sum < -0.15 → signal = -1 (卖出)
  otherwise            → signal = 0  (观望)

overall_confidence = min(Σ(confidence_i × weight_i), 0.95)
```

### 6.3 风险过滤规则

```python
def apply_risk_filter(preliminary, ra_opinion):
    risk_level = ra_opinion.raw_data.get("risk_level", 3)

    if risk_level >= 5:
        result["final_signal"] = 0        # 强制观望
        result["risk_override"] = "风险等级 5，强制观望"
    elif risk_level == 4 and preliminary["preliminary_signal"] == 1:
        result["final_signal"] = 0        # 买入降级为观望
        result["risk_override"] = "风险等级 4，买入降级"
    elif preliminary["overall_confidence"] < 0.55:
        result["final_signal"] = 0        # 置信度不足
    else:
        result["final_signal"] = preliminary["preliminary_signal"]
```

### 6.4 置信度校准公式

```
calibrated_confidence = raw_confidence × data_quality_factor

data_quality_factor = max(0.3, Σ(completeness_i × weight_i))
                       i ∈ {kline(0.30), fundamentals(0.25), fund_flow(0.15),
                            sentiment(0.10), market(0.10), macro(0.10)}

当 calibrated_confidence < confidence_threshold → direction = "不确定"

示例:
  raw_confidence = 0.75
  data_quality_factor = 0.85 (一行数据缺失)
  calibrated = 0.75 × 0.85 = 0.64  (降低 15%)
  threshold = 0.6 → 0.64 > 0.6 → 保留原方向
```

### 6.5 辩论引擎 (DebateEngine)

```
冲突检测:
  触发条件 1: |signal_A - signal_B| >= 2  (例如 1 vs -1)
  触发条件 2: |confidence_A - confidence_B| >= 0.3 AND signal_diff > 0
  触发条件 3: risk_flags 差异 >= 2 项

辩论流程:
  Round 1: Challenger (低置信度方) → 质疑 → Responder (高置信度方) → 回应
  Round 2: (如果第一轮未达成共识) 交替角色再一轮
  Chairman 根据辩论结果修正决策
```

---

## 7. 性能优化全景

### 7.1 v2.2 十大优化详解

#### 优化 1: 指标计算入 fan-out

```
优化前: K线(2s) → 等基本面对齐(3s) → 计算指标(50ms) → 计算风险(30ms) = 5.08s
优化后: K线(2s) → 立即计算指标(50ms) ┐
                  → 立即计算风险(30ms) ├ 这三路并行
         基本面等(3s) ────────────────┘
         总耗时 = max(2.05, 3) = 3s   (省 ~2s)
```

#### 优化 9: orjson 替换

```python
# 透明替换: json.dumps → orjson.dumps
_ORIGINAL_DUMPS = json.dumps
try:
    import orjson as _orjson
    def _fast_dumps(obj, **kwargs):
        if any(k in kwargs for k in ('indent','separators','sort_keys')):
            return _ORIGINAL_DUMPS(obj, **kwargs)  # 特殊模式回退
        return _orjson.dumps(obj, default=kwargs.get('default'),
                             option=_orjson.OPT_NON_STR_KEYS).decode()
    json.dumps = _fast_dumps
except ImportError:
    pass  # 静默降级
```

**性能对比 (50KB DecisionPackage)**：
| 库 | 耗时 | 加速比 |
|----|------|--------|
| stdlib json | 2.5ms | 1× |
| orjson | 0.45ms | **5.5×** |

### 7.2 内存管理

```
缓存层级           上限      淘汰策略       内存占用 (估算)
─────────────────────────────────────────────────────
CacheManager       无        惰性+定期清理    ~2-5MB
StockDataTool      500 条    容量淘汰(20%)    ~5-10MB
Blackboard         200 只    时间戳淘汰(20%)  ~10-20MB
TaskTracker        无        TTL 600s        ~1-5MB
```



### 7.3 线程池全景

```
╔══════════════════════════════════════════════════════════╗
║ 线程池               Workers  前缀           用途         ║
╠══════════════════════════════════════════════════════════╣
║ _CRAWLER_EXECUTOR      5     crawler-      5源并发fan-out║
║ _DATA_FETCH_EXECUTOR   5     data-fetch-   数据+指标并行 ║
║ _AGENT_EXECUTOR        6     agent-        6 Agent LLM   ║
║ _PORTFOLIO_EXECUTOR    3     portfolio-    组合并行诊断   ║
║ _DIAGNOSIS_EXECUTOR   10     diagnosis-bg- SSE后台诊断    ║
║ _DB_SAVE_EXECUTOR      2     db-save-      异步DB写入     ║
║ Waitress threads       8     —             HTTP请求处理   ║
╠══════════════════════════════════════════════════════════╣
║ 总计                  39 workers                         ║
╚══════════════════════════════════════════════════════════╝
```

---

## 8. 前端架构与流式渲染

### 8.1 SSE 流式渲染原理

```
客户端                   服务端
  │                        │
  ├─ fetch POST /stream ──→│
  │   {stock_code:"000001"} │
  │                        ├─ Phase1: 数据采集 (2-5s)
  │                        │
  │←─ data: {stage:"init"} ─┤ 立即响应
  │←─ data: {stage:"data"} ─┤
  │                        ├─ Phase2: 6 Agent 并行 (5-15s)
  │←─ data: {stage:"agent", agent_id:"TA", ...} ─┤
  │   → updateStreamAgentCardV2("TA-Agent", ...)  │ (Agent 完成即推送)
  │←─ data: {stage:"agent", agent_id:"FA", ...} ─┤
  │   → updateStreamAgentCardV2("FA-Agent", ...)  │
  │   ...                                          │
  │←─ : heartbeat\n\n ───┤ (每15秒保活)
  │                        ├─ Phase3-5: 决策+Chairman
  │←─ data: {stage:"result", data:{...}} ─┤
  │   → renderFullResult() │
  │                        │
  ├─ 连接关闭 ────────────→│
```

### 8.2 6 Agent 卡片状态机

```
┌──────────┐   等待信号    ┌──────────┐   Agent完成   ┌──────────┐
│ PENDING  │────────────→ │ ACTIVE   │─────────────→ │ COMPLETE │
│ opacity: │              │ opacity: │              │ opacity: │
│ 0.4      │              │ 1.0      │              │ 1.0      │
│ border:  │              │ border:  │              │ border:  │
│ subtle   │              │ color    │              │ color(淡)│
└──────────┘              └──────────┘              └──────────┘
                                                         │
                                                    signal=1 (买入)
                                                         ↓
                                                  ┌──────────┐
                                                  │ ACTIVE   │
                                                  │ border:  │
                                                  │ green    │
                                                  │ glow     │
                                                  └──────────┘
```

### 8.3 跨页面任务持久化

```
t=0s   用户点击"实时诊断 000001"
       → POST /stream {stock_code:"000001"}
       → 响应头 X-Task-ID: 000001_1716883200123
       → sessionStorage.setItem("mass_active_task", {task_id, stock_code, ...})
       → 后台线程开始执行，事件写入 TaskTracker buffer

t=15s  用户点击"历史记录"导航
       → 诊断页面 onbeforeunload → 无操作 (自然断开)
       → SSE 连接关闭
       → 但后台线程继续运行
       → 事件持续写入 buffer [event_15, event_16, ..., event_45]

t=45s  用户点击"个股诊断"返回
       → 页面加载 → 检测 sessionStorage.getItem("mass_active_task")
       → 发现活跃任务: task_id=000001_1716883200123, stock_code=000001
       → 自动填入输入框
       → POST /stream {stock_code:"000001", task_id:"000001_1716883200123"}
       → 服务端 TaskTracker.get_task(task_id) → status=running, buffer=30 events
       → 先回放: data: {stage:"reconnect", replay_count:30}
       → 逐条推送 30 个已缓冲事件 → 前端恢复 6 个 Agent 卡片
       → 继续接收实时事件 event_46, event_47, ... event_55
       → 收到 stage:"result" → renderFullResult() → clearActiveTask()
```

### 8.4 ECharts 图表配置

**K线图**：使用 `echarts.init()` + candlestick series + MA5/MA10/MA20 overlay + MACD 副图 + 成交量柱状图。支持缩放、十字光标、tooltip 显示完整 OHLCV。

**3D 雷达图**：使用 `echarts-gl` 的 radar3D 组件，6 个轴分别对应各 Agent 的 confidence，直观展示多维能力。

**情景仪表盘**：3 个 gauge 图表并排，分别显示牛市/基准/熊市情景的概率和预期收益。

**信号分布饼图**：环形饼图 (radius: ['50%', '75%'])，买入(绿)/观望(黄)/卖出(红)三个扇区。

---

## 9. 工程实践与设计模式

### 9.1 设计模式应用

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **单例** | CrawlerRegistry, Blackboard, StockDataTool, CacheManager, TaskTracker | 全局唯一实例，双重检查锁定 |
| **模板方法** | BaseAgent.analyze() | 基类定义骨架，子类实现细节 |
| **策略** | DecisionEngine 权重方案 | 4 种周期 → 4 套权重，运行时切换 |
| **观察者** | Blackboard → Agent | 快照发布后 Agent 自动感知 |
| **工厂** | LLMClient._default_config() | 根据 LLM_PROVIDER 创建配置 |
| **装饰器** | RateLimiter.limit() | 透明限流包装 |
| **适配器** | LLMClient (OpenAI/Claude/Ollama) | 统一接口适配不同厂商 API |

### 9.2 错误处理与降级体系

```
异常层级:
  MASSException (基类, code + status_code)
    ├─ AgentError       → HTTP 500   Agent 分析失败
    ├─ DataError        → HTTP 503   数据不可用
    ├─ LLMError         → HTTP 503   LLM 调用失败
    ├─ ValidationError  → HTTP 400   参数校验失败
    ├─ NotFoundError    → HTTP 404   资源不存在
    └─ RateLimitError   → HTTP 429   请求过频

降级链路:
  Agent.analyze() → LLM 失败 → _fallback_opinion() (规则引擎)
  Chairman LLM 失败 → _chairman_rule_decide() (规则决策)
  JSON 解析失败 → _fallback_parse() (正则提取 + 默认值)
  数据源全部失败 → HTTP 503 DATA_UNAVAILABLE
```

### 9.3 线程安全保证

```
组件              并发控制              原因
──────────────────────────────────────────────────
Blackboard        Dict[str, Lock]        按股票分片，不同股票无竞争
CacheManager      threading.RLock()      读写比 > 100:1，RLock 最优
RateLimiter       threading.Lock()       O(1) 滑动窗口，持锁时间 < 1μs
Database          threading.local()      每线程独立 SQLite 连接
TaskTracker       threading.Lock()       任务注册/查询/取消互斥
CrawlerRegistry   Lock() (register)      注册是启动时一次性操作
```

### 9.4 日志系统

```python
# app.py — loguru 配置
logger.add("logs/mass_{time}.log", rotation="10 MB",
           retention="7 days", level="INFO")
logger.add("logs/mass_error_{time}.log", rotation="10 MB",
           retention="7 days", level="ERROR")
```

**日志级别使用规范**：
- `DEBUG`：爬虫 HTTP 请求详情、缓存命中细节
- `INFO`：Agent 完成、诊断耗时、注册/初始化
- `WARNING`：数据获取失败、LLM 降级、重试
- `ERROR`：LLM 全部失败、数据关键字段缺失

---

## 10. API 协议规范

### 10.1 通用规范

```
请求格式:    Content-Type: application/json
响应格式:    Content-Type: application/json
错误格式:    {"error": "消息", "code": "ERROR_CODE", "message": "详细说明"}
成功状态码:  200
错误状态码:  400 (参数错误) / 401 (未登录) / 404 / 429 (限流) / 500 / 503 (数据不可用)
缓存控制:    Cache-Control: no-cache (SSE) / ETag + 304 (静态资源)
CORS:       Access-Control-Allow-Origin: *
认证:       Cookie: session=<flask session>
```

### 10.2 核心端点详解

#### POST /api/agent/diagnose

```
Request:
{
  "stock_code": "000001",       // 必填，6位数字
  "stock_name": "平安银行",      // 可选
  "force_refresh": false,       // 可选，绕过缓存
  "model_params": {             // 可选，覆盖 LLM 参数
    "temperature": 0.3,
    "top_p": 0.9
  }
}

Response: DecisionPackage
{
  "stock_code": "000001",
  "stock_name": "平安银行",
  "current_price": 10.76,
  "decision_date": "2026-05-27",
  "decision_time": "20:15:30",
  "market_cycle": "复苏早期",
  "opinions": {
    "TA-Agent": {"signal": 1, "confidence": 0.85, "reasoning": "...", ...},
    "FA-Agent": {"signal": 1, "confidence": 0.70, ...},
    ...
  },
  "final_decision": {
    "decision": 1,
    "confidence": 0.72,
    "position_pct": 0.10,
    "target_price": 12.50,
    "stop_loss": 9.80,
    "time_horizon": "2-4周",
    "expected_return_pct": 15.2,
    "scenario_analysis": { ... },
    "execution_plan": ["分批建仓", "严格止损"]
  },
  "processing_time_seconds": 22.14,
  "data_quality": { ... },
  "from_cache": false,
  "disclaimer": "免责声明..."
}
```

#### POST /api/agent/predict (v2.3)

```
Request:
{
  "stock_code": "600000",
  "horizon": "short",              // short/medium/long
  "risk_tolerance": "moderate",     // conservative/moderate/aggressive
  "investment_style": "swing",      // swing/trend/value
  "confidence_threshold": 0.6,      // 0.4-0.8
  "force_refresh": false,
  "model_params": { ... }
}

Response: PredictionResult (v2.3 enhanced)
{
  "stock_code": "600000",
  "prediction_horizon": "short",
  "direction": "上涨",
  "confidence": 0.72,                // 原始置信度
  "confidence_calibrated": 0.61,     // 校准后置信度
  "data_quality_factor": 0.85,       // 校准因子
  "target_price_low": 18.5,
  "target_price_high": 22.0,
  "stop_loss": 17.0,
  "risk_reward_ratio": 2.5,
  "probability_up": 0.60,
  "probability_down": 0.25,
  "probability_sideways": 0.15,
  "holding_period_days": 5,
  "risk_tolerance": "moderate",      // 回显
  "investment_style": "swing",       // 回显
  "model_used": "deepseek-v4-pro",
  "fallback_used": false,
  "prompt_tokens_estimated": 1250
}
```

### 10.3 SSE 流式协议

```
请求:
  POST /api/agent/diagnose/stream
  Content-Type: application/json
  {"stock_code": "000001", "task_id": null}

响应:
  HTTP/1.1 200 OK
  Content-Type: text/event-stream
  Cache-Control: no-cache
  X-Accel-Buffering: no
  X-Task-ID: 000001_1716883200123

  data: {"stage":"init","message":"开始诊断 000001...","progress":0}
  data: {"stage":"data","message":"并行获取多源实时数据...","progress":5}
  data: {"stage":"data","message":"数据获取完成（K线120条 / 23个基本面字段）","progress":30}
  data: {"stage":"agent_start","message":"6大Agent并行分析中...","progress":32}
  data: {"stage":"agent","agent_id":"TA-Agent","agent_result":{...},"progress":35}
  data: {"stage":"agent","agent_id":"FA-Agent","agent_result":{...},"progress":43}
  ...
  data: {"stage":"engine","message":"决策引擎加权投票与风险过滤...","progress":85}
  data: {"stage":"chairman","message":"Chairman 综合各方观点...","progress":90}
  data: {"stage":"result","message":"诊断完成","progress":100,"data":{...}}
  : heartbeat

重连模式事件:
  data: {"stage":"reconnect","message":"重连成功，回放 30 个事件","replay_count":30}
  data: (回放的历史事件...)
  data: (继续的实时事件...)
  data: {"stage":"done","message":"任务已结束（重连）"}
```

### 10.4 错误码

| 错误码 | HTTP | 说明 |
|--------|------|------|
| `MISSING_STOCK_CODE` | 400 | stock_code 为空 |
| `INVALID_STOCK_CODE` | 400 | 非 6 位数字 |
| `INVALID_HORIZON` | 400 | 预测周期无效 |
| `INVALID_RISK_TOLERANCE` | 400 | 风险偏好无效 |
| `INVALID_INVESTMENT_STYLE` | 400 | 投资风格无效 |
| `MISSING_HOLDINGS` | 400 | 组合分析持仓为空 |
| `TOO_MANY_HOLDINGS` | 400 | 持仓超过 20 只 |
| `NOT_FOUND` | 404 | 决策/任务不存在 |
| `TASK_NOT_FOUND` | 404 | task_id 无效或过期 |
| `RATE_LIMIT` | 429 | 请求频率过高 |
| `AGENT_ERROR` | 500 | Agent 分析异常 |
| `LLM_ERROR` | 503 | LLM 调用失败 |
| `DATA_UNAVAILABLE` | 503 | 数据源不可用 |

---

## 11. 测试策略与质量保障

### 11.1 测试金字塔

```
        ┌─────┐
        │ E2E │  1 项 — 完整用户工作流
        ├─────┤
        │Stress│ 40 项 — 并发/内存/边界
       ┌┴─────┴┐
       │  Int. │ 42 项 — API/管线/编排器
      ┌┴───────┴┐
      │   Unit  │ 59 项 — Agent/黑板/爬虫/指标
      └─────────┘
      总计: 141 项  通过率: 100%
```

### 11.2 压力测试关键发现

| 测试 | 结果 | 说明 |
|------|------|------|
| 10 并发诊断 | 100% 成功 | 无死锁/竞态条件 |
| 18 混合并发 (诊断+预测+SSE+健康检查) | 100% 成功 | Waitress 8 线程 + 后台线程池解耦 |
| 8 并发预测 (不同参数组合) | 100% 成功 | 缓存隔离正确 |
| 100 并发缓存读写 | 0 异常 | `RLock` + 无竞态 |
| 300 请求限流器 | 0 误拦 | 3 client × 100 限额 |
| 黑板 250 只股票 | 淘汰至 ≤200 | 时间戳淘汰正常 |
| 30 次请求后内存 | +40MB | 无泄漏趋势 |
| Cache 命中延迟 | < 30ms | 满足 < 50ms 目标 |

### 11.3 Mock 测试策略

`MockLLMClient` 替代真实 LLM 调用，使单元测试可在无 API Key 环境中运行：

```python
class MockLLMClient(LLMClient):
    def chat(self, system, user, json_mode=True, model=None, override_config=None):
        """根据 system prompt 内容识别 Agent 类型，返回合理模拟数据"""
        if "技术面" in system:
            return {"signal": random.choice([-1,0,1]), "confidence": 0.75, ...}
        elif "基本面" in system:
            return {"fundamental_score": 72, "signal": 1, ...}
        # ... 各 Agent 专属模拟数据
```

---

## 12. 部署运维手册

### 12.1 快速开始

```bash
# 1. 克隆项目
cd MASS/

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
cp .env.example .env
# 编辑 .env: 填入 DEEPSEEK_API_KEY

# 4. Mock 模式测试（无需 API Key）
python run.py --mock --port 5000

# 5. 生产模式启动
python deploy.py --port 5000 --threads 8

# 6. 访问
open http://localhost:5000/login
# 用户名: admin  密码: admin
```

### 12.2 配置项完整清单

```bash
# ── Flask ──
FLASK_HOST=0.0.0.0           # 监听地址
FLASK_PORT=5000               # 监听端口
FLASK_DEBUG=False             # 调试模式 (生产必须 False)
SECRET_KEY=mass-secret-xxx    # Session 密钥

# ── LLM ──
LLM_PROVIDER=deepseek         # deepseek/openai/claude/ollama
DEEPSEEK_API_KEY=sk-xxx       # API 密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEFAULT_MODEL=deepseek-v4-pro
CHAIRMAN_MODEL=deepseek-v4-pro
PREDICTION_MODEL=deepseek-v4-pro
FALLBACK_PREDICTION_MODEL=    # 备模型 (可选)
LLM_TIMEOUT=60                # 单次调用超时秒数
LLM_MAX_RETRIES=3             # 最大重试次数
LLM_TEMPERATURE=0.2           # 温度 (0-2)
LLM_TOP_P=1.0
LLM_MAX_TOKENS=4096

# ── 系统 ──
USE_MOCK_LLM=False            # True=不使用真实 LLM
AGENT_PARALLEL=True           # Agent 并行分析
MAX_CONCURRENT_AGENTS=6       # 最大并行 Agent 数
CACHE_TTL_SECONDS=300         # 缓存 TTL (秒)
WARMUP_STOCKS=000001,600519   # 启动预热股票 (逗号分隔)
PREDICTION_CONFIDENCE_THRESHOLD=0.6
PREDICTION_CACHE_TTL=300

# ── 爬虫 ──
CRAWLER_REQUEST_INTERVAL=0.5  # 请求间隔 (秒)
CRAWLER_MAX_RETRIES=3         # 最大重试
CRAWLER_TIMEOUT=30            # 请求超时 (秒)
CRAWLER_ENABLE_EASTMONEY=True
CRAWLER_ENABLE_THS=True
```

### 12.3 数据库维护

```sql
-- 查看决策统计
SELECT decision, COUNT(*) FROM agent_decisions GROUP BY decision;

-- 查看热门股票
SELECT stock_code, COUNT(*) as cnt FROM agent_decisions
GROUP BY stock_code ORDER BY cnt DESC LIMIT 10;

-- 查看预测准确率
SELECT horizon,
       COUNT(*) as total,
       SUM(CASE WHEN direction_correct=1 THEN 1 ELSE 0 END) as correct
FROM prediction_records WHERE validated=1
GROUP BY horizon;

-- 清理 30 天前的决策
DELETE FROM agent_decisions WHERE created_at < datetime('now', '-30 days');
```

---

## 13. 附录

### 13.1 目录结构

```
MASS/
├── agent/                          # 多智能体核心
│   ├── agents/                     # 6 专业 Agent + BaseAgent
│   │   ├── base_agent.py           # 抽象基类 (ABC)
│   │   ├── ta_agent.py             # 技术面分析师
│   │   ├── fa_agent.py             # 基本面分析师
│   │   ├── ca_agent.py             # 资金面分析师
│   │   ├── sa_agent.py             # 情绪面分析师
│   │   ├── ma_agent.py             # 宏观策略师
│   │   └── ra_agent.py             # 风险控制官
│   ├── core/                       # 核心引擎
│   │   ├── orchestrator.py         # 编排器 (5阶段管线)
│   │   ├── blackboard.py           # 共享黑板 (分片锁)
│   │   ├── decision_engine.py      # 决策引擎 (加权投票)
│   │   ├── prediction_engine.py    # 预测引擎 v2.3
│   │   ├── debate.py               # 辩论引擎
│   │   ├── cache.py                # 缓存管理器
│   │   ├── validator.py            # 回测验证
│   │   └── exceptions.py           # 异常体系
│   ├── tools/                      # 工具层
│   │   ├── llm_client.py           # LLM 统一客户端
│   │   ├── stock_data_tool.py      # 股票数据工具 (单例)
│   │   ├── indicator_tool.py       # 技术指标计算
│   │   └── sentiment_tool.py       # 情感分析
│   ├── crawlers/                   # 爬虫框架
│   │   ├── base.py                 # 基类 (统一重试层)
│   │   ├── registry.py             # 注册表 (并发 fan-out)
│   │   ├── eastmoney.py            # 东方财富
│   │   ├── sina.py                 # 新浪财经
│   │   ├── akshare_crawler.py      # Akshare
│   │   ├── tx.py                   # 腾讯财经
│   │   ├── ths.py                  # 同花顺
│   │   └── utils.py                # UA轮换/JSONP解析
│   ├── models/                     # 数据模型
│   │   ├── agent_response.py       # Pydantic 决策模型
│   │   ├── prediction.py           # Pydantic 预测模型
│   │   └── database.py             # SQLite 封装 (WAL+连接池)
│   ├── auth/                       # 认证
│   │   └── manager.py              # Flask Session 认证
│   └── prompts/system/             # 可热更新提示词 (8 文件)
│
├── api/                            # API 层
│   ├── agent_bp.py                 # 32 端点 Blueprint
│   ├── middleware.py               # 限流/CORS/日志/错误
│   └── task_tracker.py             # 跨页面任务持久化
│
├── static/                         # 静态资源
│   ├── css/mass-theme.css          # Bloomberg 风格暗色主题
│   ├── js/charts/trading-charts.js # 图表引擎 + SSE 流处理
│   └── img/mass-logo.svg           # SVG Logo
│
├── templates/                      # Jinja2 模板
│   ├── base.html                   # 基础布局 (Logo + 导航 + Footer)
│   ├── login.html                  # 登录页
│   ├── dashboard.html              # 工作台
│   ├── agent_trading.html          # 个股诊断 (SSE 流式面板)
│   ├── agent_portfolio.html        # 组合分析
│   ├── agent_history.html          # 历史记录 (双标签页)
│   ├── agent_backtest.html         # 量化回测
│   ├── agent_monitor.html          # 系统监控
│   └── agent_report.html           # 研报生成
│
├── tests/                          # 141 项测试
│   ├── unit/                       # 59 单元测试
│   ├── integration/                # 42 集成测试
│   ├── e2e/                        # 1 端到端测试
│   └── stress/                     # 40 压力测试
│
├── docs/                           # 技术文档
│   ├── MASS_Technical_Report.md    # 本报告
│   ├── stress_test_report.md       # 压力测试报告
│   ├── prediction_enhancement_plan.txt  # 预测引擎增强方案
│   └── cross_page_stream_persistence_plan.txt  # 跨页面持久化方案
│
├── data/mass.db                    # SQLite 数据库
├── logs/                           # 日志目录
├── app.py                          # Flask 应用工厂
├── config.py                       # 全局配置
├── deploy.py                       # 生产部署脚本 (Waitress)
├── run.py                          # 开发启动脚本
├── requirements.txt                # 依赖清单
├── .env                            # 环境变量
└── .env.example                    # 配置模板
```

### 13.2 版本演进

| 版本 | 日期 | 里程碑 |
|------|------|--------|
| v1.0 | 2025 Q4 | 基础多智能体架构：6 Agent + Chairman，Mock 数据模式 |
| v2.0 | 2026 Q1 | SSE 流式诊断、组合分析、ECharts 图表体系、Bloomberg 风格 UI |
| v2.1 | 2026 Q2 | 认证系统、真实数据爬虫框架 (5 源)、字段级合并、异常体系 |
| v2.2 | 2026-05 | **十大性能优化**：并发 fan-out、orjson 5.5×、itertuples 25×、缓存扩面 |
| v2.3 | 2026-05 | **预测引擎增强**：分层 Prompt (-72% Token)、9 种分析模式、置信度校准、跨页面持久化、40 项压力测试 |

### 13.3 性能基准

| 指标 | v2.2 优化前 | v2.3 | 提升 |
|------|-----------|------|------|
| Token 消耗 (预测 short) | ~4,500 | ~1,250 | **-72%** |
| JSON 序列化 (50KB) | 2.5ms | 0.45ms | **5.5×** |
| RSV 指标计算 | 509μs | 157μs | **3.2×** |
| Prompt 构建 (10行K线) | 500μs | 20μs | **25×** |
| Cache 命中延迟 | ~5s | < 30ms | **166×** |
| 爬虫总耗时 | 10-15s | 3-5s | **3×** |
| 并发请求 | 4 (Waitress 限制) | 10+ (后台线程池解耦) | **2.5×** |
| 预测模式 | 3 | 9 | **3×** |
| 测试覆盖 | ~100 项 | 141 项 | **+41%** |

### 13.4 关键指标定义

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **confidence** | 原始 LLM 置信度 | LLM 输出 0-1 值 |
| **calibrated_confidence** | 数据质量校准后置信度 | raw × data_quality_factor |
| **data_quality_factor** | 数据完整度加权 | Σ(completeness_i × weight_i), min=0.3 |
| **processing_time** | 诊断全流程耗时 | Phase1 到 Phase6 的 wall-clock time |
| **cache_hit_rate** | 缓存命中率 | hits / (hits + misses) |
| **token_budget** | Prompt 输入 Token 估算 | (sys_chars + user_chars) / 2.5 |
| **prompt_tokens_estimated** | 同上 | 同上 |

---

*MASS v2.3 — Multi-Agent Stock System*
*技术白皮书 v2.0 — 生成于 2026-05-27*
*文档字符数: ~35,000 | 章节数: 13 | 图表数: 12+*
