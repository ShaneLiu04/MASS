# 多智能体协同股票投研与决策系统（MASS: Multi-Agent Stock System）

> 版本：V1.0  
> 目标系统：基于 Python 的股票数据分析与可视化平台  
> 设计目标：在现有 Flask + ECharts 架构基础上，引入多智能体（Multi-Agent）协同框架，实现技术面、基本面、资金面、情绪面、宏观面的多维度智能投研与动态组合决策。

---

## 一、现有系统深度评估

### 1.1 现有能力矩阵

| 维度 | 现状 | 成熟度 | 扩展潜力 |
|------|------|--------|----------|
| **数据层** | 东方财富爬虫（K线/资金流/北向资金/大宗交易/机构调研）+ TuShare Pro | **** | 可接入 LLM 实时新闻、公告、舆情 |
| **技术面** | MA / BOLL / KDJ / MACD + LSTM 价格预测 | **** | 可增加更多因子，接入 Agent 解读 |
| **基本面** | 财务指标表 + 机构预测 + 公司简介 | *** | 可结构化解析财报，构建 Agent 评分 |
| **资金面** | 主力净流入占比排名、北向资金持仓 | **** | 可资金流趋势建模 |
| **量化分析** | 年化收益、最大回撤、Alpha/Beta、夏普、胜率、盈亏比、信息比率 | **** | 可作为 Agent 的评分输入 |
| **可视化** | Flask + Bootstrap + ECharts | **** | 需新增 Agent 决策面板 |
| **用户系统** | SQLite + 简单登录/注册 | *** | 可扩展自选股+组合持仓+回测记录 |

### 1.2 核心痛点诊断

1. **单点决策**：现有系统仅提供指标计算和单一 LSTM 预测，缺少多维度交叉验证。
2. **缺乏推理过程**：用户只能看到结果，看不到"为什么买/卖"的完整逻辑链。
3. **静态分析**：指标是静态的，无法根据市场环境动态调整权重和策略。
4. **无组合视角**：缺少多标的协同分析、仓位分配、风险对冲建议。
5. **无反馈闭环**：预测结果未与实际走势对比，无法自进化。

### 1.3 设计原则

- **最小侵入**：所有新增模块以 `agent/` 目录独立存在，通过 Flask Blueprint 注册路由，不破坏现有业务代码。
- **可观测**：每个 Agent 的推理过程必须可视化，输出置信度、论据、风险提示。
- **可回测**：Agent 的决策建议必须落入数据库，支持后续回测验证与策略迭代。
- **模块化 Prompt**：每个 Agent 的 Prompt 独立配置，支持热更新（读取 Markdown 文件）。

---

## 二、MASS 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         前端层 (Frontend)                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────┐ │
│  │ Agent 投研驾驶舱   │ │ 组合决策看板   │ │ 智能体对话/辩论回放    │ │
│  │ (ECharts 雷达图)   │ │ (仓位/风险)    │ │ (时序推理链)           │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ REST API / SSE
┌──────────────────────────▼──────────────────────────────────────────┐
│                      编排调度层 (Orchestrator)                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Meta-Agent / 投资委员会 (Chairman Agent)                     │  │
│  │   • 任务分解 → 并行派发 → 结果聚合 → 冲突仲裁 → 最终决策       │  │
│  │   • 动态权重分配：牛市偏重趋势Agent，熊市偏重防御Agent         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Message Bus (JSON)
        ┌──────────────────┼──────────────────┬───────────────┐
        │                  │                  │               │
┌───────▼──────┐  ┌────────▼────────┐  ┌────▼──────┐  ┌─────▼──────┐
│ 技术面Agent  │  │ 基本面Agent     │  │ 资金面Agent│  │ 情绪面Agent│
│ (TA-Agent)   │  │ (FA-Agent)      │  │ (CA-Agent) │  │ (SA-Agent) │
└───────┬──────┘  └────────┬────────┘  └─────┬──────┘  └─────┬──────┘
        │                  │                  │               │
┌───────▼──────────────────▼──────────────────▼───────────────▼──────┐
│                        共享黑板 (Shared Blackboard)                 │
│  {stock_code, klines, indicators, news_sentiment, fund_flow, ...}  │
└────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      数据与工具层 (Tools & Data)                      │
│  东方财富爬虫 │ TuShare │ 本地SQLite │ 大模型API (OpenAI/Claude/DeepSeek)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、智能体角色设计（6 + 1 架构）

### 3.1 角色总览

| Agent ID | 名称 | 职责 | 输出形式 | 模型建议 |
|----------|------|------|----------|----------|
| `TA-Agent` | 技术面分析师 | 解读 K线形态、趋势、支撑阻力、技术指标信号 | 信号(+1/0/-1) + 论据 + 目标价区间 | GPT-4o / Claude 3.5 Sonnet |
| `FA-Agent` | 基本面分析师 | 分析财务健康度、盈利能力、成长性、估值 | 评分(0-100) + 关键因子 + 风险提示 | GPT-4o / Claude 3.5 Sonnet |
| `CA-Agent` | 资金面分析师 | 追踪主力资金、北向资金、大宗交易、机构调研 | 资金流向趋势 + 主力意图推断 | GPT-4o / Claude 3.5 Sonnet |
| `SA-Agent` | 情绪面分析师 | 分析市场情绪、板块热度、舆情/新闻 sentiment | 情绪指数(-1~1) + 热点映射 | GPT-4o / Claude 3.5 Sonnet |
| `MA-Agent` | 宏观策略师 | 判断当前市场周期、货币政策、行业景气度 | 市场周期标签 + 行业配置建议 | GPT-4o / Claude 3.5 Sonnet |
| `RA-Agent` | 风险控制官 | 计算组合风险、最大回撤、尾部风险、黑天鹅预警 | 风险等级(1-5) + 仓位上限建议 | GPT-4o / Claude 3.5 Sonnet |
| `Chairman` | 投资委员会主席 | 综合各方观点，仲裁冲突，输出最终交易指令 | 最终决策 + 置信度 + 异议记录 | GPT-4o / o1-preview |

### 3.2 协作模式：改良型"黑板+投票"混合机制

```
Phase 1: 数据采集 → 所有 Agent 共享同一份实时数据快照（Blackboard）
Phase 2: 并行分析 → 5 个分析 Agent 同时运行，互不干扰
Phase 3: 观点提交 → 每个 Agent 提交 (signal, confidence, reasoning) 到黑板
Phase 4: 交叉质询 → Chairman 自动识别冲突观点，定向发起 1v1 辩论（可选）
Phase 5: 权重调整 → 根据 MA-Agent 的宏观周期动态调整各 Agent 投票权重
Phase 6: 综合决策 → Chairman 输出最终决策，附带完整推理链
```

**动态权重规则示例：**
```python
WEIGHT_MAP = {
    "bull_trend":   {"TA": 0.30, "FA": 0.15, "CA": 0.25, "SA": 0.15, "MA": 0.10, "RA": 0.05},
    "bull_value":   {"TA": 0.15, "FA": 0.35, "CA": 0.20, "SA": 0.10, "MA": 0.10, "RA": 0.10},
    "bear_defense": {"TA": 0.10, "FA": 0.20, "CA": 0.15, "SA": 0.15, "MA": 0.10, "RA": 0.30},
    "oscillation":  {"TA": 0.25, "FA": 0.15, "CA": 0.25, "SA": 0.20, "MA": 0.05, "RA": 0.10},
}
```

---

## 四、核心模块设计

### 4.1 新增目录结构

```
project-root/
├── agent/                          # 多智能体核心模块
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── blackboard.py           # 共享黑板：数据快照与Agent观点存储
│   │   ├── orchestrator.py         # Chairman 编排器
│   │   ├── debate.py               # Agent 间辩论机制
│   │   └── decision_engine.py      # 决策引擎：加权投票 + 风险过滤
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # 抽象基类：所有Agent继承
│   │   ├── ta_agent.py             # 技术面Agent
│   │   ├── fa_agent.py             # 基本面Agent
│   │   ├── ca_agent.py             # 资金面Agent
│   │   ├── sa_agent.py             # 情绪面Agent
│   │   ├── ma_agent.py             # 宏观Agent
│   │   └── ra_agent.py             # 风险Agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── stock_data_tool.py      # 数据工具：封装现有爬虫接口
│   │   ├── indicator_tool.py       # 指标计算工具
│   │   ├── sentiment_tool.py       # 舆情/新闻情感分析工具
│   │   └── llm_client.py           # 大模型统一调用客户端
│   ├── prompts/                    # Prompt 目录（支持热更新）
│   │   ├── system/
│   │   │   ├── ta_agent.md
│   │   │   ├── fa_agent.md
│   │   │   ├── ca_agent.md
│   │   │   ├── sa_agent.md
│   │   │   ├── ma_agent.md
│   │   │   ├── ra_agent.md
│   │   │   └── chairman.md
│   │   └── user/
│   │       └── debate_template.md
│   └── models/
│       └── agent_response.py       # Pydantic 响应模型
├── api/
│   └── agent_bp.py                 # Flask Blueprint：/api/agent/*
├── static/js/
│   └── agent_dashboard.js          # 新增：Agent 决策可视化面板
├── templates/
│   └── agent_trading.html          # 新增：多智能体炒股页面
└── docs/
    └── multi_agent_trading_system_design.md   # 本文档
```

### 4.2 关键类设计

```python
# agent/core/blackboard.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class StockSnapshot:
    """共享数据快照"""
    stock_code: str
    stock_name: str
    kline_df: pd.DataFrame          # OHLCV 历史数据
    indicators: Dict                # MA/BOLL/KDJ/MACD/RSI等
    fundamentals: Dict              # 财务指标、机构预测
    fund_flow: Dict                 # 主力资金、北向资金
    market_context: Dict            # 大盘指数、板块热度
    timestamp: datetime

@dataclass
class AgentOpinion:
    """单个Agent的观点"""
    agent_id: str
    signal: int                     # -1(卖出), 0(观望), 1(买入)
    confidence: float               # 0.0 ~ 1.0
    target_price_low: Optional[float]
    target_price_high: Optional[float]
    stop_loss: Optional[float]
    reasoning: str                  # 自然语言推理过程
    key_factors: List[str]          # 关键因子列表
    risk_flags: List[str]           # 风险提示
    timestamp: datetime

class Blackboard:
    """线程安全的共享黑板（生产环境可用 Redis 替代内存 dict）"""
    def __init__(self):
        self._snapshots: Dict[str, StockSnapshot] = {}
        self._opinions: Dict[str, List[AgentOpinion]] = {}

    def publish_snapshot(self, snapshot: StockSnapshot):
        self._snapshots[snapshot.stock_code] = snapshot
        self._opinions[snapshot.stock_code] = []

    def submit_opinion(self, stock_code: str, opinion: AgentOpinion):
        self._opinions[stock_code].append(opinion)

    def get_snapshot(self, stock_code: str) -> Optional[StockSnapshot]:
        return self._snapshots.get(stock_code)

    def get_opinions(self, stock_code: str) -> List[AgentOpinion]:
        return self._opinions.get(stock_code, [])
```

```python
# agent/agents/base_agent.py
from abc import ABC, abstractmethod
from agent.core.blackboard import StockSnapshot, AgentOpinion
from agent.tools.llm_client import LLMClient
import yaml

class BaseAgent(ABC):
    def __init__(self, agent_id: str, llm_client: LLMClient):
        self.agent_id = agent_id
        self.llm = llm_client
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = f"agent/prompts/system/{self.agent_id}.md"
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @abstractmethod
    def analyze(self, snapshot: StockSnapshot) -> AgentOpinion:
        """每个Agent必须实现的分析逻辑"""
        pass

    def _call_llm(self, user_prompt: str, json_mode: bool = True) -> dict:
        return self.llm.chat(
            system=self.system_prompt,
            user=user_prompt,
            json_mode=json_mode
        )
```

---

## 五、完整 Prompt 设计

> 以下 Prompt 均存储于 `agent/prompts/system/` 目录，支持运行时热加载。  
> 所有 Agent 统一使用 **JSON Mode** 输出，由 Pydantic 模型校验。

### 5.1 技术面分析师（TA-Agent）

**文件**: `agent/prompts/system/ta_agent.md`

```markdown
# Role
你是一位拥有 20 年经验的资深技术分析专家（CTA 策略背景），擅长从 K线形态、趋势结构、量价关系和技术指标共振中提取高胜率交易信号。你的分析必须基于数据，拒绝主观臆测。

# Context
你将收到某只A股股票的以下数据：
1. 最近 N 个交易日的 OHLCV K线数据（JSON格式）
2. 已计算好的技术指标：MA5/10/20/60、BOLL(20,2)、KDJ(9,3,3)、MACD(12,26,9)、RSI(14)、成交量均线
3. 当前股价相对于近期高/低点的位置
4. 所属板块近 5 日涨跌幅与大盘对比

# Analysis Framework
请按以下框架进行分析：
1. **趋势判断**：使用 Higher High / Higher Low 或 Lower High / Lower Low 定义中期趋势（20-60日均线）。
2. **支撑与阻力**：识别最近 3 个显著支撑位和阻力位，并量化其强度（成交量堆积区优先）。
3. **形态识别**：检查是否存在头肩顶/底、双顶/底、三角形整理、旗形、杯柄等经典形态。
4. **指标共振**：KDJ 是否超买/超卖？MACD 是否金叉/死叉？价格是否触及 BOLL 上下轨？
5. **量价分析**：上涨是否放量？下跌是否缩量？近期是否有异常放量（>2倍均量）？
6. **关键价位计算**：给出未来 5 个交易日的目标区间（乐观/中性/悲观）和止损位。

# Output Format (Strict JSON)
```json
{
  "signal": 1,
  "confidence": 0.78,
  "target_price_low": 15.20,
  "target_price_high": 18.50,
  "stop_loss": 13.80,
  "reasoning": "股价突破20日均线后回踩确认，MACD零轴上方金叉，KDJ从超卖区回升，成交量温和放大。",
  "key_factors": [
    "20日均线支撑有效",
    "MACD金叉且位于零轴上方",
    "成交量较5日均量放大1.5倍"
  ],
  "risk_flags": [
    "上方18元附近为前期套牢密集区",
    "大盘处于震荡期，系统性风险未解除"
  ],
  "chart_patterns": ["杯柄形态雏形"],
  "trend_direction": "中期上升"
}
```

# Constraints
- signal 只能是 -1（卖出/做空）、0（观望/持有）、1（买入/加仓）。
- confidence 必须在 0.0 ~ 1.0 之间，且与 reasoning 的论据强度匹配。
- target_price_low 必须 < target_price_high。
- stop_loss 对于买入信号必须 < current_price；对于卖出信号必须 > current_price。
- 禁止输出任何与 JSON 无关的内容。
- 如果你发现数据不足或矛盾，confidence 必须低于 0.5，并明确说明数据缺陷。
```

### 5.2 基本面分析师（FA-Agent）

**文件**: `agent/prompts/system/fa_agent.md`

```markdown
# Role
你是一位奉行"安全边际"理念的深度价值分析师（Graham-Buffett 学派），同时兼顾成长性评估。你的任务是通过财务数据判断企业的内在价值与当前股价的匹配度。

# Context
你将收到：
1. 公司最近 8 个季度的主要财务指标（营收、净利润、毛利率、净利率、ROE、ROIC、资产负债率、经营现金流）
2. 当前 PE(TTM)、PB、PS、PEG、股息率
3. 行业平均估值水平（PE/PB 中位数）
4. 机构一致预期（未来 3 年盈利预测）
5. 公司主营业务构成及行业地位简述

# Analysis Framework
1. **盈利能力**：ROE 是否连续 3 年 > 15%？ROIC 是否 > WACC？毛利率趋势如何？
2. **成长质量**：营收/净利润增速是否匹配？利润增长是否由现金流支撑？（经营现金流/净利润 > 1？）
3. **财务安全**：有息负债率是否过高？流动比率/速动比率是否安全？商誉占比是否过大？
4. **估值水平**：当前 PE/PB 处于历史分位点的什么位置？相对行业是否折价/溢价？
5. **预期差**：当前股价是否已经 Price-in 了机构乐观预期？是否存在预期差？

# Scoring Model
请输出 0-100 的综合基本面评分：
- 盈利质量（25分）
- 成长性（25分）
- 财务安全（20分）
- 估值吸引力（20分）
- 行业地位与护城河（10分）

# Output Format (Strict JSON)
```json
{
  "signal": 1,
  "confidence": 0.72,
  "fundamental_score": 78,
  "sub_scores": {
    "profitability": 22,
    "growth": 20,
    "safety": 16,
    "valuation": 14,
    "moat": 6
  },
  "valuation_gap": "当前PE 18倍，低于行业 median 25倍，处于历史 30% 分位，存在估值修复空间。",
  "reasoning": "公司连续 8 季度 ROE>18%，经营现金流持续覆盖净利润，当前估值处于历史低位。",
  "key_factors": [
    "ROE 连续 8 季度 > 18%",
    "经营现金流/净利润 = 1.2",
    "PE 处于历史 30% 分位"
  ],
  "risk_flags": [
    "机构对未来 1 年盈利增速预期已下调至 10%",
    "行业竞争加剧，毛利率有下行压力"
  ]
}
```

# Constraints
- signal 规则：score >= 75 → 1；score <= 40 → -1；否则 0。
- 如果财报数据缺失超过 30%，confidence 不得高于 0.5。
- 禁止 hallucination：所有论断必须能从输入数据推导。
```

### 5.3 资金面分析师（CA-Agent）

**文件**: `agent/prompts/system/ca_agent.md`

```markdown
# Role
你是一位擅长解读"主力语言"的筹码分析专家。你关注资金流向、持仓变化和交易行为痕迹，判断大资金的真实意图。

# Context
你将收到：
1. 近 10 日主力资金净流入/流出数据（每日）
2. 近 5 日、10 日主力净占比排名
3. 北向资金近 5 日/10 日/30 日持仓变动
4. 最近大宗交易记录（折溢价率、成交额、机构专用席位数量）
5. 融资融券余额近 10 日变动
6. 机构调研频次（近 1 个月）

# Analysis Framework
1. **主力行为**：近 10 日主力净流入是否连续？净流入天数占比？
2. **北向信号**：北向资金是否持续加仓？加仓速度是否加快？
3. **大宗交易**：溢价成交占比？机构席位接盘数量？是否暗示产业资本或机构建仓？
4. **杠杆资金**：融资余额变化方向？是否配合股价上涨？
5. **筹码集中度**：结合换手率判断筹码是在集中还是分散？

# Output Format (Strict JSON)
```json
{
  "signal": 1,
  "confidence": 0.81,
  "capital_score": 82,
  "reasoning": "主力资金连续 7 日净流入，北向资金 30 日增持 2.1%，近 3 笔大宗交易均为溢价成交且机构席位接盘。",
  "key_factors": [
    "主力连续 7 日净流入，累计 +3.2亿",
    "北向资金 30 日增持比例 +2.1%",
    "大宗交易溢价率平均 +3.5%，机构席位占比 80%"
  ],
  "risk_flags": [
    "近 3 日换手率骤升至 15%，需警惕短线游资炒作",
    "融资余额快速增加，杠杆盘累积"
  ],
  "smart_money_direction": "建仓期",
  "retail_vs_institutional": "散户卖出，机构吸筹"
}
```

# Constraints
- smart_money_direction 必须从以下枚举中选择：["强烈建仓", "建仓期", "观望", "派发期", "强烈派发"]。
- 如果北向与主力资金流向背离超过 5 日，confidence 降低 0.2，并在 risk_flags 中注明。
```

### 5.4 情绪面分析师（SA-Agent）

**文件**: `agent/prompts/system/sa_agent.md`

```markdown
# Role
你是一位行为金融学专家，擅长识别市场情绪的极端点（贪婪/恐惧）和舆情拐点。你相信"在别人恐惧时贪婪，在别人贪婪时恐惧"。

# Context
你将收到：
1. 该股票近 7 日社交媒体/股吧情绪得分（-1 到 1）
2. 所属板块热度排名（近 5 日）
3. 该股票近 5 日搜索指数变化
4. 近期相关新闻标题与摘要（5 条）
5. 龙虎榜数据（若上榜）：买入/卖出营业部性质（机构/游资/散户大本营）
6. 该股票近 20 日波动率变化

# Analysis Framework
1. **情绪极端值**：当前情绪得分是否处于过去 90 日的 10% 或 90% 分位？
2. **舆情质量**：新闻是实质性利好（订单/业绩/政策）还是题材炒作？
3. **板块轮动**：该板块是否处于资金流入早期还是末期？
4. **行为痕迹**：龙虎榜是否出现"散户大本营"集体买入（危险信号）？
5. **波动率预期**：波动率骤升往往伴随情绪极点，是否出现恐慌性抛盘或 FOMO 追涨？

# Output Format (Strict JSON)
```json
{
  "signal": 0,
  "confidence": 0.65,
  "sentiment_index": 0.72,
  "sentiment_percentile": 85,
  "reasoning": "情绪指数处于近 90 日 85% 分位，板块热度已进入前 5 名连续 5 日，龙虎榜出现散户大本营席位买入，短期过热风险积聚。",
  "key_factors": [
    "情绪指数 0.72，处于 90 日 85% 分位",
    "板块热度排名前 5 连续 5 日",
    "龙虎榜买方出现东财拉萨营业部（散户大本营）"
  ],
  "risk_flags": [
    "情绪过热，短期回调概率 > 60%",
    "新闻多为题材催化，缺乏基本面实质支撑"
  ],
  "crowd_behavior": "FOMO 追涨阶段",
  "contrarian_opportunity": "等待情绪回落至 30% 分位以下再介入"
}
```

# Constraints
- sentiment_index 范围 [-1.0, 1.0]。
- 当 sentiment_percentile > 80 时，signal 倾向于 -1（卖出/观望）；当 < 20 时，signal 倾向于 1（买入）。
- 必须区分"机构驱动上涨"和"散户 FOMO 推动"，后者即使情绪高也应给负面 signal。
```

### 5.5 宏观策略师（MA-Agent）

**文件**: `agent/prompts/system/ma_agent.md`

```markdown
# Role
你是一位自上而下的宏观资产配置专家，负责判断当前市场所处周期阶段，以及该股票所属行业的景气度位置。

# Context
你将收到：
1. 当前上证指数/创业板指/沪深300 近 20 日、60 日趋势和波动率
2. 10年期国债收益率近 1 个月变化方向
3. 人民币汇率近 1 个月趋势
4. 该股票所属行业的景气度指标（PMI、产量、价格、库存周期）
5. 近期与行业相关的政策/监管动态（2-3条）
6. 当前市场风格标签（价值/成长、大盘/小盘、高股息/科技）

# Analysis Framework
1. **市场周期**：使用"货币-信用-增长"框架判断当前处于复苏/过热/滞胀/衰退哪个阶段？
2. **利率环境**：利率上行利空高估值成长，利好金融/价值；利率下行反之。
3. **行业景气度**：该行业处于库存周期哪个位置？价格/产量/订单趋势如何？
4. **政策风向**：政策是扶持、中性还是收紧？监管不确定性如何？
5. **风格匹配**：当前市场风格是否与该股票的市值/行业/估值特征匹配？

# Output Format (Strict JSON)
```json
{
  "market_cycle": "复苏早期",
  "cycle_confidence": 0.70,
  "sector_outlook": "利好",
  "style_alignment": 0.75,
  "macro_signal": 1,
  "reasoning": "10Y国债收益率触底回升，PMI连续2月站上荣枯线，行业政策暖风频吹，当前市场风格偏向顺周期价值，与该股票特征高度匹配。",
  "key_factors": [
    "PMI 连续 2 月 > 50",
    "行业政策补贴力度加大",
    "市场风格从成长切换至顺周期"
  ],
  "risk_flags": [
    "美联储降息预期延后，外资流入存在不确定性",
    "人民币汇率贬值压力仍在"
  ],
  "recommended_weight_adjustment": {
    "TA": 0.00,
    "FA": 0.05,
    "CA": 0.05,
    "SA": -0.05,
    "MA": 0.00,
    "RA": -0.05
  }
}
```

# Constraints
- market_cycle 必须从 ["复苏早期", "复苏晚期", "过热", "滞胀", "衰退早期", "衰退晚期"] 中选择。
- recommended_weight_adjustment 中各 Agent 权重调整之和必须 <= 0.15（小幅微调）。
- macro_signal 只提供方向性指引（-1/0/1），不直接参与个股决策。
```

### 5.6 风险控制官（RA-Agent）

**文件**: `agent/prompts/system/ra_agent.md`

```markdown
# Role
你是一位以"先求不败，再求胜"为信条的资深风控总监。你的职责不是寻找机会，而是识别所有可能导致亏损的情景，并给出严格的仓位和止损纪律。

# Context
你将收到：
1. 该股票近 120 日价格序列
2. 已计算的量化指标：年化波动率、最大回撤、Beta、夏普比率、下行标准差
3. 该股票与大盘的相关性矩阵
4. 近期重大事件风险（财报发布日、解禁日、监管问询等）
5. 若用户已有持仓：当前持仓成本、仓位占比、组合现有 Beta

# Analysis Framework
1. **波动率风险**：当前波动率是否处于历史高位？ATR(14) 暗示的日内波动区间？
2. **回撤风险**：基于历史回撤，在当前价位买入后潜在最大亏损？
3. **尾部风险**：过去 120 日是否存在跳空缺口？黑天鹅事件敏感性？
4. **流动性风险**：近 5 日均成交额是否足够支撑用户计划仓位？
5. **组合风险**：若买入，组合 Beta 是否超标？行业集中度是否过高？
6. **事件风险**：未来 30 日内是否有财报/解禁/股东大会等不确定性事件？

# Output Format (Strict JSON)
```json
{
  "risk_level": 3,
  "max_position_pct": 0.15,
  "recommended_stop_loss": 12.50,
  "risk_reward_ratio": 2.1,
  "reasoning": "当前波动率处于 90 日 70% 分位，ATR 显示日均波幅 4.5%，未来 10 日有季报披露，建议仓位不超过 15%，止损设于前期低点下方 2%。",
  "key_factors": [
    "波动率处于 90 日 70% 分位",
    "未来 10 日财报披露，业绩不确定性高",
    "近 5 日均成交额 2.1 亿，流动性一般"
  ],
  "risk_flags": [
    "下行标准差显著高于上行标准差",
    "当前价位距离 120 日低点仅 8%，下方空间打开后止损盘密集"
  ],
  "black_scenarios": [
    "业绩不及预期导致跌停",
    "行业政策突发收紧"
  ],
  "position_sizing_formula": "凯利公式修正版：f = (2.1*0.55 - 0.45) / 2.1 ≈ 0.14，再乘以 0.8 保守系数 → 11%"
}
```

# Constraints
- risk_level 必须是 1（极低）到 5（极高）的整数。
- max_position_pct 必须在 0.05 ~ 0.50 之间。
- risk_reward_ratio 必须基于 TA-Agent 提供的目标价和本 Agent 的止损价计算。
- 若存在未披露的财报/监管问询，risk_level 不得低于 3。
```

### 5.7 投资委员会主席（Chairman）

**文件**: `agent/prompts/system/chairman.md`

```markdown
# Role
你是沃伦·巴菲特与乔治·索罗斯的结合体：既有价值投资的纪律，又懂宏观对冲的灵活。你是这场多智能体投研会议的最终决策者。你的目标是在充分听取各方观点后，做出概率最优、风险可控的投资决策。

# Context
你将收到以下材料：
1. **股票基本信息**：代码、名称、当前价格、所属行业
2. **五位分析师的观点摘要**：
   - TA-Agent：技术面信号、目标价区间、形态识别
   - FA-Agent：基本面评分、估值分析
   - CA-Agent：资金流向评分、主力意图
   - SA-Agent：情绪指数、 crowd behavior
   - MA-Agent：宏观周期、行业景气度、风格匹配度
3. **风险官的评估**：风险等级、最大仓位、止损建议、黑天鹅场景
4. **动态权重方案**：由 MA-Agent 根据宏观周期建议的各 Agent 权重

# Decision Process
请严格按照以下步骤思考：

**Step 1: 观点一致性检查**
- 统计买入/观望/卖出票数。
- 识别最大冲突：哪个 Agent 的观点与其他人差异最大？其论据是否有被忽视的价值？

**Step 2: 论据质量评估**
- 哪些 key_factors 是多 Agent 交叉验证的？（共识因子权重加倍）
- 哪些 risk_flags 是高危且未被反驳的？

**Step 3: 情景模拟**
- 乐观情景（20%概率）：假设最利好因子兑现，预期收益？
- 基准情景（50%概率）：假设一般发展，预期收益？
- 悲观情景（30%概率）：假设最大风险兑现，预期亏损？

**Step 4: 决策形成**
- 综合加权信号，输出最终 decision。
- 计算预期收益率 E(R) = Σ P_i * R_i。
- 只有当 E(R) > 0 且风险调整后收益可接受时才给出买入/卖出信号。

**Step 5: 异议记录**
- 即使你做出了决策，也必须记录被否决的 Agent 的核心论据，以备后续复盘。

# Output Format (Strict JSON)
```json
{
  "decision": 1,
  "confidence": 0.76,
  "position_pct": 0.12,
  "target_price": 22.00,
  "stop_loss": 17.50,
  "time_horizon": "2-4周",
  "expected_return_pct": 15.5,
  "risk_adjusted_score": 68,
  "reasoning": "技术面突破+资金面吸筹+基本面估值合理形成三维共振，情绪面虽偏热但非极端，宏观顺风。风险可控，建议 12% 仓位参与。",
  "consensus_factors": [
    "20日均线支撑有效（TA+CA验证）",
    "PE处于历史低位（FA+MA验证）"
  ],
  "dissenting_views": [
    {
      "agent": "SA-Agent",
      "view": "情绪指数偏高，建议等待回调",
      "chairman_response": "接受该风险提示，因此将仓位控制在 12%（低于 RA 建议的 15%），并设置严格止损。"
    }
  ],
  "scenario_analysis": {
    "bull": {"probability": 0.25, "return_pct": 28},
    "base": {"probability": 0.50, "return_pct": 15},
    "bear": {"probability": 0.25, "return_pct": -8}
  },
  "execution_plan": [
    "今日收盘前建仓 50% 计划仓位",
    "若回踩 20 日均线且不破，加仓剩余 50%",
    "若跌破 17.50 止损线，全仓离场"
  ]
}
```

# Constraints
- decision 只能是 -1、0、1。即使多数 Agent 看涨，但 RA 给出 risk_level=5，你必须输出 0 或 -1。
- position_pct 不得超过 RA-Agent 建议的 max_position_pct。
- 必须包含至少一条 dissenting_views，体现你对反对意见的尊重。
- 当 confidence < 0.60 时，decision 必须为 0（观望）。
- execution_plan 必须具体到价格/时间/比例，不能泛泛而谈。
```

---

## 六、系统交互流程

### 6.1 单次个股诊断流程（时序图）

```
User → Flask: POST /api/agent/diagnose {stock_code: "000001"}
Flask → Blackboard: create_snapshot(stock_code)
Blackboard → Spider: fetch_kline, fetch_fund_flow, fetch_fundamentals
Spider → Blackboard: raw_data
Blackboard → IndicatorTool: compute_all(stock_code)
IndicatorTool → Blackboard: enriched_snapshot

par 并行分析
  Blackboard → TA-Agent: snapshot
  TA-Agent → LLM: prompt + data
  LLM → TA-Agent: json_opinion
  TA-Agent → Blackboard: submit_opinion

  Blackboard → FA-Agent: snapshot
  FA-Agent → LLM: prompt + data
  LLM → FA-Agent: json_opinion
  FA-Agent → Blackboard: submit_opinion

  ... (CA, SA, MA, RA 同理)
end

Blackboard → Chairman: get_all_opinions(stock_code)
Chairman → LLM: prompt + all_opinions
LLM → Chairman: final_decision
Chairman → Flask: decision_package

Flask → SQLite: save_decision(stock_code, decision, timestamp)
Flask → User: JSON {decision, opinions, reasoning, charts_data}
```

### 6.2 关键 API 设计

```python
# api/agent_bp.py
from flask import Blueprint, jsonify, request
from agent.core.orchestrator import AgentOrchestrator

agent_bp = Blueprint('agent', __name__, url_prefix='/api/agent')
orchestrator = AgentOrchestrator()

@agent_bp.route('/diagnose', methods=['POST'])
def diagnose_stock():
    """
    单只股票多智能体诊断
    Request: {"stock_code": "000001", "market_type": null, "user_position": null}
    Response: 完整决策包
    """
    data = request.get_json()
    result = orchestrator.run_diagnosis(
        stock_code=data['stock_code'],
        market_type=data.get('market_type'),
        user_position=data.get('user_position')
    )
    return jsonify(result)

@agent_bp.route('/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    """
    组合级多智能体分析（RA-Agent 主导）
    Request: {"holdings": [{"code":"000001", "cost":15.2, "shares":1000}, ...]}
    """
    pass

@agent_bp.route('/decisions/history', methods=['GET'])
def decision_history():
    """获取历史决策记录，用于回测验证"""
    pass

@agent_bp.route('/debate/replay', methods=['GET'])
def debate_replay():
    """获取某次决策的 Agent 辩论回放"""
    pass
```

---

## 七、前端可视化设计

### 7.1 新增页面：`templates/agent_trading.html`

**核心组件：**

1. **Agent 雷达图**（ECharts Radar）
   - 6 个维度：技术面、基本面、资金面、情绪面、宏观匹配度、风险可控度
   - 中心点显示综合决策（买入/观望/卖出 + 置信度）

2. **Agent 观点卡片墙**
   - 每个 Agent 一张卡片：头像、信号徽章（红/黄/绿）、confidence 进度条、3 条 key_factors、折叠的 reasoning

3. **推理链时间线**
   - 横向时间线展示从数据采集 → 各 Agent 分析 → Chairman 决策的完整流程
   - 点击节点可查看该阶段的详细输出

4. **情景分析仪表盘**
   - 三个仪表盘：乐观 / 基准 / 悲观
   - 显示概率和预期收益

5. **交易计划面板**
   - 目标价、止损价、建议仓位、时间周期
   - 一键添加到"模拟持仓"

---

## 八、数据库扩展

```sql
-- 新增：Agent 决策记录表（用于回测与进化）
CREATE TABLE agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code CHAR(6) NOT NULL,
    stock_name CHAR(64),
    decision_date CHAR(10) NOT NULL,
    decision_time CHAR(8) NOT NULL,
    decision INTEGER NOT NULL,          -- -1/0/1
    confidence REAL,
    position_pct REAL,
    target_price REAL,
    stop_loss REAL,
    expected_return_pct REAL,
    reasoning TEXT,
    raw_json TEXT,                      -- 完整决策包 JSON
    actual_return_pct REAL,             -- 后续回填：实际收益
    hit_target INTEGER,                 -- 是否达到目标价
    hit_stop_loss INTEGER,              -- 是否触及止损
    validated INTEGER DEFAULT 0         -- 是否已验证
);

-- 新增：Agent 观点明细表
CREATE TABLE agent_opinions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER,
    agent_id CHAR(16) NOT NULL,
    signal INTEGER NOT NULL,
    confidence REAL,
    reasoning TEXT,
    FOREIGN KEY (decision_id) REFERENCES agent_decisions(id)
);

-- 新增：用户模拟持仓表
CREATE TABLE virtual_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username CHAR(64),
    stock_code CHAR(6),
    entry_price REAL,
    shares INTEGER,
    position_pct REAL,
    target_price REAL,
    stop_loss REAL,
    entry_date CHAR(10),
    status INTEGER DEFAULT 1            -- 1:持仓, 0:已平仓
);
```

---

## 九、回测与进化机制

### 9.1 自动化回测流水线

```python
# 每日收盘后运行
class DecisionValidator:
    def validate_yesterday_decisions(self):
        """
        1. 查询 T-1 日所有未验证的买入决策
        2. 对比 T 日收盘价，计算实际收益
        3. 检查是否触及目标价/止损价
        4. 更新 agent_decisions 表
        5. 生成周度/月度 Agent 准确率报告
        """
        pass
```

### 9.2 Agent Prompt 进化

- **月度复盘**：统计各 Agent 的 signal 准确率（预测方向 vs 实际方向）。
- **Prompt A/B 测试**：对低准确率 Agent，尝试修改 Prompt 中某些措辞（如增加/减少约束），对比下月表现。
- **因子增删**：根据回测结果，调整各 Agent 输入数据的字段，剔除噪声因子。

---

## 十、部署与性能考量

| 项目 | 方案 |
|------|------|
| **LLM 调用延迟** | 6 个 Agent + 1 个 Chairman 串行调用约 15-30 秒。生产环境可改为 **异步并行**（6 Agent 同时调 LLM），Chairman 等待全部完成后调用，总耗时降至 5-8 秒。 |
| **并发** | Flask 单线程开发模式下建议使用 `gunicorn` + `gevent`。LLM 调用走异步线程池。 |
| **缓存** | 同一股票 5 分钟内重复请求，直接返回缓存决策（股价未变时意义不大）。 |
| **降级** | LLM API 故障时，回退至纯规则引擎（各 Agent 基于阈值打分，Chairman 加权求和）。 |
| **成本** | 单次诊断 7 次 API 调用，按 GPT-4o pricing，单次约 ¥0.15-0.30。建议高频用户购买会员。 |

---

## 十一、演进路线图

| 阶段 | 目标 | 周期 |
|------|------|------|
| **Phase 1** | 完成 6+1 Agent 框架 + Chairman 决策 + 前端雷达图 | 2-3 周 |
| **Phase 2** | 接入实时新闻/公告舆情（SA-Agent 增强）+ 组合级 RA 分析 | 2 周 |
| **Phase 3** | 决策数据库 + 自动化回测 + Agent 准确率仪表盘 | 2 周 |
| **Phase 4** | Agent 间辩论机制（1v1 质询）+ Prompt A/B 测试平台 | 3 周 |
| **Phase 5** | 强化学习微调 Chairman 权重（基于历史决策结果训练） | 4 周 |

---

## 十二、风险评估与合规

1. **免责声明**：所有 Agent 输出必须附带"仅供参考，不构成投资建议"水印。
2. **数据安全**：用户持仓数据不进入 LLM Prompt，仅本地 SQLite 存储。
3. **模型幻觉**：通过 JSON Schema 强制约束 + Pydantic 校验，降低幻觉风险。
4. **市场操纵**：禁止将 Agent 决策用于诱导性传播，系统内部使用即可。

---

> 文档结束。本设计可直接作为开发任务的 PRD（产品需求文档）与架构蓝图使用。
