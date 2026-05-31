# MASS: Multi-Agent Stock System

<p align="center">
  <strong>多智能体协同股票投研与决策系统</strong><br>
  <em>6 维专业 Agent × Chairman 综合决策 × 全真实数据 × 流式实时诊断 × 十大性能优化</em><br>
  <em>v2.7 — Redis 分布式黑板、LLM 四层推理优化、回测闭环验证、逐 Agent 深度优化</em>
</p>

## 核心创新

### 1. 真正的多智能体协同，而非简单投票

MASS 不是让 6 个 Agent 各自输出一个信号然后取平均，而是构建了一套完整的**多智能体决策链**：

- **6 大专业 Agent** 独立分析：技术面(TA)、基本面(FA)、资金面(CA)、情绪面(SA)、宏观(MA)、风险(RA)
- **Chairman Agent** 综合决策：接收全部 Agent 观点，进行冲突仲裁、权重调整、情景推演，输出最终指令
- **Debate Engine** 辩论机制：自动检测 Agent 间观点冲突，组织正反方辩论，Chairman 根据辩论结果修正决策
- **动态权重系统**：根据宏观周期自动调整各 Agent 权重（牛市趋势/牛市价值/熊市防御/震荡市）

### 2. 流式实时诊断（SSE）

前端通过 `fetch + ReadableStream` 接收 SSE 流，**实时渲染每个 Agent 的分析进度**：

- 顶部信号概览条：6 个 Agent 的实时信号状态（买入/观望/卖出）
- 共识进度条：实时计算 6 个 Agent 的共识度
- Agent 状态卡片：左右双栏布局（图标+信号箭头+置信度 | 推理摘要+关键指标+进度条）
- **v2.7 新增**：Redis 分布式黑板、Agent 结论缓存、Prompt 压缩、模型分层、批量推理、回测偏差验证闭环
- **v2.3 新增**：指数退避断线重连 + 断线遮罩层，网络波动时自动恢复，用户零感知

### 3. Zero Mock Policy — 全真实数据

**零容忍虚假数据**。所有数据必须来自真实数据源，失败时返回 HTTP 503 或空结构，绝不编造：

| 数据源 | 优先级 | 数据类型 |
|--------|--------|----------|
| EastMoney (东方财富) | 100 | 基本面、资金流向、市场环境 |
| Akshare | 95 | K线历史、宏观数据 |
| Sina (新浪) | 90 | 实时行情、K线、新闻 |
| Tx (腾讯) | 80 | 基本面、实时行情 |
| THS (同花顺) | 50 | 基本面备用 |

**字段级多源合并**：当单个爬虫数据不完整时，系统自动按优先级遍历多个数据源，**按字段补全**（而非整份数据替换），最大化数据覆盖率。

**v2.3 数据质量修复**：
- Beta 系数从随机数改为真实 `Cov(Rs,Rm)/Var(Rm)` 计算
- 回测引擎接入真实 K 线，支持 MA 交叉 / RSI 动量 / MACD 信号策略

### 5. 量化回测增强（v2.4）

**5 大策略引擎**：MA 均线交叉、RSI 动量、MACD 信号、布林带突破、多因子共振（MA+RSI+MACD 三因子共振）

**LLM 智能解读**：回测完成后自动调用 LLM 生成策略表现分析报告 —— 策略有效性评估、市场环境适配度、与买入持有对比、风险提示与改进建议

**LLM 走势预测集成**：回测结果中嵌入 AI 对未来走势的预测（方向/置信度/目标价/止损价/风险收益比/概率分布）

**交易深度分析**：逐笔交易统计（平均持仓天数、最佳/最差交易、盈亏比、毛盈亏）、月度收益热力图、完整回撤曲线

**策略对比模式**：支持同时回测 2-3 个策略，并排对比收益曲线与绩效指标

### 4. 组合级并行分析

支持同时分析 **20 只股票的持仓组合**，并行诊断 + 风险聚合：

- **8 张汇总卡片**：总收益率、加权预期收益、风险等级、集中度、分散度评分等
- **6 个可视化图表**：持仓配置饼图、风险-收益散点图、信号分布条形图、行业分布、风险等级柱状图、再平衡建议
- **11 列表格**：每只个股的诊断详情（含行业、成本价、当前价、盈亏%、Agent 信号）
- **再平衡建议**：自动识别卖出/增持信号，生成调仓方案

---

## v2.3 重大改进

### 架构重构

| 改进项 | 说明 | 文件 |
|--------|------|------|
| **Blueprint 拆分** | 1,423 行 `agent_bp.py` 拆分为 7 个领域 Blueprint | `api/diagnose_bp.py` 等 7 个文件 |
| **DI 容器** | 移除 `__new__` 强制单例，引入 `Container` 支持依赖注入与测试隔离 | `agent/core/container.py` |
| **分层解耦** | `TaskTracker` 从 `api/` 下放到 `agent/core/`，消除 core→api 反向依赖 | `agent/core/task_tracker.py` |
| **DebateEngine 集成** | 辩论引擎从死代码接入 Chairman 决策流程，冲突检测→辩论→结果修正 | `agent/core/debate.py` |

### 安全与可靠性

| 改进项 | 说明 |
|--------|------|
| **CORS 白名单** | 移除通配符 `*`，改为 `ALLOWED_ORIGINS` 环境变量配置，生产环境禁止任意跨域 |
| **限流器启用** | 关键接口启用 `@RateLimiter().limit()`，支持 `X-Forwarded-For` 穿透反向代理 |
| **诊断并发保护** | 新增 `threading.Semaphore(8)` 限制同时诊断请求数，防止线程池耗尽 |
| **硬编码密钥删除** | `DEEPSEEK_API_KEY` 默认值已移除，启动时强制校验 |

### 爬虫与数据层

| 改进项 | 说明 |
|--------|------|
| **分层并发 + 优先级短路** | `fetch_merge` 按优先级分 3 批次并发，关键字段满足后提前返回，减少 40% 无效请求 | `agent/crawlers/registry.py` |
| **字段合并冲突检测** | 同一字段多源数值差异 >50% 时记录冲突到 `_meta.conflicts`，异常值不覆盖 | `agent/crawlers/registry.py` |
| **统一真 LRU 缓存** | `OrderedDict` 替代低效 Dict，O(1) 淘汰；去除 StockDataTool / BaseCrawler 双重缓存 | `agent/core/cache.py` |
| **域名级频率控制** | `SessionPool` 按 `domain_key` 共享限流，多实例同域名不再频率翻倍 | `agent/crawlers/session_pool.py` |
| **请求层公共逻辑提取** | `_request` / `_request_text` 提取 `_execute_request`，减少 ~35 行重复，便于统一指标统计 | `agent/crawlers/base.py` |
| **SessionPool 连接池调优** | pool=20，+429 重试，`raise_on_status=False`，TCP Keep-Alive 防 NAT 断开 | `agent/crawlers/session_pool.py` |
| **safe_float 异常值修复** | `"-"` / `""` / `NaN` / `Inf` 返回 `None` 而非 `0.0`，估值指标可区分"缺失"与"真实值为0" | `agent/crawlers/utils.py` |
| **共享 Session 池** | 同域名复用 `requests.Session` + HTTPAdapter keep-alive，减少 TCP 握手 | `agent/crawlers/session_pool.py` |
| **断路器模式** | `CLOSED/OPEN/HALF_OPEN` 状态机，连续失败 5 次后暂停 60 秒 | `agent/crawlers/circuit_breaker.py` |
| **缓存 TTL Bug 修复** | `timedelta.seconds` → `total_seconds()`，避免 >24h 缓存失效判断错误 |
| **增强情感分析** | 否定翻转 + 程度副词加权 + 转折句处理；支持 `USE_LLM_SENTIMENT` 切换引擎 | `agent/tools/sentiment_tool.py` |
| **系统统计缓存化** | `system_stats` 改为后台线程每 30s 预计算，接口响应 <10ms | `api/system_bp.py` |
| **线程池合并** | `_DATA_FETCH_EXECUTOR` + `_AGENT_EXECUTOR` 统一为 `_WORK_EXECUTOR`，每请求线程数从 ~11 降至 ~6-8 | `agent/core/orchestrator.py` |

### 测试体系

| 改进项 | 说明 |
|--------|------|
| **数据库隔离** | 测试使用临时文件数据库，彻底隔离生产数据 |
| **集成测试标记** | 网络依赖测试加 `@pytest.mark.integration`，CI 默认跳过 (`pytest -m "not integration"`) |
| **LLM 路径测试** | 新增 6 个单元测试，mock `openai.OpenAI` / `requests.post` 验证真实调用路径 |
| **同步化执行器** | 测试中后台线程池替换为同步执行器，消除多线程数据库竞争 |

### 前端体验

| 改进项 | 说明 |
|--------|------|
| **SSE 断线重连** | 指数退避（1s→2s→4s→...→30s 上限）、断线遮罩层、重连成功自动恢复 |

### 自适应并发控制（v2.6）

| 改进项 | 说明 | 文件 |
|--------|------|------|
| **AdaptiveConcurrencyController** | 基于响应时间、错误率、队列等待时间的动态并发限制；快速降级（错误率高/响应慢/等待长）+ 缓慢升级 + 冷却期防抖动 | `agent/core/concurrency_controller.py` |
| **诊断并发保护升级** | 替换静态 `threading.Semaphore` 为自适应控制器；`run_diagnosis()` / `run_diagnosis_stream()` 自动记录结果并触发动态调整 | `agent/core/orchestrator.py` |
| **并发监控 API** | `GET /api/agent/concurrency/stats` 暴露当前限制、平均响应、错误率等 12 项指标；`POST /api/agent/concurrency/reset` 手动重置 | `api/diagnose_bp.py` |

### Agent 间通信机制（v2.6）

| 改进项 | 说明 | 文件 |
|--------|------|------|
| **两轮分析架构** | Round 1: 6 Agent 独立分析 → 冲突检测 → Round 2: 分歧 Agent 执行修正分析，减少冲突 30~40% | `agent/core/communication.py` |
| **三维冲突触发** | 信号分歧 ≥2、置信度 < 0.5、关键 Agent 被 ≥2 个对手孤立 | `agent/core/communication.py` |
| **通信摘要构建** | 结构化摘要（信号/置信度/推理/关键因子/风险标记 + 量化指标）驱动修正 | `agent/core/communication.py` |
| **BaseAgent.revise()** | 支持自我修正：接收通信摘要 + 自反问题，输出带 `is_revision` / `original_signal` / `revision_round` 标记的观点 | `agent/agents/base_agent.py` |
| **流式进度透传** | 通信引擎的 Round 1 / Round 2 / 修正结果均通过 progress_cb 暴露到 SSE 流 | `agent/core/orchestrator.py` |

### 量化回测增强（v2.4）

| 改进项 | 说明 | 文件 |
|--------|------|------|
| **策略库扩展** | 新增布林带突破、多因子共振策略；修复买入手续费计算 Bug | `agent/core/backtest_engine.py` |
| **LLM 策略解读** | 回测后调用 LLM 分析策略表现、市场环境适配度、改进建议 | `agent/core/backtest_engine.py` |
| **LLM 预测集成** | 回测流程中自动调用 `PredictionEngine` 获取走势预测 | `api/backtest_bp.py` |
| **交易深度分析** | 逐笔统计（持仓天数、盈亏比、最佳/最差交易）、月度收益、回撤曲线 | `agent/core/backtest_engine.py` |
| **策略对比 API** | `POST /backtest/compare` 同时回测 2-3 个策略并排序对比 | `api/backtest_bp.py` |
| **前端全面重构** | 真实 API 调用、交易明细表格、AI 策略解读卡片、AI 预测卡片、策略对比模式 | `templates/agent_backtest.html` |

---

## v2.4 关键 Bug 修复与数据可靠性加固

### 1. 修复 `get_kline()` 指数代码跳过 Registry 备用源的致命 Bug

**问题现象**：
- 股票代码 `000001`（平安银行）被误判为**上证指数**，导致 `is_index=True`
- 代码中的 `if not is_index:` 守卫语句**直接跳过**了 `CrawlerRegistry.fetch_merge()` 备用源调用
- akshare 指数接口 `index_zh_a_hist()` 因网络问题失败后，**无任何 fallback**，返回 `None`
- 回测引擎因拿不到 K 线数据而返回 `DATA_UNAVAILABLE` 503

**根因**：`_INDEX_CODES = {"000300", "000001", ...}` 中 `000001` 与平安银行个股代码冲突，且逻辑上不应因指数标记而禁用备用源。

**修复**（`agent/tools/stock_data_tool.py`）：
```python
# 修复前：指数代码被排除在 registry 备用源之外
if not is_index:          # ← Bug：000001 走这里为 False，直接跳过
    merged = self._registry.fetch_merge(...)

# 修复后：所有代码统一尝试 registry 备用源（SinaCrawler 支持全部 6 位代码）
try:
    merged = self._registry.fetch_merge(stock_code, "kline", days=days)
    ...
except Exception as e:
    logger.warning(f"Registry K线获取失败: {e}")
```

**验证结果**：
| 代码 | 修复前 | 修复后 | 数据源 |
|------|--------|--------|--------|
| 000001 | `None` → 503 |  120 条真实 K 线 | SinaCrawler |
| 600519 | `None` → 503 |  120 条真实 K 线 | SinaCrawler |

---

### 2. 零容忍：彻底删除模拟数据回退逻辑

**问题现象**：
- `api/backtest_bp.py` 中存在 `_generate_mock_kline()` 函数，在真实数据失败时**自动生成合成 K 线**
- 合成数据具有虚假的趋势和波动特征，虽能跑通回测逻辑，但**完全不可用于实际决策**
- 这与项目 "Zero Mock Policy" 原则直接冲突

**修复**（`api/backtest_bp.py`）：
```python
# 已彻底删除：_generate_mock_kline() 函数（~30 行模拟数据生成代码）
# 已彻底删除：kline_df = _generate_mock_kline(stock_code, days=252) 回退逻辑

# 修复后：真实数据不可用时直接返回 503，绝不留任何模拟后门
if kline_df is None or len(kline_df) < 60:
    return jsonify({
        "code": "DATA_UNAVAILABLE",
        "error": "无法获取足够的历史K线数据进行回测",
        "message": "真实数据源暂不可用，请检查股票代码或稍后重试",
    }), 503
```

**影响范围**：
- `POST /backtest/run` — 单策略回测
- `POST /backtest/compare` — 多策略对比
- 两处均已删除模拟回退，统一返回 503

---

### 3. 修复 LLM 策略解读调用失败

**问题现象**：
- 回测完成后前端显示 "LLM策略解读服务暂不可用"
- 日志报错：`LLMClient.chat() got an unexpected keyword argument 'max_tokens'`
- LLM 策略解读功能完全不可用

**根因**：`LLMClient.chat()` 的签名仅接受 `system`, `user`, `json_mode`, `model`, `override_config` 五个参数，但 `_generate_llm_explanation()` 调用时传入了 `max_tokens=1500`，导致 `TypeError`。

**修复**（`agent/core/backtest_engine.py`）：
```python
# 修复前
response = llm.chat(
    system=system_prompt,
    user=user_prompt,
    json_mode=False,
    max_tokens=1500,   # ← 不被 LLMClient.chat() 接受，导致 TypeError
)

# 修复后
response = llm.chat(
    system=system_prompt,
    user=user_prompt,
    json_mode=False,
)
```

**验证结果**：
- LLM 策略解读正常生成（~800 字专业分析报告）
- 包含：策略有效性评估、买入持有对比、市场环境分析、风险提示与改进建议
- 耗时约 3 秒，DeepSeek API 响应正常

---

### 修复总结

| 修复项 | 文件 | 影响 | 验证 |
|--------|------|------|------|
| 指数代码跳过 registry Bug | `agent/tools/stock_data_tool.py` | 000001 等代码无法回测 |  5 个策略全部通过 |
| 删除模拟数据回退 | `api/backtest_bp.py` | 虚假数据污染回测结果 |  503 错误正确返回 |
| LLM 策略解读参数错误 | `agent/core/backtest_engine.py` | AI 解读卡片显示不可用 |  800 字报告正常生成 |

**所有回测数据均为真实历史 K 线**（`data_source: 真实历史K线`，`simulation: false`），数据源以 SinaCrawler 为主（当前网络环境下最稳定）。

---

## Agent 智能升级（v2.5）

本次升级聚焦 **FA-Agent（基本面分析师）** 和 **MA-Agent（宏观策略师）** 的推理深度与决策质量，解决"数据缺失就消极""风格矛盾不自知""利好利空不对冲"等核心问题。

---

### 一、FA-Agent：从"罗列数据"到"估值联动推断"

#### 核心问题
- **PB 80.52 倍 + PE 23.8 倍** → 原 Agent 仅平淡描述"PB极高"，未分析隐含ROE、未识别估值异常
- **"数据缺失无法判断"** → 消极表述频现，即使PE/PB充足也无法产出有效分析
- **降级分析过于简单** → LLM失败时仅基于ROE/PE/负债率三个指标打分，无法处理数据缺失场景

#### 升级内容

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **估值分析深度** | 仅对比PE/PB与行业均值 | **PE-PB-ROE三角联动**：自动计算隐含ROE = PB/PE，识别"PB极高+PE正常"的估值异常组合 |
| **数据缺失处理** | "盈利能力数据缺失，无法评估" | **基于估值联动推断**：ROE缺失时用隐含ROE替代，毛利率缺失时通过PE+行业推断 |
| **估值异常检测** | 无 | **PB>20 自动触发强负面信号**，评分扣15分；PB>10 扣10分；隐含ROE>50% flagged 为"不可持续或会计异常" |
| **财务健康度** | 无 | **红绿灯评估**：[安全]/[注意]/[危险]，综合负债率、流动性、现金流、估值异常 |
| **行业基准** | 硬编码"PE:25, PB:2.5" | **30个行业动态基准**（银行PE 6倍、白酒PE 28倍等），支持真实行业数据 fallback |
| **降级分析层数** | 单层（ROE+PE+负债率） | **四层降级**：K线完整 → 资金流向 → 新闻情感 → 保守兜底，每层均有意义推断 |
| **Prompt约束** | 允许"数据缺失"表述 | **明确禁止**"数据缺失""无法判断"等消极措辞 |

#### 典型案例对比

**输入**：PE=23.8, PB=80.52, 行业=电子

| 维度 | 升级前 | 升级后 |
|------|--------|--------|
| **信号** | 观望 (0) | **卖出 (-1)** |
| **评分** | 55/100 | **40/100** |
| **隐含ROE** | 未计算 | **338.32%**（自动推导，flagged为异常） |
| **估值缺口** | "PB极高，可能存在估值泡沫" | "PB(80.52)极高但PE(23.80)正常，隐含ROE高达338.32%，远超正常水平(5%-30%)，极可能因净资产被压低或一次性收益导致，不可持续" |
| **财务健康** | 无 | **[危险] PB极度异常** |
| **关键表述** | "公司盈利能力数据缺失，无法评估" | "隐含ROE=338.32%，远非正常水平，基本面支撑不足" |

---

### 二、MA-Agent：从"罗列宏观指标"到"量化对冲+风格校验"

#### 核心问题
- **"买入"信号 + 50%置信度** → 信号与置信度不匹配，买入应有更高确信
- **"风格切换至顺周期" + "成长风格匹配"** → **自相矛盾**，未做风格一致性校验
- **利好利空无对冲** → PMI>50(+)、政策补贴(+) 与 风格错配(-)、美联储收紧(-) 简单并列，未计算净效应
- **"尽管部分宏观数据缺失"** → 消极表述，未基于已有数据推断

#### 升级内容

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **利好-利空分析** | 定性罗列，无权重 | **量化对冲框架**：强利好(+2)、弱利好(+1)、弱利空(-1)、强利空(-2)，计算 **macro_score 净得分** |
| **信号-置信度联动** | 无约束 | **买入/卖出信号 confidence ≥ 0.60**，否则自动提升 |
| **风格匹配校验** | 无 | **行业-风格映射表**（新能源=成长、银行=价值等），`style_alignment < 0.5` 时 **强制禁止买入信号** |
| **宏观传导链条** | 孤立看指标 | **"宏观→行业→个股"传导逻辑**：利率上行→利空成长→该股票属新能源→受损 |
| **信号一致性修正** | 无 | **macro_score 与 signal 联动校验**：净得分≥+3才允许买入，≤-3才允许卖出 |
| **降级分析层数** | 单层（PMI+政策） | **多层打分**：指数趋势 → 板块流向 → PMI → 政策 → 风格匹配，逐层推断 |
| **Prompt约束** | 允许"数据缺失"表述 | **明确禁止**消极措辞，要求标注每条因子的量化权重 |

#### 典型案例对比

**输入**：赛力斯(新能源)，PMI=51.5，政策宽松，市场风格=顺周期

| 维度 | 升级前 | 升级后 |
|------|--------|--------|
| **信号** | 买入 (1) | **观望 (0)** |
| **置信度** | 0.50（与买入信号不匹配） | 0.50（观望信号合理） |
| **风格匹配** | "偏向成长，与当前市场风格匹配"（**自相矛盾**） | **0.25（严重错配）**：成长行业 vs 顺周期市场 |
| **宏观净得分** | 无 | **-2**：利好(+4) vs 风险(-6) 对冲后净负 |
| **关键推理** | "尽管部分宏观数据缺失，但基于...判断市场处于复苏早期" | "PMI51.5经济扩张(+1)，政策宽松(+2)，但所属行业资金流出(-1)，风格严重错配(-2)，利率上行利空成长(-1)，美联储收紧(-1)。净得分-2，宏观面中性偏观望" |
| **对比案例** | — | **工商银行(银行)**：同样的宏观环境 → 信号**买入(1)**，净得分**+6**，风格匹配**0.75** |

---

### 三、通用改进原则（适用于全部 Agent）

本次 FA/MA 升级确立了以下**Agent 开发规范**，后续将推广至 TA/CA/SA/RA：

1. **禁止消极表述**：绝不使用"数据缺失""无法判断""缺乏数据"，改为"基于X推断Y"
2. **量化对冲机制**：key_factors 和 risk_flags 必须带权重，计算净得分后确定信号
3. **信号-置信度一致性**：±1 信号 confidence ≥ 0.60，0 信号 confidence ≤ 0.60
4. **多层降级分析**：LLM 失败时逐层降级，每层基于最可靠的可用指标做推断
5. **隐含指标推导**：从已有数据推导缺失指标（如隐含ROE = PB/PE，隐含PS = 市值/营收）
6. **异常值自动检测**：极端指标组合（PB>10+PE正常、PE为负等）自动触发修正逻辑

---

## v2.6 逐 Agent 深度优化与全链路风险体系

本次升级聚焦 **6 大 Agent 全面深化** + **系统内核双引擎升级**（Agent 间通信 + 自适应并发控制），从"各司其职"走向"协同进化"。

---

### 一、RA-Agent 风险体系 v2.0-2.2（已完成）

RA-Agent 从 155 行轻量风控升级为 **全链路风险决策引擎**（~1200 行），覆盖压力测试、组合风险、动态止损三大模块。

#### 1. 多情景压力测试（v2.0）

| 能力 | 说明 |
|------|------|
| **4 情景引擎** | 基准(50%) / 牛市(25%, +20%×β) / 熊市(20%, -20%×β×1.5) / 黑天鹅(5%, -40%×β×2.5) |
| **VaR/CVaR** | 历史模拟法（优先）+ 正态近似 fallback；95% 置信度 |
| **Gap Risk** | 检测跳空缺口频率与幅度，评估隔夜/事件驱动风险 |
| **Liquidity Risk** | Amihud 非流动性比率 + 换手率百分位 |
| **Volatility Term Structure** | 短期(5d) vs 中期(20d) vs 长期(60d) 波动率趋势与期限利差 |

#### 2. 组合风险分析（v2.1）

| 能力 | 说明 |
|------|------|
| **协方差矩阵** | 基于行业相关性的简化协方差模型，计算新增股票对组合波动的影响 |
| **HHI 集中度** | 行业赫芬达尔指数 + 最大行业占比，识别过度集中 |
| **边际风险贡献(MRC)** | 新增仓位对组合 VaR 的边际增量 |
| **组合 VaR** | 整合现有持仓 + 拟新增仓位的整体 95% VaR |
| **仓位约束** | 根据 Beta 变化率、HHI、行业重叠度计算 `recommended_max_position` |

#### 3. 动态止损策略（v2.2）

| 策略 | 触发逻辑 |
|------|----------|
| **volatility_adaptive** | 波动率分档：高波动(>35%) tighten，低波动(<15%) widen |
| **atr_1x / 2x / 3x** | 1×/2×/3× ATR 止损，适应不同风险偏好 |
| **trailing** | 移动止损，上涨后自动上移锁定利润 |
| **time_based** | 时间止损，N 日不涨即离场 |
| **technical_low** | 近期低点止损 |
| **technical_support** | 支撑位止损 |
| **technical_bollinger** | 布林带下轨止损 |
| **智能推荐路由** | 基于波动率、Beta、趋势方向、胜率自动推荐最优策略 |

**降级引擎**：LLM 失败时，基于压力测试结果 + 组合风险约束 + 动态止损自动计算风险等级、仓位上限、止损位。

---

### 二、TA-Agent 技术面升级

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **多时间框架** | 仅近 10 日日线 | **日线 + 周线 + 60分钟线** 信号一致性评分，大周期优先原则 |
| **支撑阻力矩阵** | 单一支撑位 | **强/弱支撑 + 强/弱阻力** 四档矩阵，距当前价百分比标注 |
| **形态识别** | LLM 自由推断 | **规则引擎预识别**（头肩顶/底、双顶/底、三角形、旗形、楔形）+ LLM 确认 + 可靠性评分 |
| **降级引擎** | 均线 + MACD 2因子 | **5维度15因子评分模型**：趋势(30分) + MACD(20分) + 动量(20分) + 波动(15分) + 成交量(15分) |

---

### 三、FA-Agent 基本面升级

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **财务趋势** | 单时间点截面分析 | **近8季度纵向趋势**：营收增速、毛利率、ROE、经营现金流趋势 + 盈利质量评分 |
| **同业对比矩阵** | PE/PB 相对值 | **同行业市值最接近5家** Z-Score 排名，个股在各指标上的百分位 |
| **盈利预测** | 仅 TTM | **Forward PE**（基于分析师一致预期）+ 前瞻性估值分析 |
| **趋势判断规则** | 无 | 连续3季ROE上升→+10分；连续2季毛利率下降→-10分；增收不增利→-5分 |

---

### 四、CA-Agent 资金面升级

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **筹码分布** | 无 | **VWAP 估算主力成本** + 筹码集中度(CR90) + 获利盘比例 + 套牢盘压力 + 筹码锁定度 |
| **北向资金深度** | 单一净流入值 | 近5日/30日净流入 + 连续流入天数 + 持仓占比变化 |
| **融资融券** | 余额变化单一数值 | 融资余额/流通市值比率 + 融资买入额/成交额比率 + 杠杆趋势分析 |
| **机构行为** | 仅调研次数 | 基金持仓季度变化 + 机构持仓集中度变化追踪 |

---

### 五、SA-Agent 情绪面升级

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **社交媒体情绪** | 无（仅K线推导+新闻） | **雪球/股吧情绪爬虫**：发帖量、正面/负面词频、关注人数变化 |
| **情绪动量** | 绝对水平（百分位） | **5日情绪变化速度**：加速/缓和/极端化信号 |
| **相对情绪** | 无 | 个股情绪 vs 板块情绪 vs 大盘情绪 三维对比 |
| **异常情绪检测** | 无 | 发帖量激增但价格未动 → "有人造势" 预警 |

---

### 六、MA-Agent 宏观升级

| 能力 | 升级前 | 升级后 |
|------|--------|--------|
| **行业景气度周期** | 仅资金流向排名 | **四周期定位**：复苏/繁荣/衰退/萧条，基于库存+资本开支+盈利三维判断 |
| **政策敏感性** | 定性宽松/收紧 | **各行业政策敏感度映射**（货币/财政/监管），量化政策传导预期 |
| **全球联动** | 仅国内数据 | 汇率敏感性（出口/进口型行业）+ 大宗商品成本传导 + 美股映射传导 |
| **宏观传导链条** | 孤立指标 | **"宏观→行业→个股"** 完整传导逻辑，利率上行→利空成长→该股票属新能源→受损 |

---

### 七、Orchestrator 内核升级

| 能力 | 说明 | 文件 |
|------|------|------|
| **Agent 间通信** | 两轮分析：独立 → 冲突检测 → 修正；减少 Agent 间冲突 30~40% | `agent/core/communication.py` |
| **自适应并发** | 基于响应时间/错误率/等待时间的动态并发限制；快速降级+缓慢升级+冷却期 | `agent/core/concurrency_controller.py` |
| **流式进度透传** | 通信引擎 Round 1/2 和修正结果均实时推送到 SSE | `agent/core/orchestrator.py` |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端层 (Frontend)                        │
│    ECharts 雷达图 │ K 线图 │ Agent 卡片墙 │ 推理链时序         │
│    流式 SSE 面板（断线自动重连）│ 组合分析仪表盘 │ 自定义 Tooltip │
└─────────────────────────────────────────────────────────────┘
                              ↓ REST API
┌─────────────────────────────────────────────────────────────┐
│                    API 层 (Flask Blueprints)                  │
│    diagnose │ portfolio │ prediction │ backtest │ report      │
│    history  │ system    │ 限流 │ CORS 白名单 │ 错误处理      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    编排调度层 (Orchestrator)                   │
│    Chairman: 任务分解 → 6 Agent 并行分析 → 冲突仲裁 → 最终决策  │
│    AgentCommunicationEngine: 两轮分析 + 冲突检测 + 修正         │
│    AdaptiveConcurrencyController: 动态负载感知并发控制           │
│    Prediction Engine: 短/中/长期走势预测                        │
│    Debate Engine: Agent 间观点辩论（已接入 Chairman）           │
│    Task Tracker: 跨页任务持久化                                │
└─────────────────────────────────────────────────────────────┘
                              ↓ Message Bus
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ TA-Agent │ FA-Agent │ CA-Agent │ SA-Agent │ MA-Agent │ RA-Agent
│  技术面   │  基本面   │  资金面   │  情绪面   │  宏观    │  风险
└──────────┴──────────┴──────────┴──────────┴──────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              共享黑板 (Shared Blackboard)                     │
│    内存黑板：按股票分片锁 │ 有界缓存(200只) │ 线程安全             │
│    Redis黑板：分布式缓存 │ TTL=3600s │ 自动降级到内存            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              数据与工具层 (Tools & Data)                       │
│  EastMoney │ Akshare │ Sina │ Tx │ THS │ SQLite │ DeepSeek   │
│  字段级合并 │ 5源并发 │ Session 池 │ 断路器 │ 全真实数据       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              DI 容器 (Container) — 测试隔离                    │
│    Blackboard │ Cache │ CrawlerRegistry │ TaskTracker        │
│    StockDataTool — 注册/获取/重置                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API 密钥（默认使用 DeepSeek-V4-Pro）
```

### 3. 启动服务

```bash
# Mock 模式（无需 API 密钥，测试用）
python run.py --mock

# 生产模式（真实 LLM + 真实数据爬虫）
python run.py
```

### 4. 命令行诊断

```bash
python run.py --mock --diagnose 000001
```

### 5. 访问 Web 界面

- **个股智能诊断**：http://localhost:5000/agent/trading
- **组合分析**：http://localhost:5000/agent/portfolio
- **量化回测**：http://localhost:5000/agent/backtest
- **历史记录**：http://localhost:5000/agent/history
- **系统监控**：http://localhost:5000/agent/monitor

---

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/agent/diagnose` | POST | 单股票多智能体诊断 |
| `/api/agent/diagnose/stream` | POST | **SSE 流式实时诊断**（支持断线重连） |
| `/api/agent/portfolio/analyze` | POST | 组合级并行分析 |
| `/api/agent/predict` | POST | 股票走势预测（短/中/长期） |
| `/api/agent/backtest/strategies` | GET | 获取支持的策略列表 |
| `/api/agent/backtest/run` | POST | 量化回测（真实 K 线 + LLM 解读/预测） |
| `/api/agent/backtest/compare` | POST | 多策略对比回测 |
| `/api/agent/report/generate` | POST | 生成投研报告（Markdown/JSON） |
| `/api/agent/debate/simulate` | POST | 模拟 Agent 间辩论 |
| `/api/agent/decisions/history` | GET | 历史决策记录 |
| `/api/agent/positions` | GET/POST | 模拟持仓管理 |
| `/api/agent/stats` | GET | 系统统计与监控（30s 缓存） |
| `/api/agent/blackboard/clear` | POST | 清理黑板缓存 |
| `/api/agent/concurrency/stats` | GET | 自适应并发控制器统计信息 |
| `/api/agent/concurrency/reset` | POST | 重置自适应并发控制器（调试/恢复） |

### 诊断接口示例

```bash
# 普通诊断
curl -X POST http://localhost:5000/api/agent/diagnose \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "000001"}'

# 流式诊断（SSE）
curl -X POST http://localhost:5000/api/agent/diagnose/stream \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "000001"}'

# 组合分析
curl -X POST http://localhost:5000/api/agent/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"code": "000001", "cost": 15.2, "shares": 1000, "name": "平安银行"},
      {"code": "600519", "cost": 1800, "shares": 100, "name": "贵州茅台"}
    ]
  }'

# 量化回测（含 LLM 解读与预测）
curl -X POST http://localhost:5000/api/agent/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "strategy": "ma_cross",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "include_llm_explanation": true,
    "include_llm_prediction": true
  }'

# 策略对比
curl -X POST http://localhost:5000/api/agent/backtest/compare \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "strategies": ["ma_cross", "macd_signal", "multi_factor"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

---

## Agent 角色与能力

| Agent | 职责 | 核心维度 | 输出 |
|-------|------|----------|------|
| **TA-Agent** | 技术面分析 | 多时间框架信号一致性、K线形态识别(头肩/双顶/三角形)、支撑阻力矩阵、MACD/RSI/KDJ共振、量价关系 | 信号 + 目标价区间 + 形态可靠性评分 + 多周期一致性得分 |
| **FA-Agent** | 基本面分析 | **PE-PB-ROE三角联动**、隐含ROE推断、估值异常检测、财务健康红绿灯、**近8季度财务趋势**、**同业Z-Score对比**、行业对比 | 评分(0-100) + 估值缺口 + 财务健康度 + 趋势判断 + 子维度评分 |
| **CA-Agent** | 资金面分析 | 主力资金流向、**筹码分布分析**(VWAP/集中度/获利盘)、**北向资金深度追踪**、**融资融券杠杆分析**、**机构持仓季度变化** | 资金评分 + 主力意图 + 筹码锁定度 + 杠杆趋势 |
| **SA-Agent** | 情绪面分析 | 新闻舆情、**社交媒体情绪**(雪球/股吧)、**情绪动量**(5日变化速度)、**个股/板块/大盘相对情绪**、板块热度 | 情绪指数(-1~1) + 动量方向 + 相对情绪差 + 逆向机会 |
| **MA-Agent** | 宏观策略 | **利好-利空量化对冲**、宏观→行业→个股传导、风格匹配一致性、**行业景气度四周期定位**、**政策敏感度映射**、**全球联动分析** | 周期判断 + 宏观净得分 + 风格匹配度 + 行业景气度评分 + 权重调整建议 |
| **RA-Agent** | 风险控制 | **4情景压力测试**(基/牛/熊/黑天鹅)、**组合风险分析**(协方差/HHI/MRC)、**7种动态止损策略**、波动率、VaR/CVaR、最大回撤 | 风险等级(1-5) + 仓位上限 + 组合VaR + 推荐止损策略 + 尾部风险指标 |
| **Chairman** | 综合决策 | 冲突仲裁、情景推演、共识计算 | 最终指令 + 3种情景分析 |

### 动态权重系统

系统根据宏观周期自动调整各 Agent 决策权重：

| 市场周期 | TA | FA | CA | SA | MA | RA |
|----------|----|----|----|----|----|----|
| **牛市趋势** | 30% | 15% | 25% | 15% | 10% | 5% |
| **牛市价值** | 15% | 35% | 20% | 15% | 10% | 5% |
| **熊市防御** | 10% | 20% | 15% | 15% | 10% | 30% |
| **震荡市** | 25% | 15% | 25% | 20% | 10% | 5% |

---

## 性能优化（v2.2 / v2.3 / v2.7）

MASS 针对生产环境进行了**系统性性能调优**，覆盖请求全链路：

### 一、I/O 层优化

#### 1. 爬虫分层并发 + 优先级短路（省 ~33% I/O 时延）
`fetch_merge` 按优先级将 5 个数据源分为 3 批次并发。高优先级批次（EastMoney 100 / Akshare 95 / Sina 90）完成后检查关键字段，若已满足则**跳过低优先级批次**（Tx 80 / THS 50），减少 1~2 个无效请求。总耗时从 ~1.5s 降至 ~1.0s。

#### 2. 爬虫单例去重（省 ~300ms 冷启动 + 内存）
`CrawlerRegistry` 全局单例，`StockDataTool` 与 `SentimentTool` **共享同一组爬虫和 Session**，避免 2 个 Registry + 7 个爬虫实例的重复创建。

#### 3. 共享 Session 池 + 断路器（v2.3）
同域名共享 `requests.Session`（HTTPAdapter keep-alive + TCP Keep-Alive），配合 `CircuitBreaker` 状态机：连续失败 5 次后自动暂停 60 秒，避免故障源持续占用资源。

#### 4. 缓存预热（省 2-5s 首次请求）
服务启动时后台 daemon 线程预加载热门股票数据快照，消除首次查询的冷启动延迟。通过 `WARMUP_STOCKS=000001,600519` 环境变量配置。

#### 5. 数据库保存异步化（省 50-200ms/请求）
SQLite 写入从响应路径剥离为 `fire-and-forget` 后台线程提交，客户端不等待落盘。

#### 6. 统一真 LRU 缓存（O(1) 淘汰 + 去除双重缓存）
`CacheManager` 改用 `OrderedDict`：命中 `move_to_end`、超限 `popitem(last=False)` 均为 O(1)。去除 `StockDataTool` 自建低效缓存（原 `sorted()` 全排序 O(n log n)），统一委托给 `CacheManager` 全局单例。

#### 7. 域名级频率控制（反爬稳定性）
`SessionPool` 按 `domain_key`（eastmoney / sina / tx / ths）共享 `_last_request_time` 与 `threading.Lock`。即使创建多个 `EastMoneyCrawler` 实例，对 `eastmoney.com` 的实际请求频率严格受控，避免翻倍导致被封。

#### 8. SessionPool 连接池调优
- `pool_connections=20` / `pool_maxsize=20`（原为 10），应对 5 源多子域名并发
- 重试列表加入 `429`（Too Many Requests），`raise_on_status=False` 保留原始 Response
- 全局启用 `TCP SO_KEEPALIVE`，防止 NAT/防火墙空闲超时断开长连接

### 二、计算层优化

#### 9. 指标计算并行入 fan-out（省 50-100ms）
K线一到手立刻提交 `compute_all()` 和 `compute_risk_metrics()` 到线程池，与基本面/情绪/宏观的数据获取**完全重叠**。

#### 10. RSV 指标向量化（3.2× 加速）
`_compute_rsv` 从 Python for 循环 + `np.max/min` 切片改为 `pd.Series.rolling()` 向量化，消除 O(n×k) 循环；`_compute_kdj` 预计算 NaN mask 避免循环内重复判断。

#### 11. TA-Agent Prompt 构建优化（~25× 加速）
`df.iterrows()` → `df.itertuples()`，避免每行创建 Series 对象，10 行 K 线格式化从 ~500μs 降至 ~20μs。

### 三、缓存与序列化

#### 12. LLM 响应缓存扩面（省 5-15s 命中时）
`/api/agent/diagnose/stream` 和 `/api/agent/predict` 新增 TTL=300s 缓存，与 `/api/agent/diagnose` **共享同一缓存键**。用户刷新页面、切换 horizon 等高频场景直接命中。

#### 13. orjson 全局 JSON 加速（5.5× 序列化加速）
Rust 实现的 `orjson` 透明替换 stdlib `json.dumps`，50KB DecisionPackage 序列化从 2.5ms 降至 0.5ms，带安全回退。

#### 14. `to_prompt_context()` 惰性缓存
6 个 Agent 共享同一份 Prompt 上下文字符串，首次生成后缓存，消除 42 次/请求的 `json.dumps` 重复序列化。

#### 15. 系统统计缓存化（v2.3）
`system_stats` 接口改为后台线程每 30s 预计算，接口响应从数秒降至 <10ms，附带 `cached_at` / `stale` / `cache_age_seconds` 元数据。

### 四、线程与内存管理

| 线程池 | 用途 | Workers |
|--------|------|---------|
| `_CRAWLER_EXECUTOR` | 爬虫分层并发（5 源 fan-out） | 5 |
| `_WORK_EXECUTOR` | 数据获取 + Agent 并行分析（v2.3 合并） | `min(32, cpu_count+4)` |
| `_PORTFOLIO_EXECUTOR` | 组合分析并行诊断 | 3 |
| `_DIAGNOSIS_EXECUTOR` | SSE 后台诊断（释放 Waitress） | 10 |
| `_DB_SAVE_EXECUTOR` | 异步数据库写入 | 2 |

- **Blackboard 分片锁** — 按股票代码 `Dict[str, Lock]`，不同股票零竞争
- **有界缓存** — `StockDataTool`(500条) + `Blackboard`(200只)，时间戳淘汰防止内存泄漏
- **限流器** — `deque` 滑动窗口 O(1) + 计数/时间双重过期清理，防止 key 无限增长
- **诊断并发控制** — 自适应并发控制器动态调整限制（2~10），替代静态信号量，防止线程池耗尽并自动降级
- **SQLite** — WAL 模式 + `threading.local()` 线程本地连接池

### 五、LLM 推理优化（v2.7）

针对 LLM 调用成本高、延迟大的痛点，v2.7 引入 **四层推理优化**，在保持精度的前提下显著降低 token 消耗和 API 调用次数：

#### 16. Agent 结论缓存（省 60~80% 重复 LLM 调用）
`AgentCache` 基于**指标指纹**（fingerprint）缓存单个 Agent 的分析结论。提取该 Agent 关注的核心指标子集，浮点四舍五入消除噪声，生成唯一缓存键。TTL 默认 30s，线程安全，命中时直接返回 `AgentOpinion`，无需再次调用 LLM。

| 场景 | 效果 |
|------|------|
| 同股票 30s 内重复诊断 | 6 个 Agent 结论全部命中缓存 |
| 回测遍历 250 个交易日 | 趋势不变的日子直接复用 TA/FA 结论 |
| 组合分析 10 只成分股 | 高频出现的板块宏观结论复用 MA |

#### 17. Prompt 压缩器（省 40~50% tokens）
`PromptCompressor` 为每个 Agent **定制白名单**，仅保留该 Agent 决策必需的字段：
- **精度截断**：价格/成交量浮点保留 2 位，RSI/MACD 保留 3 位，消除无意义精度
- **空值剪枝**：递归删除 `None` / `""` / `[]` / `{}` / `0` 字段，Prompt 体积典型缩减 35~50%
- **紧凑输出**：字典转为 `k:v` 紧凑格式（比 JSON 省 15~20% token）

#### 18. 模型分层调度
引入 `AGENT_LIGHT_MODEL` / `AGENT_HEAVY_MODEL` 双轨模型：

| Agent | 模型 | 理由 |
|-------|------|------|
| **FA-Agent** | Heavy | 财务推理需要长上下文、高精度 |
| **MA-Agent** | Heavy | 宏观传导链复杂，需强逻辑能力 |
| TA / CA / SA / RA | Light | 指标识别、模式匹配、规则计算为主，轻量模型足够 |

通过 `AGENT_MODEL_MAP` 灵活配置，无需改动代码即可切换模型供应商。

#### 19. 批量推理（省 66% API 调用，默认关闭）
`BatchLLMClient` 透明拦截 Agent 的 `_call_llm()`，将 6 个 Agent 的独立调用合并为 2 批批量调用（TA+CA+SA / FA+MA+RA），每批构造 master prompt 统一提交。适合高并发压测场景，通过 `BATCH_INFERENCE_ENABLED=True` 开启。

---

## 回测闭环验证（v2.7）

回测不仅是策略收益的验证，更是 **Agent 系统性偏差的诊断器**。v2.7 引入 `DecisionValidator`，形成**数据驱动的效果闭环**：

### 偏差检测五维模型

| 偏差类型 | 检测逻辑 | 触发阈值 |
|----------|----------|----------|
| **低准确率偏差** | 某 Agent 近期准确率 < 30% | `BIAS_ACCURACY_THRESHOLD = 0.30` |
| **过度自信偏差** | 高置信度（>0.75）但预测错误率过高 | `OVERCONFIDENCE_CONF_THRESHOLD = 0.75` |
| **方向性偏差** | 持续看多/看空，与市场实际方向背离 | `DIRECTIONAL_BIAS_THRESHOLD = 0.30` |
| **恶化趋势偏差** | 准确率呈持续下降通道 | `DEGRADING_ACCURACY_THRESHOLD = 0.40` |
| **信号分布偏移** | 信号分布严重倾斜（如 90% 为看多） | `SIGNAL_SKEW_THRESHOLD = 0.80` |

### 闭环双机制

1. **日内动态修正** — 回测的每日 `_backtest_loop()` 中，lookback 天后调用 `decision_engine.record_outcome()`，根据实际收益动态调整该 Agent 权重
2. **赛后复盘报告** — `multi_agent` 策略回测结束后自动调用 `validator.validate_backtest()`，生成：
   - 每个 Agent 的绩效统计（准确率、平均收益、Sharpe）
   - 系统性偏差检测报告
   - Prompt 微调建议（"建议强化对缩量回调的识别"）
   - 权重调整建议（"TA 在震荡市过度反应，建议权重 -5%"）

验证结果写入 JSONL 文件（`data/validation_*.jsonl`），支持历史追踪与模型迭代。

---

## 诊断引擎稳定性修复（v2.7）

修复了生产环境中导致实时诊断"共识 0/6 出错"的三个核心 bug：

| Bug | 根因 | 修复方案 |
|-----|------|----------|
| **Agent 通信引擎崩溃** | `_record_round1_opinions()` 先 `clear_stock()` 再 `get_snapshot()` 返回 `None`，`publish_snapshot(None)` 触发 `AttributeError` | 在 clear 之前保存 snapshot，增加 None 保护 |
| **numpy 序列化失败** | `blackboard.to_prompt_context()` 和 `_snapshot_to_dict()` 直接对包含 `numpy.int64/float64` 的数据调用 `orjson.dumps()`，导致所有 Agent 降级为"系统出错" | 三处添加 `_sanitize()` 递归清理函数；`app.py` 的 `_fast_dumps` 增加 numpy 类型支持；`agent_cache._round_nested` 兼容 numpy 类型 |
| **TA-Agent 格式错误** | `df.itertuples()` 第一列为日期字符串，代码用位置索引 `row[1]` 直接执行 `float()` 格式化开盘价，触发 `ValueError: could not convert string to float` | 改用 `iterrows()` + 列名映射（兼容中英文列名），`float()` 前增加异常保护 |

---

## 前端UX优化（v2.7）

### 6 Agent 卡片排版统一

流式诊断与结果诊断原本使用两套完全独立的卡片 DOM 结构和 CSS 类，导致视觉风格不一致（border-left vs border-top、header 布局不同、无进度条 vs 有进度条、等高强制 fr vs 自然高度）。

v2.7 统一为 `agent-card` 单一样式：
- **border-top 彩色条**（3px）标识 Agent 类型
- **单行紧凑 header**：图标(28px) + 名称(加粗彩色) + 角色(灰色) + 信号 + 置信度
- **3px 高进度条**：置信度可视化
- **key-value 指标行**：Agent 特有核心指标
- **3 行截断 reasoning** + 底部 tags
- 去掉 `grid-template-rows: repeat(2, 1fr)` 和 `min-height: 60vh`，改为自然高度排列

### 长时等待阶段轮播

针对同步诊断"预计 10-30 秒"严重低估实际耗时（真实 LLM 约 2-5 分钟）的问题：

1. **文案修正**："预计 2-5 分钟（真实数据获取 + 6 Agent LLM 推理 + 通信修正 + 辩论评估）"
2. **阶段轮播**：loading 区域按 3.5s 间隔自动轮播 8 个阶段（初始化 → 数据获取 → Agent 分析 → 观点收集 → 通信修正 → 决策引擎 → Chairman 决策 → 报告生成）
3. **智能诊断改为 SSE 流式**：`startDiagnosis()` 不再调用同步 API，而是直接订阅 `/api/agent/diagnose/stream`，根据后端真实事件驱动轮播文案，完成后自动切换到结果视图

回测页面同样增加 6 阶段轮播（获取K线 → 计算指标 → 策略判定 → Agent 决策 → LLM 解读 → 生成报告），并显示"预计耗时 3-15 秒"提示。

---

## 数据库

SQLite 数据库位于 `data/mass.db`，包含以下表：

- `agent_decisions` — 决策记录（含原始 JSON）
- `agent_opinions` — Agent 观点明细
- `virtual_positions` — 模拟持仓
- `validation_records` — 回测验证
- `prediction_records` — 预测记录与精度追踪（v2.3）
- `system_settings` — 系统配置（支持 Web UI 热更新）
- `agent_accuracy` — Agent 准确率统计

支持历史决策查询、准确率追踪、虚拟持仓盈亏计算。

---

## 目录结构

```
MASS/
├── agent/                      # 多智能体核心
│   ├── core/                   # 黑板、编排器、决策引擎、辩论引擎、缓存、容器
│   │   ├── orchestrator.py     # 编排调度（线程池复用、并行管线、信号量保护）
│   │   ├── blackboard.py       # 共享黑板（分片锁、有界缓存）
│   │   ├── cache.py            # 线程安全缓存管理器
│   │   ├── debate.py           # Agent 辩论引擎（已接入 Chairman）
│   │   ├── container.py        # DI 容器（替代强制单例）
│   │   ├── backtest_engine.py  # 回测引擎 v2.1+（5策略 + LLM解读 + 验证闭环）
│   │   ├── validator.py        # 决策验证器（5维偏差检测 + Prompt微调建议）
│   │   ├── agent_cache.py      # Agent结论缓存（指纹 + TTL）
│   │   ├── prompt_compressor.py # Prompt压缩器（白名单 + 精度截断 + 空值剪枝）
│   │   └── task_tracker.py     # 任务追踪（跨页持久化）
│   ├── agents/                 # 6 个专业 Agent + BaseAgent 基类
│   ├── tools/                  # LLM 客户端、数据工具、指标计算、情绪分析
│   │   ├── stock_data_tool.py  # 股票数据获取（单例、有界缓存、TTL 修复）
│   │   ├── indicator_tool.py   # 技术指标计算（真实 Beta）
│   │   ├── sentiment_tool.py   # 情绪分析（否定翻转+程度加权+转折句）
│   │   ├── llm_client.py       # 多厂商 LLM 统一客户端
│   │   └── batch_llm_client.py # 批量LLM客户端（透明拦截 + 批量合并）
│   ├── crawlers/               # 统一爬虫框架
│   │   ├── base.py             # 爬虫基类（断路器 + Session 池）
│   │   ├── session_pool.py     # 域名共享 Session
│   │   ├── circuit_breaker.py  # 断路器状态机
│   │   └── registry.py         # 注册表（字段级合并）
│   ├── models/                 # Pydantic 模型、数据库封装
│   │   └── database.py         # SQLite 线程本地连接池
│   └── prompts/system/         # 可热更新的 Prompt 模板
├── api/                        # Flask Blueprint + 中间件
│   ├── diagnose_bp.py          # 诊断 + SSE 流式
│   ├── portfolio_bp.py         # 组合分析
│   ├── prediction_bp.py        # 预测
│   ├── backtest_bp.py          # 回测（策略对比 + LLM预测集成）
│   ├── report_bp.py            # 报告生成
│   ├── history_bp.py           # 历史记录
│   ├── system_bp.py            # 系统统计（后台缓存）
│   ├── common.py               # 共享线程池与辅助函数
│   └── middleware.py           # 请求日志、错误处理、限流、CORS 白名单
├── static/                     # 前端静态资源
│   ├── js/charts/              # ECharts 图表引擎 + SSE 流处理器（断线重连）
│   └── css/                    # MASS 设计系统主题
├── templates/                  # HTML 模板
│   ├── agent_trading.html      # 个股诊断（流式 SSE 面板 + 重连遮罩）
│   ├── agent_portfolio.html    # 组合分析（8卡6图11表）
│   ├── agent_backtest.html     # 量化回测（5策略 + AI解读/预测 + 策略对比）
│   └── ...                     # 历史记录、监控、报告
├── tests/                      # 测试套件
│   ├── unit/                   # 376+ 个单元测试（全部通过）
│   ├── integration/            # 集成测试（网络依赖，@pytest.mark.integration）
│   ├── stress/                 # 压力测试
│   └── conftest.py             # 临时数据库隔离 + 同步执行器
├── app.py                      # Flask 应用入口
├── run.py                      # 启动脚本
├── config.py                   # 全局配置
├── pytest.ini                 # 默认跳过集成测试
└── requirements.txt
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | HTML5 + ECharts 5.x + Bootstrap 5 + 自定义 CSS 变量主题 |
| 后端 | Python 3.11 + Flask 2.3 + Waitress WSGI |
| 多智能体 | 6 专业 Agent + Chairman 决策 + Debate 辩论引擎 |
| LLM | DeepSeek-V4-Pro（默认）/ OpenAI GPT-4o / Claude / Ollama |
| 数据爬虫 | 5 源真实数据：EastMoney / Akshare / Sina / Tx / THS |
| 数据库 | SQLite + WAL 模式 + 线程本地连接池 |
| 缓存 | 内存缓存（TTL + 容量淘汰）+ Redis 分布式缓存 + orjson 序列化加速 |
| 部署 | `python deploy.py`（Waitress, host=0.0.0.0, port=5000） |

---

## 测试

```bash
# 运行单元测试（376+ 个，含 LLM 路径验证、Agent 通信、并发控制、RA 风险体系、回测验证、LLM 优化）
pytest tests/unit/ -v

# 运行集成测试（需外部网络）
pytest tests/integration/ -v -m integration

# 运行全部测试（含集成）
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=agent --cov=api --cov-report=html
```

---

## 配置说明

复制 `.env.example` 为 `.env` 并配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥 | - |
| `USE_MOCK_LLM` | 使用 Mock 模式（免密钥测试） | `True` |
| `USE_LLM_SENTIMENT` | 情感分析使用 LLM（高精度慢速） | `False` |
| `FLASK_DEBUG` | Flask 调试模式 | `True` |
| `FLASK_PORT` | 服务端口 | `5000` |
| `ALLOWED_ORIGINS` | CORS 白名单（逗号分隔） | `http://localhost:5000` |
| `WARMUP_STOCKS` | 启动预热的股票代码（逗号分隔） | （空） |
| `MAX_DIAGNOSIS_CONCURRENCY` | 最大并发诊断请求数（静态模式） | `8` |
| `ADAPTIVE_CONCURRENCY` | 启用自适应并发控制 | `True` |
| `CONCURRENCY_MIN` | 最小并发限制 | `2` |
| `CONCURRENCY_MAX` | 最大并发限制 | `10` |
| `CONCURRENCY_SLOW_THRESHOLD` | 响应时间慢阈值（秒） | `30.0` |
| `CONCURRENCY_FAST_THRESHOLD` | 响应时间快阈值（秒） | `10.0` |
| `CONCURRENCY_ERROR_HIGH` | 高错误率阈值 | `0.20` |
| `CONCURRENCY_ERROR_LOW` | 低错误率阈值 | `0.05` |
| `CONCURRENCY_WINDOW_SIZE` | 滑动窗口大小 | `50` |
| `AGENT_CACHE_ENABLED` | Agent 结论缓存开关 | `True` |
| `AGENT_CACHE_TTL` | Agent 缓存 TTL（秒） | `30` |
| `PROMPT_COMPRESSION_ENABLED` | Prompt 压缩开关 | `True` |
| `AGENT_LIGHT_MODEL` | 轻量 Agent 模型 | `deepseek-chat` |
| `AGENT_HEAVY_MODEL` | 重型 Agent 模型 | `deepseek-v4-pro` |
| `BATCH_INFERENCE_ENABLED` | 批量 LLM 推理开关（压测用） | `False` |
| `BATCH_INFERENCE_SIZE` | 每批合并的 Agent 数 | `3` |

---

## 扩展指南

### 添加新 Agent

1. 在 `agent/agents/` 创建 `{xxx}_agent.py`，继承 `BaseAgent`，实现 `analyze()`
2. 创建 `agent/prompts/system/xxx_agent.md` 系统提示词
3. 在 `agent/agents/__init__.py` 导出
4. 在 `agent/core/orchestrator.py` 注册
5. 在 `agent/models/agent_response.py` 添加输出模型

### 添加新数据源

在 `agent/tools/stock_data_tool.py` 中添加新方法，遵循**字段级合并**和**缓存模式**。

---

## 免责声明

本系统所有 Agent 输出仅供参考，不构成投资建议。股市有风险，投资需谨慎。
