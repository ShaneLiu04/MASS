"""
MASS MA-Agent: 宏观策略师
优化重点：
1. 利好-利空量化对冲分析（macro_score净得分）
2. 风格匹配一致性校验（行业vs市场风格）
3. 信号-置信度联动约束
4. 多层降级分析，绝不返回"数据不足"
5. 宏观→行业→个股传导链条
"""
from typing import Dict, Any, Optional, List, Tuple

from loguru import logger

from agent.agents.base_agent import BaseAgent
from agent.core.blackboard import StockSnapshot, AgentOpinion


class MA_Agent(BaseAgent):
    """宏观策略师 Agent"""

    # 行业-风格映射表
    INDUSTRY_STYLE_MAP = {
        "成长": ["新能源", "半导体", "计算机", "传媒", "医药", "医疗器械", "电子", "通信", "军工", "软件", "互联网", "光伏", "锂电", "储能"],
        "价值": ["银行", "保险", "证券", "房地产", "钢铁", "煤炭", "有色", "建材", "化工", "机械", "汽车", "建筑"],
        "防御": ["电力", "港口", "航运", "零售", "食品饮料", "白酒", "旅游", "公路", "机场"],
    }

    # 市场周期判断参考
    CYCLE_INDICATORS = [
        "复苏早期", "复苏晚期", "过热", "滞胀", "衰退早期", "衰退晚期"
    ]

    # 行业政策敏感度映射（monetary=货币政策, fiscal=财政政策, regulatory=监管政策,
    # trade=贸易政策, industrial=产业政策）
    POLICY_SENSITIVITY = {
        "房地产":   {"monetary": 0.90, "fiscal": 0.80, "regulatory": 0.95, "trade": 0.20, "industrial": 0.30},
        "银行":     {"monetary": 0.95, "fiscal": 0.50, "regulatory": 0.90, "trade": 0.30, "industrial": 0.20},
        "保险":     {"monetary": 0.85, "fiscal": 0.40, "regulatory": 0.80, "trade": 0.20, "industrial": 0.20},
        "证券":     {"monetary": 0.80, "fiscal": 0.30, "regulatory": 0.90, "trade": 0.20, "industrial": 0.20},
        "新能源":   {"monetary": 0.60, "fiscal": 0.90, "regulatory": 0.70, "trade": 0.50, "industrial": 0.95},
        "半导体":   {"monetary": 0.50, "fiscal": 0.70, "regulatory": 0.85, "trade": 0.90, "industrial": 0.95},
        "计算机":   {"monetary": 0.55, "fiscal": 0.60, "regulatory": 0.60, "trade": 0.40, "industrial": 0.80},
        "软件":     {"monetary": 0.50, "fiscal": 0.55, "regulatory": 0.55, "trade": 0.30, "industrial": 0.75},
        "传媒":     {"monetary": 0.40, "fiscal": 0.50, "regulatory": 0.80, "trade": 0.10, "industrial": 0.60},
        "医药":     {"monetary": 0.40, "fiscal": 0.60, "regulatory": 0.90, "trade": 0.30, "industrial": 0.85},
        "医疗器械": {"monetary": 0.35, "fiscal": 0.55, "regulatory": 0.90, "trade": 0.30, "industrial": 0.85},
        "电子":     {"monetary": 0.50, "fiscal": 0.65, "regulatory": 0.70, "trade": 0.80, "industrial": 0.85},
        "通信":     {"monetary": 0.45, "fiscal": 0.60, "regulatory": 0.65, "trade": 0.40, "industrial": 0.80},
        "军工":     {"monetary": 0.30, "fiscal": 0.80, "regulatory": 0.50, "trade": 0.20, "industrial": 0.95},
        "电力":     {"monetary": 0.70, "fiscal": 0.60, "regulatory": 0.50, "trade": 0.10, "industrial": 0.60},
        "港口":     {"monetary": 0.40, "fiscal": 0.50, "regulatory": 0.30, "trade": 0.70, "industrial": 0.40},
        "航运":     {"monetary": 0.45, "fiscal": 0.45, "regulatory": 0.35, "trade": 0.80, "industrial": 0.50},
        "零售":     {"monetary": 0.60, "fiscal": 0.70, "regulatory": 0.40, "trade": 0.20, "industrial": 0.30},
        "食品饮料": {"monetary": 0.50, "fiscal": 0.60, "regulatory": 0.45, "trade": 0.15, "industrial": 0.30},
        "白酒":     {"monetary": 0.55, "fiscal": 0.50, "regulatory": 0.50, "trade": 0.10, "industrial": 0.25},
        "旅游":     {"monetary": 0.50, "fiscal": 0.75, "regulatory": 0.40, "trade": 0.10, "industrial": 0.50},
        "公路":     {"monetary": 0.45, "fiscal": 0.55, "regulatory": 0.30, "trade": 0.05, "industrial": 0.20},
        "机场":     {"monetary": 0.40, "fiscal": 0.50, "regulatory": 0.35, "trade": 0.30, "industrial": 0.40},
        "钢铁":     {"monetary": 0.70, "fiscal": 0.80, "regulatory": 0.50, "trade": 0.60, "industrial": 0.70},
        "煤炭":     {"monetary": 0.65, "fiscal": 0.70, "regulatory": 0.55, "trade": 0.40, "industrial": 0.60},
        "有色":     {"monetary": 0.75, "fiscal": 0.60, "regulatory": 0.50, "trade": 0.80, "industrial": 0.70},
        "建材":     {"monetary": 0.65, "fiscal": 0.85, "regulatory": 0.45, "trade": 0.30, "industrial": 0.60},
        "化工":     {"monetary": 0.60, "fiscal": 0.55, "regulatory": 0.60, "trade": 0.50, "industrial": 0.65},
        "机械":     {"monetary": 0.55, "fiscal": 0.70, "regulatory": 0.40, "trade": 0.50, "industrial": 0.80},
        "汽车":     {"monetary": 0.55, "fiscal": 0.75, "regulatory": 0.60, "trade": 0.50, "industrial": 0.90},
        "建筑":     {"monetary": 0.60, "fiscal": 0.90, "regulatory": 0.40, "trade": 0.20, "industrial": 0.70},
        "石油":     {"monetary": 0.60, "fiscal": 0.50, "regulatory": 0.45, "trade": 0.70, "industrial": 0.60},
        "农业":     {"monetary": 0.45, "fiscal": 0.70, "regulatory": 0.40, "trade": 0.30, "industrial": 0.65},
    }

    # 政策方向关键词映射
    POLICY_DIRECTION_KEYWORDS = {
        "monetary": {
            "bullish": ["宽松", "降息", "降准", "流动性释放", "MLF净投放", "扩表"],
            "bearish": ["收紧", "加息", "缩表", "流动性回收", "MLF净回笼", "加息周期"],
        },
        "fiscal": {
            "bullish": ["积极", "扩张", "减税降费", "专项债", "基建投资", "财政补贴", "刺激"],
            "bearish": ["紧缩", "收缩", "减支", " austerity", "财政整固"],
        },
        "regulatory": {
            "bullish": ["放松", "松绑", "鼓励", "支持", "放宽准入", "注册制"],
            "bearish": ["收紧", "严监管", "整顿", "限制", "限购", "限贷", "反垄断", "加强监管"],
        },
        "trade": {
            "bullish": ["自贸", "关税降低", "贸易协定", "出口优惠", "RCEP", "一带一路"],
            "bearish": ["贸易战", "关税提高", "制裁", "脱钩", "出口管制", "技术封锁"],
        },
        "industrial": {
            "bullish": ["扶持", "补贴", "产业基金", "国产替代", "信创", "双碳", "新能源补贴"],
            "bearish": ["去产能", "淘汰", "限制", "补贴退坡", "退出"],
        },
    }

    # ── v2.3 全球经济联动映射表 ──
    # 1. 汇率敏感度：出口型（人民币贬值受益）vs 进口型（人民币升值受益）
    FX_SENSITIVITY = {
        # 出口导向型行业（人民币贬值利好，升值利空）
        "电子":       {"type": "export", "sensitivity": 0.85, "export_ratio": 0.60},
        "纺织":       {"type": "export", "sensitivity": 0.90, "export_ratio": 0.50},
        "家电":       {"type": "export", "sensitivity": 0.75, "export_ratio": 0.40},
        "机械":       {"type": "export", "sensitivity": 0.70, "export_ratio": 0.35},
        "化工":       {"type": "export", "sensitivity": 0.60, "export_ratio": 0.30},
        "汽车":       {"type": "export", "sensitivity": 0.65, "export_ratio": 0.20},
        "光伏":       {"type": "export", "sensitivity": 0.80, "export_ratio": 0.55},
        "锂电":       {"type": "export", "sensitivity": 0.75, "export_ratio": 0.45},
        "通信":       {"type": "export", "sensitivity": 0.55, "export_ratio": 0.25},
        "港口":       {"type": "export", "sensitivity": 0.70, "export_ratio": 0.30},
        "航运":       {"type": "export", "sensitivity": 0.80, "export_ratio": 0.35},
        # 进口依赖型行业（人民币升值利好，贬值利空）
        "航空":       {"type": "import", "sensitivity": 0.85, "import_ratio": 0.30},
        "机场":       {"type": "import", "sensitivity": 0.60, "import_ratio": 0.20},
        "造纸":       {"type": "import", "sensitivity": 0.75, "import_ratio": 0.40},
        "钢铁":       {"type": "import", "sensitivity": 0.70, "import_ratio": 0.50},
        "石油":       {"type": "import", "sensitivity": 0.80, "import_ratio": 0.60},
        "煤炭":       {"type": "import", "sensitivity": 0.50, "import_ratio": 0.10},
        "有色":       {"type": "import", "sensitivity": 0.75, "import_ratio": 0.45},
        "化工":       {"type": "import", "sensitivity": 0.65, "import_ratio": 0.35},
        # 双向敏感/对冲型
        "农业":       {"type": "mixed", "sensitivity": 0.50, "desc": "进口大豆依赖高，但部分农产品出口"},
        "汽车":       {"type": "mixed", "sensitivity": 0.50, "desc": "进口零部件+出口整车双重影响"},
        "半导体":     {"type": "import", "sensitivity": 0.70, "import_ratio": 0.50},
        "医药":       {"type": "import", "sensitivity": 0.55, "import_ratio": 0.25},
    }

    # 2. 大宗商品敏感度：各行业对原材料价格波动的敏感程度
    COMMODITY_SENSITIVITY = {
        # 油价敏感型
        "航空":       {"oil": 0.90, "oil_desc": "燃油成本占运营成本30-40%"},
        "航运":       {"oil": 0.85, "oil_desc": "燃油成本占运营成本20-30%"},
        "石油":       {"oil": -0.80, "oil_desc": "油价上涨利好上游勘探开采"},
        "化工":       {"oil": 0.70, "oil_desc": "石油化工原料成本占比高"},
        "建材":       {"oil": 0.50, "oil_desc": "能源成本占生产成本的15-20%"},
        "汽车":       {"oil": 0.40, "oil_desc": "油价影响消费者购车偏好"},
        "电力":       {"oil": 0.30, "oil_desc": "部分火电企业受煤炭/油价间接影响"},
        # 铜价敏感型（工业景气度指标）
        "有色":       {"copper": 0.80, "copper_desc": "铜价直接决定盈利水平"},
        "电力":       {"copper": 0.60, "copper_desc": "电网建设用铜量大"},
        "家电":       {"copper": 0.55, "copper_desc": "铜管、电机用铜"},
        "汽车":       {"copper": 0.50, "copper_desc": "电动车用铜量远高于燃油车"},
        "建筑":       {"copper": 0.45, "copper_desc": "建筑用线材、管材"},
        "电子":       {"copper": 0.50, "copper_desc": "PCB、连接器用铜"},
        "机械":       {"copper": 0.50, "copper_desc": "电机、设备用铜"},
        # 铁矿石敏感型
        "钢铁":       {"iron": 0.85, "iron_desc": "铁矿石占钢铁成本50-60%"},
        "机械":       {"iron": 0.40, "iron_desc": "钢材是主要原材料"},
        "建筑":       {"iron": 0.45, "iron_desc": "螺纹钢等建材成本"},
        "汽车":       {"iron": 0.40, "iron_desc": "钢材占车身成本比重高"},
        # 黄金敏感型
        "有色":       {"gold": 0.60, "gold_desc": "金矿企业直接受益"},
        # 农产品敏感型
        "农业":       {"agri": -0.70, "agri_desc": "农产品价格上涨利好种植养殖企业"},
        "食品饮料":   {"agri": 0.55, "agri_desc": "原材料成本占比高"},
        "白酒":       {"agri": 0.30, "agri_desc": "粮食原料成本"},
        "养殖":       {"agri": 0.65, "agri_desc": "饲料成本占养殖成本60-70%"},
    }

    # 3. 美股映射：A股行业/板块与美股对应板块的联动关系
    US_STOCK_MAPPING = {
        "半导体":     {"sector": "半导体/SOX", "correlation": 0.75, "lead_lag": "美股领先1-2天"},
        "电子":       {"sector": "科技/Nasdaq", "correlation": 0.65, "lead_lag": "美股领先1天"},
        "计算机":     {"sector": "软件科技/Nasdaq", "correlation": 0.60, "lead_lag": "美股领先1天"},
        "软件":       {"sector": "软件/SaaS", "correlation": 0.55, "lead_lag": "美股领先1天"},
        "互联网":     {"sector": "互联网/中概", "correlation": 0.80, "lead_lag": "同步或美股领先"},
        "传媒":       {"sector": "媒体/流媒体", "correlation": 0.45, "lead_lag": "联动较弱"},
        "医药":       {"sector": "生物科技/XBI", "correlation": 0.50, "lead_lag": "美股领先1-2天"},
        "医疗器械":   {"sector": "医疗科技", "correlation": 0.45, "lead_lag": "联动中等"},
        "新能源":     {"sector": "清洁能源", "correlation": 0.60, "lead_lag": "美股领先1天"},
        "光伏":       {"sector": "太阳能", "correlation": 0.65, "lead_lag": "美股领先1天"},
        "锂电":       {"sector": "电动汽车/TSLA产业链", "correlation": 0.60, "lead_lag": "美股领先1天"},
        "汽车":       {"sector": "电动汽车/TSLA", "correlation": 0.55, "lead_lag": "美股领先1天"},
        "银行":       {"sector": "金融/XLF", "correlation": 0.40, "lead_lag": "联动较弱"},
        "保险":       {"sector": "保险", "correlation": 0.35, "lead_lag": "联动较弱"},
        "证券":       {"sector": "投行/券商", "correlation": 0.40, "lead_lag": "联动较弱"},
        "石油":       {"sector": "能源/XLE", "correlation": 0.70, "lead_lag": "国际油价主导"},
        "化工":       {"sector": "化工", "correlation": 0.50, "lead_lag": "联动中等"},
        "钢铁":       {"sector": "材料/XLB", "correlation": 0.45, "lead_lag": "联动中等"},
        "有色":       {"sector": "材料/XLB", "correlation": 0.55, "lead_lag": "国际定价主导"},
        "煤炭":       {"sector": "煤炭", "correlation": 0.40, "lead_lag": "国内定价为主"},
        "农业":       {"sector": "农业", "correlation": 0.35, "lead_lag": "联动较弱"},
        "军工":       {"sector": "国防", "correlation": 0.30, "lead_lag": "联动弱，政策独立"},
        "房地产":     {"sector": "房地产", "correlation": 0.30, "lead_lag": "国内政策主导"},
        "建筑":       {"sector": "基建", "correlation": 0.25, "lead_lag": "联动弱"},
        "消费":       {"sector": "消费/XLY", "correlation": 0.45, "lead_lag": "联动中等"},
        "食品饮料":   {"sector": "消费必需品", "correlation": 0.35, "lead_lag": "联动较弱"},
        "白酒":       {"sector": "消费品", "correlation": 0.20, "lead_lag": "国内消费主导"},
        "旅游":       {"sector": "旅游/航空", "correlation": 0.40, "lead_lag": "联动中等"},
        "零售":       {"sector": "零售", "correlation": 0.35, "lead_lag": "联动较弱"},
        "电力":       {"sector": "公用事业", "correlation": 0.25, "lead_lag": "联动弱"},
        "通信":       {"sector": "电信", "correlation": 0.30, "lead_lag": "联动弱"},
        "港口":       {"sector": "航运", "correlation": 0.50, "lead_lag": "联动中等"},
        "航运":       {"sector": "航运", "correlation": 0.60, "lead_lag": "国际定价主导"},
    }

    def analyze(self, snapshot: StockSnapshot, user_position: Optional[Dict] = None) -> AgentOpinion:
        """宏观分析"""
        user_prompt = self._build_ma_prompt(snapshot)

        try:
            response = self._call_llm(user_prompt)
            parsed = self._safe_parse_llm_response(response)

            # 使用 macro_signal 作为信号（如果存在），否则用 signal
            macro_signal = parsed.get("macro_signal", parsed.get("signal", 0))
            parsed["signal"] = int(macro_signal)

            # 校验市场周期枚举
            cycle = parsed.get("market_cycle", "")
            if cycle not in self.CYCLE_INDICATORS:
                parsed["market_cycle"] = self._infer_market_cycle(snapshot)

            # 确保 macro_score 存在
            if "macro_score" not in parsed:
                parsed["macro_score"] = self._calculate_macro_score(snapshot)

            # 确保 style_alignment 存在
            if "style_alignment" not in parsed:
                parsed["style_alignment"] = self._infer_style_alignment(snapshot)

            # 信号-置信度一致性校验
            confidence = float(parsed.get("confidence", 0.5))
            if parsed["signal"] in (1, -1) and confidence < 0.60:
                logger.warning(f"MA-Agent: signal={parsed['signal']}但confidence={confidence:.2f}<0.60，已提升")
                confidence = max(confidence, 0.60)
                parsed["confidence"] = confidence

            # 风格匹配一致性校验：style_alignment < 0.5 时禁止买入
            style_align = float(parsed.get("style_alignment", 0.5))
            if style_align < 0.5 and parsed["signal"] == 1:
                logger.warning(f"MA-Agent: 风格不匹配({style_align:.2f})但信号为买入，修正为观望")
                parsed["signal"] = 0
                parsed["confidence"] = min(confidence, 0.55)

            # macro_score 与 signal 联动校验
            mscore = int(parsed.get("macro_score", 0))
            expected_signal = 1 if mscore >= 3 else (-1 if mscore <= -3 else 0)
            if parsed["signal"] != expected_signal:
                logger.warning(f"MA-Agent: signal({parsed['signal']})与macro_score({mscore})不匹配，已修正")
                parsed["signal"] = expected_signal

            # 校验权重调整之和
            adj = parsed.get("recommended_weight_adjustment", {})
            if isinstance(adj, dict):
                total_adj = sum(v for v in adj.values() if isinstance(v, (int, float)))
                if abs(total_adj) > 0.15:
                    logger.warning(f"MA-Agent权重调整之和({total_adj})超过0.15，已截断")
                    scale = 0.15 / abs(total_adj)
                    adj = {k: v * scale for k, v in adj.items()}
                    parsed["recommended_weight_adjustment"] = adj

            return self._build_default_opinion(
                signal=parsed["signal"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                raw_data=parsed,
            )
        except Exception as e:
            logger.error(f"MA-Agent分析失败: {e}")
            return self._fallback_opinion(snapshot)

    def _analyze_industry_cycle(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        行业景气度周期定位 — 基于季度财务数据推断行业所处周期阶段

        周期阶段判定逻辑（四周期模型）：
        - 复苏期: 盈利拐点向上 + 毛利率改善 + 营收加速
        - 繁荣期: 盈利持续上升 + 毛利率高位 + 营收强劲
        - 衰退期: 盈利下滑 + 毛利率恶化 + 营收减速
        - 萧条期: 盈利触底 + 毛利率低位企稳 + 营收萎缩但降速

        代理指标（因无法直接获取库存/资本开支数据）：
        - 库存周期代理: 营收增速变化（营收加速=被动去库存/主动补库存）
        - 资本开支代理: 盈利增速 vs 营收增速（盈利增速>营收增速=效率提升/资本开支谨慎）
        - 盈利周期: 净利润趋势 + ROE 趋势
        """
        fundamentals = snapshot.fundamentals or {}
        industry = fundamentals.get("industry", "")
        quarters = fundamentals.get("quarterly_data", [])

        result = {
            "stage": "数据不足",
            "inventory_cycle": "未知",
            "capex_cycle": "未知",
            "earnings_cycle": "未知",
            "profit_trend": "未知",
            "revenue_trend": "未知",
            "margin_trend": "未知",
            "revenue_growth_trend": "未知",
            "composite_score": 50,
            "cycle_phase_num": 0,  # 0=未知, 1=萧条, 2=复苏, 3=繁荣, 4=衰退
        }

        if not quarters or len(quarters) < 2:
            return result

        # 提取有效数据
        profits = []
        revenues = []
        margins = []
        roes = []
        for q in quarters:
            if q.get("net_profit") is not None:
                profits.append(float(q["net_profit"]))
            if q.get("revenue") is not None:
                revenues.append(float(q["revenue"]))
            if q.get("gross_margin") is not None:
                margins.append(float(q["gross_margin"]))
            elif q.get("roe") is not None:
                # 无毛利率时用ROE近似
                roes.append(float(q["roe"]))

        # 计算趋势（至少3个数据点）
        def _trend_direction(values, threshold_pct=3.0):
            if len(values) < 3:
                return "未知"
            # 用最近一期 vs 三期前的变化判断趋势
            recent = values[-1]
            prior = values[-3]
            if prior == 0:
                return "未知"
            change_pct = (recent - prior) / abs(prior) * 100
            if change_pct > threshold_pct:
                return "上升"
            elif change_pct < -threshold_pct:
                return "下降"
            else:
                return "平稳"

        def _trend_slope(values):
            """计算线性趋势斜率（标准化）"""
            if len(values) < 3:
                return 0.0
            import numpy as np
            x = np.arange(len(values))
            y = np.array(values)
            # 简单线性回归
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
            # 标准化斜率（相对于均值）
            mean_y = np.mean(y)
            if mean_y != 0:
                slope = slope / abs(mean_y)
            return float(slope)

        result["profit_trend"] = _trend_direction(profits) if profits else "未知"
        result["revenue_trend"] = _trend_direction(revenues) if revenues else "未知"

        if margins:
            result["margin_trend"] = _trend_direction(margins)
        elif roes:
            result["margin_trend"] = _trend_direction(roes)
        else:
            result["margin_trend"] = "未知"

        # 营收增速趋势（用于库存周期代理）
        if len(revenues) >= 3:
            growth_rates = []
            for i in range(1, len(revenues)):
                if revenues[i - 1] != 0:
                    growth_rates.append((revenues[i] - revenues[i - 1]) / abs(revenues[i - 1]) * 100)
            if len(growth_rates) >= 2:
                result["revenue_growth_trend"] = "加速" if growth_rates[-1] > growth_rates[0] else "减速"

        # ── 周期阶段判定 ──
        pt = result["profit_trend"]
        rt = result["revenue_trend"]
        mt = result["margin_trend"]
        rgt = result["revenue_growth_trend"]

        score = 50

        # 盈利趋势权重最高
        if pt == "上升":
            score += 20
        elif pt == "下降":
            score -= 20

        # 营收趋势
        if rt == "上升":
            score += 10
        elif rt == "下降":
            score -= 10

        # 毛利率/ROE趋势
        if mt == "上升":
            score += 15
        elif mt == "下降":
            score -= 15

        # 营收增速（库存周期代理）
        if rgt == "加速":
            score += 10
            inventory = "主动补库存或需求拉动"
        elif rgt == "减速":
            score -= 5
            inventory = "被动去库存或需求放缓"
        else:
            inventory = "库存周期不明朗"

        # 综合判定周期阶段
        if pt == "上升" and mt == "上升":
            if rgt == "加速":
                stage = "繁荣期"
                phase_num = 3
                capex = "资本开支高位扩张"
                earnings = "盈利加速释放"
            else:
                stage = "复苏期"
                phase_num = 2
                capex = "资本开支低位回升"
                earnings = "盈利拐点确认"
        elif pt == "下降" and mt == "下降":
            if rt == "下降":
                stage = "衰退期"
                phase_num = 4
                capex = "资本开支收缩"
                earnings = "盈利持续下滑"
            else:
                stage = "衰退期→萧条期过渡"
                phase_num = 4
                capex = "资本开支谨慎"
                earnings = "盈利下滑但营收有韧性"
        elif pt == "下降" and (mt == "平稳" or mt == "未知"):
            if rt == "平稳" or rt == "上升":
                stage = "萧条期"
                phase_num = 1
                capex = "资本开支冰点"
                earnings = "盈利触底企稳"
            else:
                stage = "衰退期"
                phase_num = 4
                capex = "资本开支收缩"
                earnings = "盈利下滑"
        elif pt == "平稳" and mt == "上升":
            stage = "复苏早期"
            phase_num = 2
            capex = "资本开支开始回升"
            earnings = "盈利拐点初现"
        elif pt == "平稳" and mt == "下降":
            stage = "衰退晚期"
            phase_num = 4
            capex = "资本开支底部"
            earnings = "盈利下行压力"
        else:
            # 混合信号
            if score >= 65:
                stage = "复苏期/繁荣期"
                phase_num = 2
            elif score <= 35:
                stage = "衰退期"
                phase_num = 4
            else:
                stage = "过渡期/震荡期"
                phase_num = 0
            capex = "资本开支观望"
            earnings = "盈利分化"

        result["stage"] = stage
        result["inventory_cycle"] = inventory
        result["capex_cycle"] = capex
        result["earnings_cycle"] = earnings
        result["composite_score"] = max(0, min(100, score))
        result["cycle_phase_num"] = phase_num

        return result

    def _get_policy_sensitivity(self, industry: str) -> Dict[str, float]:
        """获取行业政策敏感度配置，未知行业返回默认值"""
        if not industry:
            return {"monetary": 0.50, "fiscal": 0.50, "regulatory": 0.50, "trade": 0.30, "industrial": 0.50}
        # 精确匹配
        if industry in self.POLICY_SENSITIVITY:
            return self.POLICY_SENSITIVITY[industry]
        # 模糊匹配（子串）
        for ind_name, sens in self.POLICY_SENSITIVITY.items():
            if ind_name in industry or industry in ind_name:
                return sens
        return {"monetary": 0.50, "fiscal": 0.50, "regulatory": 0.50, "trade": 0.30, "industrial": 0.50}

    def _parse_policy_direction(self, macro: Dict) -> Dict[str, int]:
        """解析当前政策环境，返回各维度方向 (-1=利空, 0=中性, +1=利好)"""
        directions = {"monetary": 0, "fiscal": 0, "regulatory": 0, "trade": 0, "industrial": 0}

        # 货币政策信号源
        policy_stance = macro.get("policy_stance", "")
        fed_policy = macro.get("fed_policy", "")
        bond_trend = macro.get("bond_yield_trend", "")
        monetary_text = f"{policy_stance} {fed_policy} {bond_trend}"

        for kw in self.POLICY_DIRECTION_KEYWORDS["monetary"]["bullish"]:
            if kw in monetary_text:
                directions["monetary"] = 1
                break
        if directions["monetary"] == 0:
            for kw in self.POLICY_DIRECTION_KEYWORDS["monetary"]["bearish"]:
                if kw in monetary_text:
                    directions["monetary"] = -1
                    break

        # 财政政策信号源
        fiscal_text = macro.get("fiscal_policy", "") or policy_stance
        for kw in self.POLICY_DIRECTION_KEYWORDS["fiscal"]["bullish"]:
            if kw in fiscal_text:
                directions["fiscal"] = 1
                break
        if directions["fiscal"] == 0:
            for kw in self.POLICY_DIRECTION_KEYWORDS["fiscal"]["bearish"]:
                if kw in fiscal_text:
                    directions["fiscal"] = -1
                    break

        # 监管政策信号源
        regulatory_text = macro.get("regulatory_policy", "") or policy_stance
        for kw in self.POLICY_DIRECTION_KEYWORDS["regulatory"]["bullish"]:
            if kw in regulatory_text:
                directions["regulatory"] = 1
                break
        if directions["regulatory"] == 0:
            for kw in self.POLICY_DIRECTION_KEYWORDS["regulatory"]["bearish"]:
                if kw in regulatory_text:
                    directions["regulatory"] = -1
                    break

        # 贸易政策信号源
        trade_text = macro.get("trade_policy", "") or macro.get("rmb_trend", "")
        for kw in self.POLICY_DIRECTION_KEYWORDS["trade"]["bullish"]:
            if kw in trade_text:
                directions["trade"] = 1
                break
        if directions["trade"] == 0:
            for kw in self.POLICY_DIRECTION_KEYWORDS["trade"]["bearish"]:
                if kw in trade_text:
                    directions["trade"] = -1
                    break

        # 产业政策信号源
        industrial_text = macro.get("industrial_policy", "") or policy_stance
        for kw in self.POLICY_DIRECTION_KEYWORDS["industrial"]["bullish"]:
            if kw in industrial_text:
                directions["industrial"] = 1
                break
        if directions["industrial"] == 0:
            for kw in self.POLICY_DIRECTION_KEYWORDS["industrial"]["bearish"]:
                if kw in industrial_text:
                    directions["industrial"] = -1
                    break

        return directions

    def _generate_transmission_analysis(
        self, industry: str, sensitivity: Dict[str, float], directions: Dict[str, int]
    ) -> List[str]:
        """生成政策传导预期文本"""
        transmissions = []
        dim_names = {
            "monetary": "货币政策", "fiscal": "财政政策",
            "regulatory": "监管政策", "trade": "贸易政策", "industrial": "产业政策",
        }

        # 货币政策传导
        if sensitivity.get("monetary", 0) > 0.7 and directions["monetary"] != 0:
            if directions["monetary"] == 1:
                if "银行" in industry or "保险" in industry or "证券" in industry:
                    transmissions.append("利率下行将利好金融业资产端扩张，但压缩净息差")
                elif "房地产" in industry:
                    transmissions.append("降息降准直接降低融资成本，刺激购房需求")
                else:
                    transmissions.append("宽松货币降低整体融资成本，利好资本密集型行业")
            else:
                if "银行" in industry or "保险" in industry or "证券" in industry:
                    transmissions.append("加息周期提升净息差，但可能抑制信贷需求")
                elif "房地产" in industry:
                    transmissions.append("加息直接提高按揭成本，压制房地产销售")
                else:
                    transmissions.append("货币收紧推高融资成本，对高负债企业不利")

        # 财政政策传导
        if sensitivity.get("fiscal", 0) > 0.7 and directions["fiscal"] == 1:
            if "建筑" in industry or "建材" in industry or "钢铁" in industry:
                transmissions.append("基建投资扩张直接拉动上游需求")
            elif "新能源" in industry or "汽车" in industry:
                transmissions.append("财政补贴和购置税优惠刺激终端消费")
            else:
                transmissions.append("积极财政扩大总需求，带动行业景气度回升")

        # 监管政策传导
        if sensitivity.get("regulatory", 0) > 0.7:
            if directions["regulatory"] == -1:
                if "房地产" in industry:
                    transmissions.append("监管趋严限制房企融资和购房杠杆，压制行业扩张")
                elif "医药" in industry or "医疗器械" in industry:
                    transmissions.append("集采和医保控费压缩利润空间，行业面临定价压力")
                elif "互联网" in industry or "传媒" in industry:
                    transmissions.append("反垄断和内容监管增加合规成本")
                elif "银行" in industry or "证券" in industry:
                    transmissions.append("金融监管趋严提升合规成本，限制创新业务")
            elif directions["regulatory"] == 1:
                if "房地产" in industry:
                    transmissions.append("监管边际放松有助于房企融资环境改善")
                elif "医药" in industry:
                    transmissions.append("创新药审评加速，鼓励研发的政策红利释放")

        # 贸易政策传导
        if sensitivity.get("trade", 0) > 0.6 and directions["trade"] != 0:
            if directions["trade"] == -1:
                if "半导体" in industry or "电子" in industry:
                    transmissions.append("技术封锁和出口管制加速国产替代，短期阵痛长期利好头部企业")
                elif "港口" in industry or "航运" in industry:
                    transmissions.append("贸易摩擦压制进出口货量，影响港口吞吐和运价")
            elif directions["trade"] == 1:
                if "港口" in industry or "航运" in industry:
                    transmissions.append("贸易协定和关税降低提振进出口，利好物流链")

        # 产业政策传导
        if sensitivity.get("industrial", 0) > 0.8 and directions["industrial"] == 1:
            if "新能源" in industry or "光伏" in industry or "锂电" in industry or "储能" in industry:
                transmissions.append("双碳目标和新能源补贴推动行业高速增长")
            elif "半导体" in industry:
                transmissions.append("国产替代和信创政策提供确定性需求支撑")
            elif "军工" in industry:
                transmissions.append("国防预算增长和装备升级带来长期订单保障")
            elif "汽车" in industry:
                transmissions.append("新能源汽车产业扶持政策推动电动化转型")

        return transmissions if transmissions else ["当前政策环境对该行业影响中性，需关注后续政策动向"]

    def _analyze_policy_impact(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        政策敏感性分析 — 基于行业×政策维度的传导效应评估

        核心逻辑：
        - 每个行业对5类政策有不同敏感度（0~1）
        - 解析当前政策环境的方向（利好/利空/中性）
        - 敏感度 × 方向 = 维度影响得分
        - 综合所有维度 = 政策影响总得分
        - 识别最大风险维度（高敏感度 + 不利方向）
        """
        macro = snapshot.macro_data or {}
        industry = snapshot.fundamentals.get("industry", "")

        result = {
            "industry": industry or "未知",
            "sensitivity": {},
            "policy_environment": {},
            "impact_score": 0.0,
            "impact_direction": "中性",
            "dimension_scores": {},
            "transmission_analysis": [],
            "risk_radar": {},
            "policy_score": 50,
        }

        # 1. 获取行业敏感度
        sensitivity = self._get_policy_sensitivity(industry)
        result["sensitivity"] = sensitivity

        # 2. 解析政策方向
        directions = self._parse_policy_direction(macro)
        result["policy_environment"] = directions

        # 3. 计算各维度得分和总得分
        impact_score = 0.0
        dimension_scores = {}
        for dim, sens in sensitivity.items():
            dir_val = directions.get(dim, 0)
            dim_score = sens * dir_val  # 敏感度 × 方向
            dimension_scores[dim] = round(dim_score, 3)
            impact_score += dim_score

        result["dimension_scores"] = dimension_scores
        result["impact_score"] = round(impact_score, 2)

        # 4. 影响方向判定
        if impact_score >= 1.5:
            result["impact_direction"] = "利好"
        elif impact_score >= 0.5:
            result["impact_direction"] = "偏利好"
        elif impact_score <= -1.5:
            result["impact_direction"] = "利空"
        elif impact_score <= -0.5:
            result["impact_direction"] = "偏利空"
        else:
            result["impact_direction"] = "中性"

        # 5. 标准化政策评分 (0-100)
        result["policy_score"] = max(0, min(100, 50 + impact_score * 15))

        # 6. 生成传导预期文本
        result["transmission_analysis"] = self._generate_transmission_analysis(
            industry, sensitivity, directions
        )

        # 7. 风险雷达：找出敏感度最高且方向不利的维度
        risk_candidates = []
        for dim, sens in sensitivity.items():
            dir_val = directions.get(dim, 0)
            if dir_val == -1 and sens >= 0.5:
                risk_candidates.append((dim, sens))
        if risk_candidates:
            risk_candidates.sort(key=lambda x: x[1], reverse=True)
            top_risk_dim, top_risk_sens = risk_candidates[0]
            dim_names = {
                "monetary": "货币政策", "fiscal": "财政政策",
                "regulatory": "监管政策", "trade": "贸易政策", "industrial": "产业政策",
            }
            result["risk_radar"] = {
                "dimension": dim_names.get(top_risk_dim, top_risk_dim),
                "sensitivity": top_risk_sens,
                "direction": "不利",
                "warning": f"该行业对{dim_names.get(top_risk_dim, top_risk_dim)}敏感度极高({top_risk_sens:.0%})，当前政策方向不利",
            }
        else:
            result["risk_radar"] = {"warning": "当前无明显政策风险维度"}

        return result

    def _get_fx_sensitivity(self, industry: str) -> Dict[str, Any]:
        """获取行业汇率敏感度配置"""
        if not industry:
            return {"type": "neutral", "sensitivity": 0.0}
        if industry in self.FX_SENSITIVITY:
            return self.FX_SENSITIVITY[industry]
        for ind_name, sens in self.FX_SENSITIVITY.items():
            if ind_name in industry or industry in ind_name:
                return sens
        return {"type": "neutral", "sensitivity": 0.0}

    def _get_commodity_sensitivity(self, industry: str) -> Dict[str, Any]:
        """获取行业大宗商品敏感度配置"""
        if not industry:
            return {}
        if industry in self.COMMODITY_SENSITIVITY:
            return self.COMMODITY_SENSITIVITY[industry]
        for ind_name, sens in self.COMMODITY_SENSITIVITY.items():
            if ind_name in industry or industry in ind_name:
                return sens
        return {}

    def _get_us_stock_mapping(self, industry: str) -> Dict[str, Any]:
        """获取行业美股映射配置"""
        if not industry:
            return {}
        if industry in self.US_STOCK_MAPPING:
            return self.US_STOCK_MAPPING[industry]
        for ind_name, mapping in self.US_STOCK_MAPPING.items():
            if ind_name in industry or industry in ind_name:
                return mapping
        return {}

    def _analyze_global_economy(self, snapshot: StockSnapshot) -> Dict[str, Any]:
        """
        全球经济联动分析 v2.3 — 汇率敏感性、大宗商品、美股映射

        核心逻辑：
        - 汇率：出口型行业人民币贬值利好，进口型人民币升值利好
        - 大宗商品：油价/铜价/铁矿石等原材料成本对各行业的影响
        - 美股映射：中概股/ADR与A股相关板块的联动传导
        """
        macro = snapshot.macro_data or {}
        market = snapshot.market_context or {}
        industry = snapshot.fundamentals.get("industry", "")

        result = {
            "industry": industry or "未知",
            "fx_analysis": {},
            "commodity_analysis": {},
            "us_stock_mapping": {},
            "global_score": 0.0,
            "global_direction": "中性",
            "global_score_100": 50,
        }

        # ── 1. 汇率敏感性分析 ──
        fx_sens = self._get_fx_sensitivity(industry)
        if fx_sens and fx_sens.get("sensitivity", 0) > 0:
            rmb_trend = macro.get("rmb_trend", "")
            fx_impact = 0.0
            fx_reason = ""

            fx_type = fx_sens.get("type", "neutral")
            fx_sensitivity = fx_sens.get("sensitivity", 0)

            if "贬值" in rmb_trend or "走弱" in rmb_trend:
                if fx_type == "export":
                    fx_impact = fx_sensitivity  # 出口型受益
                    fx_reason = f"人民币贬值利好出口竞争力，{industry}出口型行业受益"
                elif fx_type == "import":
                    fx_impact = -fx_sensitivity  # 进口型受损
                    fx_reason = f"人民币贬值推高进口成本，{industry}进口依赖型行业承压"
                elif fx_type == "mixed":
                    fx_impact = fx_sensitivity * 0.2  # 混合影响较小
                    fx_reason = f"人民币贬值对{industry}影响复杂，需区分进出口结构"
            elif "升值" in rmb_trend or "走强" in rmb_trend:
                if fx_type == "export":
                    fx_impact = -fx_sensitivity  # 出口型受损
                    fx_reason = f"人民币升值削弱出口竞争力，{industry}出口型行业承压"
                elif fx_type == "import":
                    fx_impact = fx_sensitivity  # 进口型受益
                    fx_reason = f"人民币升值降低进口成本，{industry}进口依赖型行业受益"
                elif fx_type == "mixed":
                    fx_impact = fx_sensitivity * 0.2
                    fx_reason = f"人民币升值对{industry}影响复杂，需区分进出口结构"
            else:
                fx_reason = f"人民币汇率趋势不明，{industry}汇率影响暂无法量化"

            result["fx_analysis"] = {
                "industry_type": fx_type,
                "sensitivity": fx_sensitivity,
                "rmb_trend": rmb_trend or "未知",
                "impact_score": round(fx_impact, 2),
                "reasoning": fx_reason,
            }
            result["global_score"] += fx_impact

        # ── 2. 大宗商品价格分析 ──
        commodity_sens = self._get_commodity_sensitivity(industry)
        commodity_impacts = []
        commodity_total = 0.0

        # 油价
        oil_trend = macro.get("oil_trend", "")
        oil_sens = commodity_sens.get("oil", 0)
        if oil_sens != 0 and oil_trend:
            if "上涨" in oil_trend or "上行" in oil_trend:
                oil_impact = oil_sens  # 正值=成本上升利空，负值=收入增加利好
            elif "下跌" in oil_trend or "下行" in oil_trend:
                oil_impact = -oil_sens
            else:
                oil_impact = 0.0
            if oil_impact != 0:
                commodity_total += oil_impact
                direction = "利空" if oil_impact > 0 else "利好"
                commodity_impacts.append({
                    "commodity": "原油",
                    "trend": oil_trend,
                    "impact": round(oil_impact, 2),
                    "reasoning": f"油价{oil_trend}对{industry}为{direction}（{commodity_sens.get('oil_desc', '')}）",
                })

        # 铜价（全球经济景气度指标）
        copper_trend = macro.get("copper_trend", "")
        copper_sens = commodity_sens.get("copper", 0)
        if copper_sens != 0 and copper_trend:
            if "上涨" in copper_trend or "上行" in copper_trend:
                copper_impact = copper_sens
            elif "下跌" in copper_trend or "下行" in copper_trend:
                copper_impact = -copper_sens
            else:
                copper_impact = 0.0
            if copper_impact != 0:
                commodity_total += copper_impact
                direction = "利空" if copper_impact > 0 else "利好"
                commodity_impacts.append({
                    "commodity": "铜",
                    "trend": copper_trend,
                    "impact": round(copper_impact, 2),
                    "reasoning": f"铜价{copper_trend}对{industry}为{direction}（{commodity_sens.get('copper_desc', '')}）",
                })

        # 铁矿石
        iron_trend = macro.get("iron_trend", "")
        iron_sens = commodity_sens.get("iron", 0)
        if iron_sens != 0 and iron_trend:
            if "上涨" in iron_trend or "上行" in iron_trend:
                iron_impact = iron_sens
            elif "下跌" in iron_trend or "下行" in iron_trend:
                iron_impact = -iron_sens
            else:
                iron_impact = 0.0
            if iron_impact != 0:
                commodity_total += iron_impact
                direction = "利空" if iron_impact > 0 else "利好"
                commodity_impacts.append({
                    "commodity": "铁矿石",
                    "trend": iron_trend,
                    "impact": round(iron_impact, 2),
                    "reasoning": f"铁矿石{iron_trend}对{industry}为{direction}（{commodity_sens.get('iron_desc', '')}）",
                })

        # 农产品
        agri_trend = macro.get("agri_trend", "")
        agri_sens = commodity_sens.get("agri", 0)
        if agri_sens != 0 and agri_trend:
            if "上涨" in agri_trend or "上行" in agri_trend:
                agri_impact = agri_sens
            elif "下跌" in agri_trend or "下行" in agri_trend:
                agri_impact = -agri_sens
            else:
                agri_impact = 0.0
            if agri_impact != 0:
                commodity_total += agri_impact
                direction = "利空" if agri_impact > 0 else "利好"
                commodity_impacts.append({
                    "commodity": "农产品",
                    "trend": agri_trend,
                    "impact": round(agri_impact, 2),
                    "reasoning": f"农产品{agri_trend}对{industry}为{direction}（{commodity_sens.get('agri_desc', '')}）",
                })

        if commodity_impacts:
            result["commodity_analysis"] = {
                "total_impact": round(commodity_total, 2),
                "impacts": commodity_impacts,
            }
            result["global_score"] += commodity_total

        # ── 3. 美股映射分析 ──
        us_mapping = self._get_us_stock_mapping(industry)
        if us_mapping:
            us_sector = us_mapping.get("sector", "")
            correlation = us_mapping.get("correlation", 0)
            lead_lag = us_mapping.get("lead_lag", "")

            # 从 market_context 获取美股/中概股数据
            us_data = market.get("us_market", {})
            us_impact = 0.0
            us_reason = f"{industry}与美股{us_sector}相关系数{correlation}，{lead_lag}"

            if us_data:
                us_pct = us_data.get("pct_change", 0)
                if us_pct > 0.02:
                    us_impact = correlation * 1.0
                    us_reason += f"；隔夜美股大涨({us_pct:+.2%})，正向传导"
                elif us_pct > 0:
                    us_impact = correlation * 0.5
                    us_reason += f"；隔夜美股上涨({us_pct:+.2%})，轻微正向传导"
                elif us_pct < -0.02:
                    us_impact = -correlation * 1.0
                    us_reason += f"；隔夜美股大跌({us_pct:+.2%})，负向传导"
                elif us_pct < 0:
                    us_impact = -correlation * 0.5
                    us_reason += f"；隔夜美股下跌({us_pct:+.2%})，轻微负向传导"
                else:
                    us_reason += "；隔夜美股波动不大"
            else:
                us_reason += "；无隔夜美股数据"

            result["us_stock_mapping"] = {
                "us_sector": us_sector,
                "correlation": correlation,
                "lead_lag": lead_lag,
                "us_change_pct": us_data.get("pct_change", 0) if us_data else None,
                "impact_score": round(us_impact, 2),
                "reasoning": us_reason,
            }
            result["global_score"] += us_impact

        # ── 4. 综合判定 ──
        gs = result["global_score"]
        if gs >= 1.5:
            result["global_direction"] = "利好"
        elif gs >= 0.5:
            result["global_direction"] = "偏利好"
        elif gs <= -1.5:
            result["global_direction"] = "利空"
        elif gs <= -0.5:
            result["global_direction"] = "偏利空"
        else:
            result["global_direction"] = "中性"

        result["global_score"] = round(gs, 2)
        result["global_score_100"] = max(0, min(100, 50 + gs * 15))

        return result

    def _build_ma_prompt(self, snapshot: StockSnapshot) -> str:
        """构建宏观分析Prompt v2.3 — 自动推导隐含指标、风格匹配检测、行业景气度、政策敏感度、全球经济联动"""
        fundamentals = snapshot.fundamentals
        market = snapshot.market_context or {}
        macro = snapshot.macro_data or {}
        industry = fundamentals.get("industry", "")

        parts = [
            f"## 分析对象",
            f"股票代码: {snapshot.stock_code}",
            f"股票名称: {snapshot.stock_name}",
            f"所属行业: {industry if industry else '未知'}",
            f"当前价格: {snapshot.current_price}",
            "",
            "## 分析任务",
            "你是一位宏观策略师。请基于以下数据（部分可能缺失），",
            "构建'宏观→行业→个股'的传导链条，对利好与风险进行量化对冲分析。",
            "",
            "**重要：即使部分宏观数据缺失，也必须基于已有数据做合理推断，",
            '禁止返回"数据缺失"、"无法判断"等消极表述。**',
            "",
        ]

        # ── 1. 市场环境 ──
        indices = market.get("indices", {})
        if indices:
            parts.extend(["### 一、市场环境（大盘指数）"])
            for name, data in indices.items():
                close = data.get("close", 0)
                pct = data.get("pct_change", 0)
                up = data.get("up_count", 0)
                down = data.get("down_count", 0)
                trend = "上行" if pct > 0.02 else ("下行" if pct < -0.02 else "震荡")
                parts.append(f"- {name}: {close:.2f} ({pct:+.2f}%) [{trend}]")
                if up > 0 and down > 0:
                    ratio = up / (up + down) if (up + down) > 0 else 0.5
                    parts.append(f"  涨跌家数比: {up}/{down} ({ratio:.0%})")
            parts.append("")

        # 市场波动率
        vol = market.get("market_volatility_20d")
        if vol is not None:
            vol_desc = "高波动/恐慌" if vol > 25 else ("低波动/压抑" if vol < 15 else "正常波动")
            parts.append(f"- 20日市场波动率: {vol:.2f}% ({vol_desc})")

        # 市场风格
        mstyle = market.get("market_style", "")
        mcstyle = market.get("market_cap_style", "")
        if mstyle or mcstyle:
            parts.append(f"- 当前市场风格: {mstyle}/{mcstyle}")

        # ── 风格匹配预检测 ──
        stock_style = self._get_industry_style(industry)
        style_match = self._infer_style_alignment(snapshot)
        parts.extend([
            "",
            f"### 风格匹配预检测",
            f"- 该股票所属行业风格: {stock_style}",
            f"- 当前市场风格: {mstyle}/{mcstyle}",
            f"- 风格匹配度: {style_match:.2f} {'(匹配)' if style_match >= 0.5 else '(不匹配，需谨慎) '}",
            "",
        ])

        # ── 2. 板块与行业数据 ──
        sector_top = market.get("sector_top", [])
        if sector_top:
            parts.extend(["### 二、板块资金流向（前10）"])
            industry_in_top = False
            industry_flow = 0
            for s in sector_top[:10]:
                sname = s.get("name", "")
                pct = s.get("pct_change", 0)
                flow = s.get("main_net_inflow", 0)
                flow_desc = "流入" if flow > 0 else "流出"
                marker = ""
                if industry and industry in sname:
                    marker = " [该股票所在行业]"
                    industry_in_top = True
                    industry_flow = flow
                parts.append(f"- {sname}: {pct:+.2f}% ({flow_desc}{abs(flow)/1e8:.1f}亿){marker}")
            parts.append("")
            if industry_in_top:
                parts.append(f"- **该股票所属行业资金动向: {industry_flow/1e8:+.1f}亿**")
            else:
                parts.append(f"- 该股票所属行业未进入板块流向前十")
            parts.append("")

        # ── 3. 行业景气度周期定位（v2.1 新增） ──
        cycle = self._analyze_industry_cycle(snapshot)
        if cycle.get("stage") != "数据不足":
            parts.extend([
                "### 三、行业景气度周期定位",
                f"- **行业**: {industry if industry else '未知'}",
                f"- **当前周期阶段**: {cycle['stage']}",
                f"- 盈利趋势: {cycle['profit_trend']}",
                f"- 营收趋势: {cycle['revenue_trend']}",
                f"- 毛利率/ROE趋势: {cycle['margin_trend']}",
                f"- 营收增速（库存周期代理）: {cycle['revenue_growth_trend']}",
                f"- 库存周期解读: {cycle['inventory_cycle']}",
                f"- 资本开支周期: {cycle['capex_cycle']}",
                f"- 盈利周期: {cycle['earnings_cycle']}",
                f"- **周期综合评分**: {cycle['composite_score']}/100",
                "",
                "**周期定位规则**：",
                "- 复苏期: 盈利拐点向上 + 毛利率改善 + 营收加速 → 最佳买入时机",
                "- 繁荣期: 盈利持续上升 + 毛利率高位 + 营收强劲 → 持有但警惕过热",
                "- 衰退期: 盈利下滑 + 毛利率恶化 + 营收减速 → 回避",
                "- 萧条期: 盈利触底 + 毛利率低位企稳 + 营收萎缩但降速 → 左侧布局",
                "",
                "**行业景气度与宏观传导**：",
                "- 行业处于复苏期/繁荣期 + 宏观宽松 → 双重利好，权重上调",
                "- 行业处于衰退期 + 宏观收紧 → 双重利空，权重下调",
                "- 行业周期与宏观周期背离 → 关注独立逻辑（如政策扶持、技术突破）",
                "- 周期评分 > 70 → 行业景气度支持买入",
                "- 周期评分 < 30 → 行业景气度不支持买入",
                "",
            ])

        # ── 4. 政策敏感性分析（v2.2 新增） ──
        policy_impact = self._analyze_policy_impact(snapshot)
        sensitivity = policy_impact.get("sensitivity", {})
        dim_scores = policy_impact.get("dimension_scores", {})
        dim_names = {
            "monetary": "货币政策", "fiscal": "财政政策",
            "regulatory": "监管政策", "trade": "贸易政策", "industrial": "产业政策",
        }
        sens_level = lambda v: "极高" if v >= 0.8 else ("高" if v >= 0.6 else ("中等" if v >= 0.4 else "低"))

        parts.extend([
            "### 四、政策敏感性分析",
            f"- **行业**: {industry if industry else '未知'}",
            "",
            "**该行业对各类型政策的敏感度**：",
        ])
        for dim, sens in sensitivity.items():
            parts.append(f"- {dim_names.get(dim, dim)}敏感度: {sens:.2f}（{sens_level(sens)}）")
        parts.append("")

        # 各维度政策影响得分
        has_policy_signal = any(v != 0 for v in policy_impact.get("policy_environment", {}).values())
        if has_policy_signal:
            parts.extend(["**当前政策环境影响评估**："])
            for dim, score in dim_scores.items():
                dir_text = "利好" if score > 0 else ("利空" if score < 0 else "中性")
                strength = "强" if abs(score) >= 0.6 else ("中等" if abs(score) >= 0.3 else "弱")
                if score != 0:
                    parts.append(f"- {dim_names.get(dim, dim)}: {dir_text}{strength}影响（得分{score:+.2f}）")
            parts.append("")

        parts.extend([
            f"- **综合政策影响得分**: {policy_impact['impact_score']:+.2f}（{policy_impact['impact_direction']}）",
            f"- **政策评分**: {policy_impact['policy_score']:.0f}/100",
            "",
            "**政策传导预期**：",
        ])
        for tx in policy_impact.get("transmission_analysis", [])[:3]:
            parts.append(f"- {tx}")
        parts.append("")

        risk_radar = policy_impact.get("risk_radar", {})
        if risk_radar and "warning" in risk_radar:
            parts.extend([
                "**政策风险雷达**：",
                f"- {risk_radar['warning']}",
                "",
            ])

        parts.extend([
            "**政策敏感度纳入量化对冲规则**：",
            "- 综合政策影响得分 ≥ +2.0 → macro_score +2（强利好）",
            "- 综合政策影响得分 ≥ +1.0 → macro_score +1（弱利好）",
            "- 综合政策影响得分 ≤ -2.0 → macro_score -2（强利空）",
            "- 综合政策影响得分 ≤ -1.0 → macro_score -1（弱利空）",
            "- 政策风险雷达中的维度若发生方向变化，需重新评估",
            "",
        ])

        # ── 5. 宏观数据 ──
        has_macro = any(v not in (None, "", "未知", "N/A") for v in macro.values() if isinstance(v, (str, int, float)))
        if has_macro:
            parts.extend(["### 五、宏观数据"])
            bond = macro.get("bond_yield_10y")
            if bond is not None:
                parts.append(f"- 10Y国债收益率: {bond:.2f}%")
            bond_trend = macro.get("bond_yield_trend")
            if bond_trend and bond_trend != "未知":
                parts.append(f"- 国债收益率趋势: {bond_trend} {'(利率上行利空成长)' if bond_trend == '上行' else '(利率下行利好成长)'}")

            pmi = macro.get("pmi")
            if pmi is not None:
                pmi_desc = "扩张" if pmi > 50 else ("收缩" if pmi < 50 else "临界")
                parts.append(f"- PMI: {pmi:.1f} ({pmi_desc})")

            pmi_trend = macro.get("pmi_trend")
            if pmi_trend and pmi_trend != "未知":
                parts.append(f"- PMI趋势: {pmi_trend}")

            rmb = macro.get("rmb_trend")
            if rmb and rmb != "未知":
                parts.append(f"- 人民币汇率: {rmb}")

            policy = macro.get("policy_stance")
            if policy and policy != "未知":
                parts.append(f"- 国内政策: {policy}")

            fed = macro.get("fed_policy")
            if fed and fed != "未知":
                parts.append(f"- 美联储政策: {fed}")
            parts.append("")
        else:
            parts.extend([
                "### 五、宏观数据",
                "- 宏观指标数据暂不可用",
                "- **请基于大盘指数趋势和板块资金流向推断宏观环境**",
                "",
            ])

        # ── 6. 全球经济联动分析（v2.3 新增） ──
        global_econ = self._analyze_global_economy(snapshot)
        fx = global_econ.get("fx_analysis", {})
        commodity = global_econ.get("commodity_analysis", {})
        us_map = global_econ.get("us_stock_mapping", {})

        has_global_data = bool(fx or commodity or us_map)
        if has_global_data:
            parts.extend([
                "### 六、全球经济联动分析",
                f"- **行业**: {industry if industry else '未知'}",
                "",
            ])

            # 汇率敏感性
            if fx:
                fx_type = fx.get("industry_type", "neutral")
                type_desc = {"export": "出口导向型", "import": "进口依赖型", "mixed": "进出口混合型", "neutral": "中性"}
                parts.extend([
                    "**汇率敏感性分析**：",
                    f"- 行业汇率敏感度类型: {type_desc.get(fx_type, fx_type)}（敏感度{fx.get('sensitivity', 0):.0%}）",
                    f"- 人民币趋势: {fx.get('rmb_trend', '未知')}",
                    f"- 汇率影响得分: {fx.get('impact_score', 0):+.2f}",
                    f"- 传导逻辑: {fx.get('reasoning', '')}",
                    "",
                ])

            # 大宗商品
            if commodity:
                parts.extend([
                    "**大宗商品价格影响**：",
                    f"- 综合大宗商品影响得分: {commodity.get('total_impact', 0):+.2f}",
                ])
                for imp in commodity.get("impacts", []):
                    parts.append(f"- {imp.get('commodity', '')} {imp.get('trend', '')} → 影响{imp.get('impact', 0):+.2f}（{imp.get('reasoning', '')}）")
                parts.append("")

            # 美股映射
            if us_map:
                parts.extend([
                    "**美股映射与联动传导**：",
                    f"- A股{industry} ↔ 美股{us_map.get('us_sector', '')}",
                    f"- 历史相关系数: {us_map.get('correlation', 0):.0%}",
                    f"- 领先滞后关系: {us_map.get('lead_lag', '')}",
                ])
                if us_map.get("us_change_pct") is not None:
                    parts.append(f"- 隔夜美股变动: {us_map.get('us_change_pct', 0):+.2%}")
                parts.append(f"- 美股传导得分: {us_map.get('impact_score', 0):+.2f}")
                parts.append(f"- 传导分析: {us_map.get('reasoning', '')}")
                parts.append("")

            parts.extend([
                f"- **全球经济联动综合得分**: {global_econ['global_score']:+.2f}（{global_econ['global_direction']}）",
                f"- **全球评分**: {global_econ['global_score_100']:.0f}/100",
                "",
                "**全球经济联动纳入量化对冲规则**：",
                "- 汇率+大宗商品+美股综合得分 ≥ +2.0 → macro_score +2（强利好）",
                "- 汇率+大宗商品+美股综合得分 ≥ +1.0 → macro_score +1（弱利好）",
                "- 汇率+大宗商品+美股综合得分 ≤ -2.0 → macro_score -2（强利空）",
                "- 汇率+大宗商品+美股综合得分 ≤ -1.0 → macro_score -1（弱利空）",
                "",
            ])

        # ── 分析指引 ──
        parts.extend([
            "## 分析指引",
            "",
            "1. **宏观→行业→个股传导链条**：",
            "   - 当前宏观环境如何影响该股票所属行业？",
            "   - 该行业处于库存周期哪个位置？",
            "   - 最终传导到这只个股是利好还是利空？",
            "",
            "2. **行业景气度周期定位（必须参考）**：",
            "   - 行业处于复苏/繁荣/衰退/萧条哪个阶段？",
            "   - 盈利、营收、毛利率三维度趋势是否一致？",
            "   - 行业周期与宏观周期是否共振？（共振=信号强化，背离=关注独立逻辑）",
            "   - 周期评分 > 70 → 行业景气度支持配置",
            "   - 周期评分 < 30 → 行业景气度不支持配置",
            "",
            "3. **利好-利空量化对冲（必须执行）**：",
            "   - 每条key_factor标注权重（+2强利好/+1弱利好）",
            "   - 每条risk_flag标注权重（-1弱风险/-2强风险）",
            "   - 计算净得分macro_score，并据此确定macro_signal",
            "   - 行业周期评分纳入macro_score：复苏期(+1~+2)、繁荣期(0~+1)、衰退期(-1~-2)、萧条期(-1~0)",
            "   - 政策敏感度纳入macro_score：综合政策得分≥+2(+2)、≥+1(+1)、≤-2(-2)、≤-1(-1)",
            "   - 全球经济联动纳入macro_score：综合全球得分≥+2(+2)、≥+1(+1)、≤-2(-2)、≤-1(-1)",
            "",
            "4. **风格匹配一致性检查（强制）**：",
            f"   - 该股票行业风格: {stock_style}",
            f"   - 若市场风格与该股票错配，style_alignment必须<0.5",
            "   - style_alignment<0.5时，macro_signal不得为1",
            "",
            "5. **数据不完整时的推断**：",
            "   - PMI缺失 → 通过大盘指数趋势推断经济景气度",
            "   - 政策缺失 → 通过国债收益率和板块流向推断",
            "   - 季度数据缺失 → 通过板块资金流向和行业排名推断行业景气度",
            "   - 全球数据缺失 → 通过汇率趋势和大宗商品价格走势推断",
            '   - 禁止"数据缺失"、"无法判断"等消极表述',
            "",
            "输出严格JSON格式：",
        ])

        return "\n".join(parts)

    def _fallback_opinion(self, snapshot: StockSnapshot) -> AgentOpinion:
        """多层降级分析 v2.1 — 基于任何可用数据做有意义推断，新增行业景气度修正"""
        market = snapshot.market_context or {}
        macro = snapshot.macro_data or {}
        fundamentals = snapshot.fundamentals
        industry = fundamentals.get("industry", "")

        # ── 计算宏观净得分 ──
        score, factors, risks = self._score_macro_from_data(market, macro, industry, snapshot)

        # ── v2.2 新增：政策敏感度分析 ──
        policy_impact = self._analyze_policy_impact(snapshot)
        policy_raw = policy_impact if policy_impact.get("sensitivity") else {}

        # ── v2.3 新增：全球经济联动分析 ──
        global_econ = self._analyze_global_economy(snapshot)
        global_raw = global_econ if any([
            global_econ.get("fx_analysis"),
            global_econ.get("commodity_analysis"),
            global_econ.get("us_stock_mapping"),
        ]) else {}
        if global_raw:
            gs = global_raw.get("global_score", 0.0)
            if gs >= 2.0:
                score += 2
                factors.append(f"全球经济环境强利好（综合得分{gs:+.2f}） (+2)")
            elif gs >= 1.0:
                score += 1
                factors.append(f"全球经济环境偏利好（综合得分{gs:+.2f}） (+1)")
            elif gs <= -2.0:
                score -= 2
                risks.append(f"全球经济环境强利空（综合得分{gs:+.2f}） (-2)")
            elif gs <= -1.0:
                score -= 1
                risks.append(f"全球经济环境偏利空（综合得分{gs:+.2f}） (-1)")

        # ── v2.1 新增：行业景气度周期修正 ──
        cycle_data = self._analyze_industry_cycle(snapshot)
        cycle_raw = cycle_data if cycle_data.get("stage") != "数据不足" else {}
        if cycle_raw:
            cycle_score = cycle_raw.get("composite_score", 50)
            phase = cycle_raw.get("cycle_phase_num", 0)
            cycle_stage = cycle_raw.get("stage", "")

            if phase == 2:  # 复苏期
                score += 1
                factors.append(f"行业处于{cycle_stage}，盈利拐点确认 (+1)")
            elif phase == 3:  # 繁荣期
                if cycle_score >= 80:
                    score += 1
                    factors.append(f"行业处于{cycle_stage}且景气度高 ({cycle_score}分) (+1)")
                else:
                    factors.append(f"行业处于{cycle_stage}，景气度中等 ({cycle_score}分)")
            elif phase == 4:  # 衰退期
                score -= 2
                risks.append(f"行业处于{cycle_stage}，盈利下滑 ({cycle_score}分) (-2)")
            elif phase == 1:  # 萧条期
                score -= 1
                risks.append(f"行业处于{cycle_stage}，盈利底部 ({cycle_score}分) (-1)")
            else:
                if cycle_score >= 65:
                    score += 1
                    factors.append(f"行业周期评分偏正面 ({cycle_score}分) (+1)")
                elif cycle_score <= 35:
                    score -= 1
                    risks.append(f"行业周期评分偏负面 ({cycle_score}分) (-1)")

        # ── 风格匹配 ──
        style_align = self._infer_style_alignment(snapshot)

        # ── 确定信号 ──
        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)

        # 风格不匹配时禁止买入
        if style_align < 0.5 and signal == 1:
            signal = 0

        # ── 确定周期 ──
        cycle = self._infer_market_cycle(snapshot)

        # ── 确定置信度 ──
        confidence = 0.65 if signal in (1, -1) else 0.50
        data_count = len(factors) + len(risks)
        if data_count < 2:
            confidence = max(0.40, confidence - 0.15)

        # ── 构建推理 ──
        reasoning_parts = ["【规则引擎降级分析】"]
        if factors:
            reasoning_parts.append("利好：" + "；".join(factors[:3]) + "。")
        if risks:
            reasoning_parts.append("风险：" + "；".join(risks[:3]) + "。")
        reasoning_parts.append(f"宏观净得分{score}，风格匹配度{style_align:.0%}。")
        if cycle_raw:
            reasoning_parts.append(f"行业周期：{cycle_raw.get('stage')}（评分{cycle_raw.get('composite_score')}分）。")
        if policy_raw:
            reasoning_parts.append(f"政策影响：{policy_raw.get('impact_direction')}（得分{policy_raw.get('impact_score'):+.2f}）。")
        reasoning = " ".join(reasoning_parts)

        # ── 行业展望 ──
        sector_outlook = "利好" if score >= 2 else ("利空" if score <= -2 else "中性")

        raw = {
            "macro_signal": signal,
            "market_cycle": cycle,
            "cycle_confidence": confidence,
            "sector_outlook": sector_outlook,
            "style_alignment": round(style_align, 2),
            "macro_score": score,
            "recommended_weight_adjustment": {
                "TA": 0.0, "FA": 0.0, "CA": 0.0,
                "SA": 0.0, "MA": 0.0, "RA": 0.0,
            },
        }
        if cycle_raw:
            raw["industry_cycle"] = cycle_raw
        if policy_raw:
            raw["policy_impact"] = policy_raw
        if global_raw:
            raw["global_economy"] = global_raw

        return AgentOpinion(
            agent_id=self.agent_id,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            key_factors=factors if factors else ["基于有限数据的规则引擎推断"],
            risk_flags=risks if risks else ["宏观数据有限，推断置信度低"],
            raw_data=raw,
        )

    def _score_macro_from_data(
        self, market: Dict, macro: Dict, industry: str, snapshot: Optional[StockSnapshot] = None
    ) -> Tuple[int, List[str], List[str]]:
        """从可用数据计算宏观净得分、利好因子、风险因子
        
        v2.2 新增：当传入 snapshot 时，使用政策敏感度精细化评分；
        否则保留原有简单逻辑作为降级。
        """
        score = 0
        factors = []
        risks = []

        indices = market.get("indices", {})

        # ── 1. 大盘指数趋势 ──
        if indices:
            # 上证指数20日趋势
            sz = indices.get("上证指数", {})
            sz_pct = sz.get("pct_change", 0)
            if sz_pct > 0.03:
                score += 2
                factors.append(f"上证指数强势上涨({sz_pct:+.2%}) (+2)")
            elif sz_pct > 0:
                score += 1
                factors.append(f"上证指数上涨({sz_pct:+.2%}) (+1)")
            elif sz_pct < -0.03:
                score -= 2
                risks.append(f"上证指数大幅下跌({sz_pct:+.2%}) (-2)")
            elif sz_pct < 0:
                score -= 1
                risks.append(f"上证指数下跌({sz_pct:+.2%}) (-1)")

            # 创业板指趋势（成长风向标）
            cy = indices.get("创业板指", {})
            cy_pct = cy.get("pct_change", 0)
            stock_style = self._get_industry_style(industry)
            if stock_style == "成长":
                if cy_pct > 0.03:
                    score += 1
                    factors.append(f"创业板指上涨({cy_pct:+.2%})，利好成长股 (+1)")
                elif cy_pct < -0.03:
                    score -= 1
                    risks.append(f"创业板指下跌({cy_pct:+.2%})，利空成长股 (-1)")

            # 涨跌家数比
            for name, data in indices.items():
                up = data.get("up_count", 0)
                down = data.get("down_count", 0)
                if up > 0 and down > 0:
                    ratio = up / (up + down)
                    if ratio > 0.6:
                        score += 1
                        factors.append(f"{name}涨跌家数比{ratio:.0%}，情绪偏暖 (+1)")
                        break

        # ── 2. 板块资金流向 ──
        sector_top = market.get("sector_top", [])
        if sector_top and industry:
            industry_in_top = False
            industry_flow = 0
            for s in sector_top[:10]:
                if industry in s.get("name", ""):
                    industry_in_top = True
                    industry_flow = s.get("main_net_inflow", 0)
                    break

            if industry_in_top:
                if industry_flow > 1e8:
                    score += 2
                    factors.append(f"所属行业资金大幅流入({industry_flow/1e8:+.1f}亿) (+2)")
                elif industry_flow > 0:
                    score += 1
                    factors.append(f"所属行业资金流入({industry_flow/1e8:+.1f}亿) (+1)")
                elif industry_flow < -1e8:
                    score -= 2
                    risks.append(f"所属行业资金大幅流出({industry_flow/1e8:+.1f}亿) (-2)")
                elif industry_flow < 0:
                    score -= 1
                    risks.append(f"所属行业资金流出({industry_flow/1e8:+.1f}亿) (-1)")
            else:
                # 行业未进入前十，中性偏弱
                risks.append(f"所属行业未进入资金流入前十 (-1)")
                score -= 1

        # ── 3. 宏观数据 ──
        pmi = macro.get("pmi")
        if pmi is not None:
            if pmi > 52:
                score += 2
                factors.append(f"PMI{pmi:.1f}，经济扩张强劲 (+2)")
            elif pmi > 50:
                score += 1
                factors.append(f"PMI{pmi:.1f}，经济扩张 (+1)")
            elif pmi > 47:
                score -= 1
                risks.append(f"PMI{pmi:.1f}，经济偏弱 (-1)")
            else:
                score -= 2
                risks.append(f"PMI{pmi:.1f}，经济收缩 (-2)")

        # ── v2.2 政策敏感度精细化评分 ──
        policy_scored = False
        if snapshot is not None:
            try:
                policy_impact = self._analyze_policy_impact(snapshot)
                impact_score = policy_impact.get("impact_score", 0.0)
                if impact_score != 0 or policy_impact.get("policy_environment"):
                    policy_scored = True
                    if impact_score >= 2.0:
                        score += 2
                        factors.append(f"政策环境强利好（综合得分{impact_score:+.2f}） (+2)")
                    elif impact_score >= 1.0:
                        score += 1
                        factors.append(f"政策环境偏利好（综合得分{impact_score:+.2f}） (+1)")
                    elif impact_score <= -2.0:
                        score -= 2
                        risks.append(f"政策环境强利空（综合得分{impact_score:+.2f}） (-2)")
                    elif impact_score <= -1.0:
                        score -= 1
                        risks.append(f"政策环境偏利空（综合得分{impact_score:+.2f}） (-1)")
                    # 风险雷达
                    risk_radar = policy_impact.get("risk_radar", {})
                    if risk_radar and "dimension" in risk_radar:
                        risks.append(f"政策风险：{risk_radar['warning']}")
            except Exception:
                pass  # 降级到简单逻辑

        # 简单逻辑降级（当精细化评分不可用时）
        if not policy_scored:
            policy = macro.get("policy_stance")
            if policy and policy != "未知":
                if "宽松" in policy:
                    score += 2
                    factors.append(f"政策宽松 (+2)")
                elif "收紧" in policy:
                    score -= 2
                    risks.append(f"政策收紧 (-2)")
                elif "中性" in policy:
                    pass  # 中性不加分

            fed = macro.get("fed_policy")
            if fed and fed != "未知":
                if "宽松" in fed or "降息" in fed:
                    score += 1
                    factors.append(f"美联储宽松，外资流入利好 (+1)")
                elif "收紧" in fed or "加息" in fed:
                    score -= 1
                    risks.append(f"美联储收紧，外资流出压力 (-1)")

        rmb = macro.get("rmb_trend")
        if rmb and rmb != "未知":
            if "升值" in rmb or "走强" in rmb:
                score += 1
                factors.append(f"人民币升值，外资流入 (+1)")
            elif "贬值" in rmb or "走弱" in rmb:
                score -= 1
                risks.append(f"人民币贬值，外资流出压力 (-1)")

        bond = macro.get("bond_yield_10y")
        bond_trend = macro.get("bond_yield_trend")
        if bond_trend and bond_trend != "未知":
            stock_style = self._get_industry_style(industry)
            if "上行" in bond_trend:
                if stock_style == "成长":
                    score -= 1
                    risks.append(f"利率上行，利空高估值成长股 (-1)")
                elif stock_style == "价值":
                    score += 1
                    factors.append(f"利率上行，利好金融价值股 (+1)")
            elif "下行" in bond_trend:
                if stock_style == "成长":
                    score += 1
                    factors.append(f"利率下行，利好成长股 (+1)")
                elif stock_style == "价值":
                    score -= 1
                    risks.append(f"利率下行，利空价值股 (-1)")

        # ── 4. 风格匹配 ──
        style_align = self._infer_style_alignment_from_raw(market, industry)
        if style_align >= 0.7:
            score += 1
            factors.append(f"行业风格与市场高度匹配 (+1)")
        elif style_align < 0.3:
            score -= 2
            risks.append(f"行业风格与市场严重错配 (-2)")
        elif style_align < 0.5:
            score -= 1
            risks.append(f"行业风格与市场不匹配 (-1)")

        return score, factors, risks

    def _get_industry_style(self, industry: str) -> str:
        """判断行业所属风格"""
        if not industry:
            return "未知"
        for style, industries in self.INDUSTRY_STYLE_MAP.items():
            for ind in industries:
                if ind in industry:
                    return style
        return "混合"

    def _infer_style_alignment(self, snapshot: StockSnapshot) -> float:
        """推断风格匹配度 (0.0 ~ 1.0)"""
        market = snapshot.market_context or {}
        industry = snapshot.fundamentals.get("industry", "")
        return self._infer_style_alignment_from_raw(market, industry)

    def _infer_style_alignment_from_raw(self, market: Dict, industry: str) -> float:
        """从原始数据推断风格匹配度"""
        mstyle = market.get("market_style", "")
        mcstyle = market.get("market_cap_style", "")
        stock_style = self._get_industry_style(industry)

        if stock_style == "未知":
            return 0.5

        # 风格关键词映射
        style_keywords = {
            "成长": ["成长", "科技", "创新", "创业板", "科创"],
            "价值": ["价值", "顺周期", "蓝筹", "大盘", "上证50"],
            "防御": ["防御", "高股息", "红利", "避险"],
        }

        combined_style = f"{mstyle} {mcstyle}"

        # 检查匹配
        matched = False
        for keyword in style_keywords.get(stock_style, []):
            if keyword in combined_style:
                matched = True
                break

        # 反向检查（严重错配）
        mismatch = False
        if stock_style == "成长":
            mismatch = any(k in combined_style for k in ["价值", "顺周期", "高股息"])
        elif stock_style == "价值":
            mismatch = any(k in combined_style for k in ["成长", "科技", "创业板"])
        elif stock_style == "防御":
            mismatch = any(k in combined_style for k in ["成长", "科技", "周期"])

        if matched:
            return 0.75
        if mismatch:
            return 0.25
        return 0.5

    def _infer_market_cycle(self, snapshot: StockSnapshot) -> str:
        """从可用数据推断市场周期"""
        market = snapshot.market_context or {}
        macro = snapshot.macro_data or {}

        indices = market.get("indices", {})
        sz = indices.get("上证指数", {})
        sz_pct = sz.get("pct_change", 0)

        pmi = macro.get("pmi")
        policy = macro.get("policy_stance", "")
        bond_trend = macro.get("bond_yield_trend", "")

        # 简单规则判断
        if pmi is not None:
            if pmi > 55:
                return "过热" if "上行" in bond_trend else "复苏晚期"
            elif pmi > 50:
                return "复苏早期" if "宽松" in policy else "复苏晚期"
            elif pmi > 47:
                return "滞胀" if "上行" in bond_trend else "衰退早期"
            else:
                return "衰退晚期" if "宽松" in policy else "衰退早期"

        # 无PMI时，基于指数趋势推断
        if sz_pct > 0.05:
            return "复苏早期"
        elif sz_pct > 0:
            return "复苏晚期"
        elif sz_pct > -0.05:
            return "滞胀"
        else:
            return "衰退早期"

    def _calculate_macro_score(self, snapshot: StockSnapshot) -> int:
        """计算宏观净得分"""
        market = snapshot.market_context or {}
        macro = snapshot.macro_data or {}
        industry = snapshot.fundamentals.get("industry", "")
        score, _, _ = self._score_macro_from_data(market, macro, industry, snapshot)
        return score

    def _default_prompt(self) -> str:
        return '''你是一位自上而下的宏观资产配置专家。

## 核心能力
1. 判断市场周期阶段（复苏/过热/滞胀/衰退）
2. 评估行业景气度位置
3. **利好-利空量化对冲分析**
4. 风格匹配一致性检查

## 评分规则
- 强利好（+2）：PMI>50且上行、政策明确扶持、行业资金大幅流入、风格高度匹配
- 弱利好（+1）：PMI企稳、政策偏暖、板块排名靠前
- 弱利空（-1）：PMI回落、政策中性偏紧、风格轻度不匹配
- 强利空（-2）：PMI<45、政策收紧、风格严重错配

## 净得分与信号
- 净得分 ≥ +3 → macro_signal=1, confidence≥0.65
- 净得分 ≤ -3 → macro_signal=-1, confidence≥0.65
- 否则 → macro_signal=0, confidence=0.50~0.60

## 约束
- market_cycle 必须从指定枚举中选择
- macro_signal=1 时 confidence≥0.60；macro_signal=-1 时 confidence≥0.60
- style_alignment<0.5 时，macro_signal 不得为 1
- 禁止"数据缺失"、"无法判断"等消极表述
- key_factors 和 risk_flags 必须标注量化权重

## 输出严格JSON格式
{
  "market_cycle": "复苏早期",
  "cycle_confidence": 0.70,
  "sector_outlook": "利好",
  "style_alignment": 0.75,
  "macro_signal": 0,
  "confidence": 0.55,
  "macro_score": 0,
  "reasoning": "详细分析（100-200字，必须包含量化对冲逻辑）",
  "key_factors": ["因素1（+2）", "因素2（+1）"],
  "risk_flags": ["风险1（-2）", "风险2（-1）"],
  "recommended_weight_adjustment": {"TA":0,"FA":0,"CA":0,"SA":0,"MA":0,"RA":0}
}'''
