"""
MASS 技术指标计算工具
封装 MA, BOLL, KDJ, MACD, RSI, ATR 等常用指标

v2.0 增强：
- 多时间框架指标计算 (日线/周线/60分钟)
- 支撑压力位矩阵计算
- K线形态识别 (头肩顶/底, 双顶/底, 三角形)
- 多因子技术面评分模型
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


from agent.tools.pattern_engine import PatternRecognitionEngine


class IndicatorTool:
    """技术指标计算工具类"""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算所有技术指标

        Args:
        df: DataFrame with columns [open, high, low, close, volume]

        Returns:
        dict of indicators
        """
        if df is None or df.empty or len(df) < 60:
            return {"error": "数据不足，无法计算指标"}

        result = {}
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # 价格位置
        result["current_price"] = round(close[-1], 2)
        result["high_20d"] = round(np.max(high[-20:]), 2)
        result["low_20d"] = round(np.min(low[-20:]), 2)
        result["high_60d"] = round(np.max(high[-60:]), 2)
        result["low_60d"] = round(np.min(low[-60:]), 2)
        result["position_20d"] = round(
            (close[-1] - result["low_20d"]) / (result["high_20d"] - result["low_20d"] + 1e-6), 2
        )
        result["position_60d"] = round(
            (close[-1] - result["low_60d"]) / (result["high_60d"] - result["low_60d"] + 1e-6), 2
        )

        # 移动平均线
        result["ma5"] = round(np.mean(close[-5:]), 2)
        result["ma10"] = round(np.mean(close[-10:]), 2)
        result["ma20"] = round(np.mean(close[-20:]), 2)
        result["ma60"] = round(np.mean(close[-60:]), 2)
        result["ma5_trend"] = "上升" if close[-1] > result["ma5"] else "下降"
        result["ma20_trend"] = "上升" if result["ma5"] > result["ma20"] else "下降"
        result["ma60_trend"] = "上升" if result["ma20"] > result["ma60"] else "下降"

        # 均线排列
        if result["ma5"] > result["ma20"] > result["ma60"]:
            result["ma_alignment"] = "多头排列"
        elif result["ma5"] < result["ma20"] < result["ma60"]:
            result["ma_alignment"] = "空头排列"
        else:
            result["ma_alignment"] = "缠绕/整理"

        # BOLL (20, 2)
        ma20 = pd.Series(close).rolling(window=20).mean()
        std20 = pd.Series(close).rolling(window=20).std()
        result["boll_upper"] = round((ma20.iloc[-1] + 2 * std20.iloc[-1]), 2)
        result["boll_mid"] = round(ma20.iloc[-1], 2)
        result["boll_lower"] = round((ma20.iloc[-1] - 2 * std20.iloc[-1]), 2)
        result["boll_width"] = round(
            (result["boll_upper"] - result["boll_lower"]) / result["boll_mid"], 3
        )
        result["boll_position"] = round(
            (close[-1] - result["boll_lower"]) / (result["boll_upper"] - result["boll_lower"] + 1e-6), 2
        )

        # KDJ (9, 3, 3)
        rsv = IndicatorTool._compute_rsv(high, low, close, 9)
        k, d, j = IndicatorTool._compute_kdj(rsv, 3, 3)
        result["kdj_k"] = round(k[-1], 2)
        result["kdj_d"] = round(d[-1], 2)
        result["kdj_j"] = round(j[-1], 2)
        result["kdj_status"] = IndicatorTool._kdj_status(k[-1], d[-1], j[-1])
        result["kdj_golden_cross"] = bool(k[-1] > d[-1] and k[-2] <= d[-2])
        result["kdj_death_cross"] = bool(k[-1] < d[-1] and k[-2] >= d[-2])

        # MACD (12, 26, 9)
        dif, dea, macd_hist = IndicatorTool._compute_macd(close, 12, 26, 9)
        result["macd_dif"] = round(dif[-1], 3)
        result["macd_dea"] = round(dea[-1], 3)
        result["macd_hist"] = round(macd_hist[-1], 3)
        result["macd_golden_cross"] = bool(dif[-1] > dea[-1] and dif[-2] <= dea[-2])
        result["macd_death_cross"] = bool(dif[-1] < dea[-1] and dif[-2] >= dea[-2])
        result["macd_above_zero"] = bool(dif[-1] > 0)

        # RSI (14)
        rsi14 = IndicatorTool._compute_rsi(close, 14)
        result["rsi14"] = round(rsi14[-1], 2)
        result["rsi_status"] = IndicatorTool._rsi_status(rsi14[-1])

        # 成交量
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])
        result["volume_ma5"] = round(vol_ma5, 0)
        result["volume_ma20"] = round(vol_ma20, 0)
        result["volume_ratio"] = round(volume[-1] / (vol_ma5 + 1e-6), 2)
        result["volume_trend"] = "放量" if volume[-1] > vol_ma5 * 1.2 else ("缩量" if volume[-1] < vol_ma5 * 0.8 else "正常")

        # ATR (14)
        atr14 = IndicatorTool._compute_atr(high, low, close, 14)
        result["atr14"] = round(atr14[-1], 3)
        result["atr_pct"] = round(atr14[-1] / close[-1] * 100, 2)

        # 波动率
        returns = np.diff(close) / close[:-1]
        result["volatility_20d"] = round(np.std(returns[-20:]) * np.sqrt(252) * 100, 2)
        result["volatility_60d"] = round(np.std(returns[-60:]) * np.sqrt(252) * 100, 2)

        # 最大回撤（近60日）
        cummax = np.maximum.accumulate(close[-60:])
        drawdown = (close[-60:] - cummax) / cummax
        result["max_drawdown_60d"] = round(np.min(drawdown) * 100, 2)

        # 量价配合
        price_change = (close[-1] - close[-2]) / close[-2]
        result["price_volume_divergence"] = IndicatorTool._check_divergence(close, volume)

        return result


    @staticmethod
    def _compute_rsv(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9) -> np.ndarray:
        """RSV 计算 — 预计算 rolling max/min，避免 O(n*k) Python 循环"""
        hh = pd.Series(high).rolling(window=n).max().values
        ll = pd.Series(low).rolling(window=n).min().values
        rng = hh - ll
        rsv = np.full_like(close, np.nan, dtype=float)
        valid = ~np.isnan(hh) & (rng > 1e-6)
        rsv[valid] = (close[valid] - ll[valid]) / rng[valid] * 100
        rsv[~np.isnan(hh) & ~valid] = 50.0
        return rsv


    @staticmethod
    def _compute_kdj(
            rsv: np.ndarray,
            k_period: int = 3,
            d_period: int = 3) -> tuple:
        """KDJ 递推 — 预计算 NaN mask，避免循环内重复 isnan 调用"""
        n = len(rsv)
        k = np.full(n, 50.0)
        d = np.full(n, 50.0)
        nan_mask = np.isnan(rsv)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            return k, d, 3 * k - 2 * d
        start = valid_idx[0]
        k[start] = rsv[start]
        d[start] = rsv[start]
        alpha = 1.0 / 3.0
        beta = 2.0 / 3.0
        for i in range(start + 1, n):
            if nan_mask[i]:
                k[i] = k[i - 1]
                d[i] = d[i - 1]
            else:
                k[i] = beta * k[i - 1] + alpha * rsv[i]
                d[i] = beta * d[i - 1] + alpha * k[i]
        return k, d, 3 * k - 2 * d


    @staticmethod
    def _kdj_status(k: float, d: float, j: float) -> str:
        if j > 100 or k > 80:
            return "超买区"
        elif j < 0 or k < 20:
            return "超卖区"
        elif k > d:
            return "强势区"
        else:
            return "弱势区"


    @staticmethod
    def _compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = IndicatorTool._ema(close, fast)
        ema_slow = IndicatorTool._ema(close, slow)
        dif = ema_fast - ema_slow
        dea = IndicatorTool._ema(dif, signal)
        hist = 2 * (dif - dea)
        return dif, dea, hist


    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema


    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)

        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])

        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan
        return rsi


    @staticmethod
    def _rsi_status(rsi: float) -> str:
        if rsi > 80:
            return "严重超买"
        elif rsi > 70:
            return "超买"
        elif rsi < 20:
            return "严重超卖"
        elif rsi < 30:
            return "超卖"
        elif rsi > 50:
            return "偏强"
        else:
            return "偏弱"


    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.full_like(close, np.nan, dtype=float)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        return atr


    @staticmethod
    def _check_divergence(close: np.ndarray, volume: np.ndarray) -> str:
        """量价背离检测"""
        if len(close) < 10:
            return "数据不足"

        # 近5日价格趋势 vs 成交量趋势
        price_trend = close[-1] > close[-5]
        vol_trend = np.mean(volume[-5:]) > np.mean(volume[-10:-5])

        if price_trend and not vol_trend:
            return "价涨量缩（顶背离风险）"
        elif not price_trend and vol_trend:
            return "价跌量增（恐慌抛售或吸筹）"
        elif price_trend and vol_trend:
            return "量价齐升"
        else:
            return "量价齐跌"


    @staticmethod
    def compute_beta(stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        计算真实 Beta 系数：Cov(Rs, Rm) / Var(Rm)

        Args:
        stock_returns: 股票日收益率序列
        market_returns: 市场指数日收益率序列（如沪深300）

        Returns:
        Beta 值，数据不足返回 1.0
        """
        if len(stock_returns) < 10 or len(market_returns) < 10:
            return 1.0
        # 对齐长度
        min_len = min(len(stock_returns), len(market_returns))
        sr = stock_returns[-min_len:]
        mr = market_returns[-min_len:]
        # 剔除 NaN
        valid = ~(np.isnan(sr) | np.isnan(mr))
        if np.sum(valid) < 10:
            return 1.0
        sr = sr[valid]
        mr = mr[valid]
        covariance = np.cov(sr, mr)[0, 1]
        market_variance = np.var(mr)
        return round(covariance / market_variance, 2) if market_variance > 1e-12 else 1.0


    # ═══════════════════════════════════════════════════════════════════
    # 多时间框架分析 (Multi-Timeframe)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_multi_timeframe(
        daily_df: Optional[pd.DataFrame] = None,
        weekly_df: Optional[pd.DataFrame] = None,
        hourly_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        计算多时间框架技术指标

        Args:
        daily_df: 日线K线 DataFrame
        weekly_df: 周线K线 DataFrame
        hourly_df: 60分钟K线 DataFrame

        Returns:
        {
            "daily": {指标字典},
            "weekly": {指标字典},
            "hourly": {指标字典},
            "alignment": {"score": float, "consistency": str, "dominant_trend": str},
        }
        """
        result = {
            "daily": {},
            "weekly": {},
            "hourly": {},
            "alignment": {},
        }

        # 各周期指标计算
        for tf_name, df in [("daily", daily_df), ("weekly", weekly_df), ("hourly", hourly_df)]:
            if df is not None and not df.empty and len(df) >= 20:
                try:
                    indicators = IndicatorTool.compute_all(df)
                    if "error" not in indicators:
                        # 精简存储：只保留关键指标
                        result[tf_name] = IndicatorTool._extract_key_indicators(indicators)
                except Exception:
                    pass

        # 计算多周期一致性
        result["alignment"] = IndicatorTool._compute_timeframe_alignment(result)

        return result


    @staticmethod
    def _extract_key_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
        """从完整指标字典提取关键字段，减少存储和传输开销"""
        keys = [
            "current_price", "ma5", "ma10", "ma20", "ma60",
            "ma_alignment", "ma5_trend", "ma20_trend", "ma60_trend",
            "macd_dif", "macd_dea", "macd_hist",
            "macd_golden_cross", "macd_death_cross", "macd_above_zero",
            "kdj_k", "kdj_d", "kdj_j", "kdj_status",
            "rsi14", "rsi_status",
            "boll_upper", "boll_mid", "boll_lower", "boll_position",
            "volume_ratio", "volume_trend",
            "atr14", "atr_pct",
            "volatility_20d", "max_drawdown_60d",
            "price_volume_divergence",
        ]
        return {k: indicators.get(k) for k in keys if k in indicators}


    @staticmethod
    def _compute_timeframe_alignment(tf_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算多时间框架信号一致性"""
        signals = {}
        confidences = {}

        for tf in ["weekly", "daily", "hourly"]:
            data = tf_data.get(tf, {})
            if not data:
                continue

            # 信号判断：1=多头, -1=空头, 0=震荡
            signal = 0
            confidence = 0.5

            # 均线排列判断
            ma_align = data.get("ma_alignment", "")
            if ma_align == "多头排列":
                signal = 1
                confidence += 0.2
            elif ma_align == "空头排列":
                signal = -1
                confidence += 0.2

            # MACD 判断
            if data.get("macd_golden_cross"):
                signal = 1 if signal >= 0 else 0
                confidence += 0.1
            elif data.get("macd_death_cross"):
                signal = -1 if signal <= 0 else 0
                confidence += 0.1

            # MACD 零轴位置
            if data.get("macd_above_zero") and signal == 1:
                confidence += 0.1
            elif not data.get("macd_above_zero", True) and signal == -1:
                confidence += 0.1

            # RSI 极端值修正
            rsi = data.get("rsi14", 50)
            if signal == 1 and rsi > 75:
                confidence -= 0.15  # 超买削弱多头信心
            elif signal == -1 and rsi < 25:
                confidence -= 0.15  # 超卖削弱空头信心

            # 趋势方向确认
            ma20_trend = data.get("ma20_trend", "")
            if signal == 1 and ma20_trend == "上升":
                confidence += 0.1
            elif signal == -1 and ma20_trend == "下降":
                confidence += 0.1

            signals[tf] = signal
            confidences[tf] = round(min(0.95, confidence), 2)

        # 一致性计算
        if len(signals) >= 2:
            vals = list(signals.values())
            # 一致性得分：全部同向=1.0，两同一异=0.5，全部不同=0.0
            if all(v == vals[0] for v in vals):
                score = 1.0 if vals[0] != 0 else 0.3
                consistency = "高度一致"
            elif len(set(v for v in vals if v != 0)) == 1:
                score = 0.6
                consistency = "基本一致"
            else:
                score = 0.2
                consistency = "存在分歧"

            # 主导趋势（大周期优先）
            dominant = signals.get("weekly", signals.get("daily", 0))
            dominant_trend = "多头" if dominant == 1 else "空头" if dominant == -1 else "震荡"
        else:
            score = 0.0
            consistency = "数据不足"
            dominant_trend = "未知"

        return {
            "score": round(score, 2),
            "consistency": consistency,
            "dominant_trend": dominant_trend,
            "signals": signals,
            "confidences": confidences,
        }


    # ═══════════════════════════════════════════════════════════════════
    # 支撑压力位矩阵
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, indicators: Optional[Dict] = None) -> Dict[str, Any]:
        """
        计算支撑压力位矩阵

        维度：
        1. 历史高低点
        2. 成交量加权平均 (VWAP)
        3. 均线支撑/压力
        4. 布林带边界
        5. 斐波那契回调位

        Returns:
        {
            "strong_support": float, "weak_support": float,
            "strong_resistance": float, "weak_resistance": float,
            "vwap_support": float, "vwap_resistance": float,
            "fib_levels": {level: price},
            "current_price": float,
            "position_in_range": float,  # 0~1, 当前价在支撑到阻力区间中的位置
        }
        """
        if df is None or df.empty or len(df) < 20:
            return {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        current = close[-1]

        result = {"current_price": round(current, 2)}

        # 1. 历史高低点 (近60日)
        high_60 = np.max(high[-60:])
        low_60 = np.min(low[-60:])
        result["strong_resistance"] = round(high_60, 2)
        result["strong_support"] = round(low_60, 2)

        # 2. 成交量加权平均价 (VWAP) ± 3% 作为动态支撑/压力
        recent = df.tail(20)
        vwap = (recent["close"] * recent["volume"]).sum() / recent["volume"].sum()
        result["vwap_support"] = round(vwap * 0.97, 2)
        result["vwap_resistance"] = round(vwap * 1.03, 2)

        # 3. 均线支撑/压力
        if indicators:
            result["weak_support"] = round(indicators.get("ma60", current * 0.95), 2)
            result["weak_resistance"] = round(indicators.get("ma20", current * 1.05), 2)
        else:
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])
            result["weak_support"] = round(ma60, 2)
            result["weak_resistance"] = round(ma20, 2)

        # 4. 布林带边界
        if indicators:
            result["boll_support"] = indicators.get("boll_lower", current * 0.95)
            result["boll_resistance"] = indicators.get("boll_upper", current * 1.05)

        # 5. 斐波那契回调位 (基于近60日高低点)
        fib_range = high_60 - low_60
        result["fib_levels"] = {
            "0.0%_high": round(high_60, 2),
            "23.6%": round(high_60 - fib_range * 0.236, 2),
            "38.2%": round(high_60 - fib_range * 0.382, 2),
            "50.0%": round(high_60 - fib_range * 0.5, 2),
            "61.8%": round(high_60 - fib_range * 0.618, 2),
            "78.6%": round(high_60 - fib_range * 0.786, 2),
            "100.0%_low": round(low_60, 2),
        }

        # 6. 当前价格区间中的位置
        price_range = high_60 - low_60
        if price_range > 0:
            result["position_in_range"] = round((current - low_60) / price_range, 2)
        else:
            result["position_in_range"] = 0.5

        # 7. 距离各关键价位的百分比
        for key in ["strong_support", "weak_support", "vwap_support",
                    "strong_resistance", "weak_resistance", "vwap_resistance"]:
            if key in result:
                price = result[key]
                if price > 0:
                    pct = (current - price) / price * 100
                    result[f"{key}_pct"] = round(pct, 2)

        return result


    # ═══════════════════════════════════════════════════════════════════
    # K线形态识别引擎 v2.0
    # 支持的形态：头肩顶/底、双顶/底、三角形(上升/下降/对称)、旗形、楔形、
    #           矩形、通道、圆顶/底、单日V形反转、持续三角形、通道
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> List[Dict]:
        """
        检测常见K线形态 — v2.0 增强版

        支持的形态：
        - 反转形态：头肩顶/底、双顶/底、圆顶/底、单日V形反转
        - 持续形态：三角形(上升/下降/对称)、旗形、楔形、矩形、通道

        Returns:
        [{
            "pattern": str,           # 形态名称
            "category": str,          # 反转/持续
            "reliability": int(1-5),  # 综合可靠度评分
            "reliability_detail": {}, # 可靠度分解
            "direction": str,         # 看涨/看跌/中性/突破
            "target": float,          # 目标价
            "stop_loss": float,       # 止损价
            "neckline": float,        # 颈线/关键位
            "status": str,            # 形成中/确认/失败
            "formation_days": int,    # 形态形成天数
            "confidence_pct": float,  # 置信度百分比
        }]
        """
        if df is None or len(df) < 30:
            return []

        engine = PatternRecognitionEngine(df)
        return engine.detect_all()


    @staticmethod
    def compute_ta_score(indicators: Dict[str, Any], support_resistance: Optional[Dict] = None) -> Dict[str, Any]:
        """
        技术面综合评分模型

        总分 100 分，5 个维度：
        - 趋势维度 (30分): 均线排列、MACD信号、MA方向等
        - 动量维度 (20分): RSI、KDJ状态
        - 成交量维度 (15分): 量比、量价配合
        - 波动维度 (15分): ATR%、波动率、最大回撤
        - 结构维度 (20分): 价格位置、支撑阻力距离、形态识别

        Returns:
        {
            "total_score": int, "signal": int, "confidence": float,
            "dimensions": {name: {"score": int, "max": int, "factors": [str]}},
            "key_factors": [str], "risk_flags": [str],
        }
        """
        score = 50  # 中性基准
        dimensions = {}
        factors = []
        risk_flags = []

        # 维度 1. 趋势维度 (30分) 计算
        trend_score = 15
        trend_factors = []

        ma_align = indicators.get("ma_alignment", "")
        if ma_align == "多头排列":
            trend_score += 10
            trend_factors.append("均线多头排列(+10)")
        elif ma_align == "空头排列":
            trend_score -= 10
            trend_factors.append("均线空头排列(-10)")
        else:
            trend_factors.append("均线缠绕(0)")

        if indicators.get("macd_golden_cross"):
            trend_score += 5
            trend_factors.append("MACD金叉(+5)")
        elif indicators.get("macd_death_cross"):
            trend_score -= 5
            trend_factors.append("MACD死叉(-5)")

        ma20_trend = indicators.get("ma20_trend", "")
        if ma20_trend == "上升":
            trend_score += 5
            trend_factors.append("MA20上升(+5)")
        elif ma20_trend == "下降":
            trend_score -= 5
            trend_factors.append("MA20下降(-5)")

        if indicators.get("macd_above_zero"):
            trend_score += 5
            trend_factors.append("MACD零轴上方(+5)")
        else:
            trend_score -= 3
            trend_factors.append("MACD零轴下方(-3)")

        trend_score = max(0, min(30, trend_score))
        score += trend_score - 15
        dimensions["trend"] = {"score": trend_score, "max": 30, "factors": trend_factors}
        factors.extend(trend_factors)

        # 维度 2. 动量维度 (20分) 计算
        momentum_score = 10
        momentum_factors = []

        rsi = indicators.get("rsi14", 50)
        if rsi > 80:
            momentum_score -= 8
            momentum_factors.append(f"RSI严重超买({rsi:.0f})(-8)")
            risk_flags.append("RSI严重超买，注意回调风险")
        elif rsi > 70:
            momentum_score -= 5
            momentum_factors.append(f"RSI超买({rsi:.0f})(-5)")
            risk_flags.append("RSI超买")
        elif rsi < 20:
            momentum_score += 8
            momentum_factors.append(f"RSI严重超卖({rsi:.0f})(+8)")
        elif rsi < 30:
            momentum_score += 5
            momentum_factors.append(f"RSI超卖({rsi:.0f})(+5)")
        elif rsi > 50:
            momentum_score += 2
            momentum_factors.append(f"RSI偏强({rsi:.0f})(+2)")
        else:
            momentum_score -= 2
            momentum_factors.append(f"RSI偏弱({rsi:.0f})(-2)")

        kdj_status = indicators.get("kdj_status", "")
        if kdj_status == "超买区":
            momentum_score -= 5
            momentum_factors.append("KDJ超买区(-5)")
        elif kdj_status == "超卖区":
            momentum_score += 5
            momentum_factors.append("KDJ超卖区(+5)")
        elif kdj_status == "强势区":
            momentum_score += 3
            momentum_factors.append("KDJ强势区(+3)")
        elif kdj_status == "弱势区":
            momentum_score -= 3
            momentum_factors.append("KDJ弱势区(-3)")

        momentum_score = max(0, min(20, momentum_score))
        score += momentum_score - 10
        dimensions["momentum"] = {"score": momentum_score, "max": 20, "factors": momentum_factors}
        factors.extend(momentum_factors)

        # 维度 3. 成交量维度 (15分) 计算
        volume_score = 7
        volume_factors = []

        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            volume_score += 5
            volume_factors.append(f"巨量放量({vol_ratio:.1f}x)(+5)")
        elif vol_ratio > 1.5:
            volume_score += 3
            volume_factors.append(f"明显放量({vol_ratio:.1f}x)(+3)")
        elif vol_ratio < 0.5:
            volume_score -= 5
            volume_factors.append(f"严重缩量({vol_ratio:.1f}x)(-5)")
            risk_flags.append("成交量极度萎缩，注意流动性风险")
        elif vol_ratio < 0.8:
            volume_score -= 2
            volume_factors.append(f"相对缩量({vol_ratio:.1f}x)(-2)")

        pv_div = indicators.get("price_volume_divergence", "")
        if "背离" in pv_div:
            volume_score += 3
            volume_factors.append("量价背离(+3)")
        elif "下跌放量" in pv_div:
            volume_score -= 5
            volume_factors.append("下跌放量警示(-5)")
            risk_flags.append("放量下跌")
        elif "同步" in pv_div or "健康" in pv_div:
            volume_score += 2
            volume_factors.append("量价同步信号(+2)")

        volume_score = max(0, min(15, volume_score))
        score += volume_score - 7
        dimensions["volume"] = {"score": volume_score, "max": 15, "factors": volume_factors}
        factors.extend(volume_factors)

        # 维度 4. 波动维度 (15分) 计算
        volatility_score = 7
        volatility_factors = []

        atr_pct = indicators.get("atr_pct", 2.0)
        if atr_pct > 5:
            volatility_score -= 5
            volatility_factors.append(f"高波动(ATR{atr_pct:.1f}%)(-5)")
            risk_flags.append(f"波动率极高(ATR{atr_pct:.1f}%)，止损需放宽")
        elif atr_pct > 3:
            volatility_score -= 2
            volatility_factors.append(f"较高波动(ATR{atr_pct:.1f}%)(-2)")
        elif atr_pct < 1:
            volatility_score += 3
            volatility_factors.append(f"低波动(ATR{atr_pct:.1f}%)(+3)")

        vol_20d = indicators.get("volatility_20d", 20)
        if vol_20d > 40:
            volatility_score -= 5
            volatility_factors.append(f"年化波动率极高({vol_20d:.0f}%)(-5)")
        elif vol_20d > 25:
            volatility_score -= 2
            volatility_factors.append(f"高波动率({vol_20d:.0f}%)(-2)")
        elif vol_20d < 15:
            volatility_score += 3
            volatility_factors.append(f"低波动率({vol_20d:.0f}%)(+3)")

        boll_width = indicators.get("boll_width", 0.1)
        if boll_width > 0.15:
            volatility_score -= 3
            volatility_factors.append(f"布林带扩张({boll_width:.2f})(-3)")
        elif boll_width < 0.05:
            volatility_score += 3
            volatility_factors.append(f"布林带收窄({boll_width:.2f})(+3)")

        volatility_score = max(0, min(15, volatility_score))
        score += volatility_score - 7
        dimensions["volatility"] = {"score": volatility_score, "max": 15, "factors": volatility_factors}
        factors.extend(volatility_factors)

        # 维度 5. 结构维度 (20分) 计算
        structure_score = 10
        structure_factors = []

        position_20d = indicators.get("position_20d", 0.5)
        if position_20d > 0.9:
            structure_score -= 5
            structure_factors.append(f"接近20日高点({position_20d:.0%})(-5)")
            risk_flags.append("价格接近近期高点")
        elif position_20d < 0.1:
            structure_score += 5
            structure_factors.append(f"接近20日低点({position_20d:.0%})(+5)")
        elif position_20d > 0.7:
            structure_score -= 2
            structure_factors.append(f"偏高位({position_20d:.0%})(-2)")
        elif position_20d < 0.3:
            structure_score += 2
            structure_factors.append(f"偏低位({position_20d:.0%})(+2)")

        position_60d = indicators.get("position_60d", 0.5)
        if position_60d > 0.85:
            structure_score -= 3
            structure_factors.append(f"接近60日高点({position_60d:.0%})(-3)")
        elif position_60d < 0.15:
            structure_score += 3
            structure_factors.append(f"接近60日低点({position_60d:.0%})(+3)")

        # 支撑阻力距离
        if support_resistance and "position_in_range" in support_resistance:
            pos_in_range = support_resistance["position_in_range"]
            if pos_in_range > 0.85:
                structure_score -= 5
                structure_factors.append(f"接近强阻力({pos_in_range:.0%})(-5)")
                risk_flags.append("价格接近强阻力位")
            elif pos_in_range < 0.15:
                structure_score += 5
                structure_factors.append(f"接近强支撑({pos_in_range:.0%})(+5)")
            elif pos_in_range > 0.6:
                structure_score -= 2
                structure_factors.append(f"偏阻力侧({pos_in_range:.0%})(-2)")
            elif pos_in_range < 0.4:
                structure_score += 2
                structure_factors.append(f"偏支撑侧({pos_in_range:.0%})(+2)")

        # 布林带位置
        boll_pos = indicators.get("boll_position", 0.5)
        if boll_pos > 0.9:
            structure_score -= 3
            structure_factors.append(f"布林带上轨上方({boll_pos:.0%})(-3)")
        elif boll_pos < 0.1:
            structure_score += 3
            structure_factors.append(f"布林带下轨下方({boll_pos:.0%})(+3)")

        structure_score = max(0, min(20, structure_score))
        score += structure_score - 10
        dimensions["structure"] = {"score": structure_score, "max": 20, "factors": structure_factors}
        factors.extend(structure_factors)

        # 汇总 总分映射
        total_score = max(0, min(100, score + 50))  # 偏移50映射到0~100

        # 信号判断
        if total_score >= 65:
            signal = 1
        elif total_score <= 35:
            signal = -1
        else:
            signal = 0

        # 置信度：偏离中性越远，置信度越高
        confidence = 0.5 + abs(total_score - 50) / 100
        confidence = min(0.92, confidence)

        # 数据完整性惩罚
        available_dims = sum(1 for d in dimensions.values() if d["score"] > 0)
        if available_dims < 3:
            confidence = max(0.4, confidence - 0.15)
            risk_flags.append("关键指标数据不足，技术面分析置信度降低")

        return {
            "total_score": total_score,
            "signal": signal,
            "confidence": round(confidence, 2),
            "dimensions": dimensions,
            "key_factors": factors,
            "risk_flags": risk_flags if risk_flags else ["无显著风险信号"],
        }


    @staticmethod
    def compute_risk_metrics(df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        风险度量指标 — 基于真实K线数据

        Args:
        df: 日线K线 DataFrame (open, high, low, close, volume, amount)
        index_df: 市场指数K线 DataFrame（如沪深300），用于计算真实Beta
        """
        if df is None or len(df) < 30:
            return {"error": "数据不足"}

        close = df["close"].values
        returns = np.diff(close) / close[:-1]

        result = {}
        result["annual_return"] = round(np.mean(returns) * 252 * 100, 2)
        result["annual_volatility"] = round(np.std(returns) * np.sqrt(252) * 100, 2)
        result["sharpe_ratio"] = round(result["annual_return"] / (result["annual_volatility"] + 1e-6), 2)

        # 最大回撤
        cummax = np.maximum.accumulate(close)
        drawdown = (close - cummax) / cummax
        result["max_drawdown"] = round(np.min(drawdown) * 100, 2)

        # 下行标准差
        downside = returns[returns < 0]
        result["downside_std"] = round(np.std(downside) * np.sqrt(252) * 100, 2) if len(downside) > 0 else 0

        # 胜率
        result["win_rate"] = round(np.sum(returns > 0) / len(returns) * 100, 1)

        # Beta — 基于真实指数数据计算（非线性回归）
        if index_df is not None and len(index_df) >= 30 and "close" in index_df.columns:
            index_close = index_df["close"].values
            index_returns = np.diff(index_close) / index_close[:-1]
            result["beta"] = IndicatorTool.compute_beta(returns, index_returns)
            result["beta_source"] = "真实计算"
        else:
            # 无指数数据时使用行业均值估计（注意：此为估计值）
            result["beta"] = 1.0
            result["beta_source"] = "指数数据不足"

        # 成交额
        result["avg_amount_5d"] = round(np.mean(df["amount"].values[-5:]) / 1e8, 2) if "amount" in df.columns else 0

        return result
