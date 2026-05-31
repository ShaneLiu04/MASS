"""
MASS K线形态识别引擎
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class PatternRecognitionEngine:
    """
    K线形态识别引擎
    
    设计原则：
    1. 规则引擎检测候选形态（基于价格极值点 + 趋势线拟合）
    2. 可靠性引擎对候选形态打分（5维度）
    3. 只返回 reliability >= 2 的可靠形态（过滤噪声）
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.high = df["high"].values
        self.low = df["low"].values
        self.close = df["close"].values
        self.open_p = df["open"].values if "open" in df.columns else self.close
        self.volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        self.n = len(df)
        
        # 预计算极值点（缓存避免重复计算）
        self._peaks_idx = None
        self._valleys_idx = None
        self._peaks = None
        self._valleys = None
    
    @property
    def peaks_idx(self):
        if self._peaks_idx is None:
            from scipy.signal import argrelextrema
            self._peaks_idx = argrelextrema(self.high, np.greater, order=3)[0]
        return self._peaks_idx
    
    @property
    def valleys_idx(self):
        if self._valleys_idx is None:
            from scipy.signal import argrelextrema
            self._valleys_idx = argrelextrema(self.low, np.less, order=3)[0]
        return self._valleys_idx
    
    def detect_all(self) -> List[Dict]:
        """检测所有形态并评分"""
        candidates = []
        
        # ── 反转形态 ──
        candidates.append(self._detect_head_shoulders_top())
        candidates.append(self._detect_head_shoulders_bottom())
        candidates.append(self._detect_double_top())
        candidates.append(self._detect_double_bottom())
        candidates.append(self._detect_rounding_bottom())
        candidates.append(self._detect_rounding_top())
        candidates.append(self._detect_v_reversal())
        
        # ── 持续形态 ──
        candidates.append(self._detect_symmetric_triangle())
        candidates.append(self._detect_ascending_triangle())
        candidates.append(self._detect_descending_triangle())
        candidates.append(self._detect_bull_flag())
        candidates.append(self._detect_bear_flag())
        candidates.append(self._detect_rising_wedge())
        candidates.append(self._detect_falling_wedge())
        candidates.append(self._detect_rectangle())
        candidates.append(self._detect_ascending_channel())
        candidates.append(self._detect_descending_channel())
        
        # 过滤无效结果 + 评分 + 去重
        results = []
        seen_patterns = set()
        for candidate in candidates:
            if candidate is None:
                continue
            
            # 去重：同类型形态只保留最可靠的一个
            pattern_key = candidate["pattern"]
            if pattern_key in seen_patterns:
                continue
            seen_patterns.add(pattern_key)
            
            # 可靠性评分
            scored = self._score_reliability(candidate)
            if scored["reliability"] >= 2:  # 只返回可靠度>=2的形态
                results.append(scored)
        
        # 按可靠性排序
        results.sort(key=lambda x: x["reliability"], reverse=True)
        return results[:5]  # 最多返回5个形态
    
    # ═══════════════════════════════════════════════════════════════════
    # 反转形态检测
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_head_shoulders_top(self) -> Optional[Dict]:
        """头肩顶 — 三个峰值，中间最高，两侧相近"""
        if len(self.peaks_idx) < 3:
            return None
        
        # 扫描最近3个峰值
        for i in range(len(self.peaks_idx) - 1, 1, -1):
            p1, p2, p3 = self.peaks_idx[i-2], self.peaks_idx[i-1], self.peaks_idx[i]
            h1, h2, h3 = self.high[p1], self.high[p2], self.high[p3]
            
            # 中间峰最高，两侧相近(±8%)
            if not (h2 > h1 and h2 > h3):
                continue
            if abs(h1 - h3) / max(h1, h3, 1e-6) > 0.08:
                continue
            
            # 找 neckline 低点
            neckline_candidates = []
            for v in self.valleys_idx:
                if p1 < v < p3:
                    neckline_candidates.append(self.low[v])
            
            if len(neckline_candidates) < 2:
                continue
            
            neckline = max(neckline_candidates)
            formation_days = p3 - p1
            
            # 排除时间过短(少于10天)或过长(超过120天)的形态
            if formation_days < 10 or formation_days > 120:
                continue
            
            target = neckline - (h2 - neckline)
            
            return {
                "pattern": "头肩顶",
                "category": "反转",
                "direction": "看跌",
                "target": round(target, 2),
                "stop_loss": round(h2 * 1.02, 2),
                "neckline": round(neckline, 2),
                "status": "确认" if self.close[-1] < neckline else "形成中",
                "formation_days": int(formation_days),
                "pivot_indices": [int(p1), int(p2), int(p3)],
                "volume_signature": self._analyze_volume_at_pivots([p1, p2, p3]),
            }
        
        return None
    
    def _detect_head_shoulders_bottom(self) -> Optional[Dict]:
        """头肩底 — 三个谷值，中间最低，两侧相近"""
        if len(self.valleys_idx) < 3:
            return None
        
        for i in range(len(self.valleys_idx) - 1, 1, -1):
            v1, v2, v3 = self.valleys_idx[i-2], self.valleys_idx[i-1], self.valleys_idx[i]
            vl1, vl2, vl3 = self.low[v1], self.low[v2], self.low[v3]
            
            if not (vl2 < vl1 and vl2 < vl3):
                continue
            if abs(vl1 - vl3) / max(vl1, vl3, 1e-6) > 0.08:
                continue
            
            neckline_candidates = []
            for p in self.peaks_idx:
                if v1 < p < v3:
                    neckline_candidates.append(self.high[p])
            
            if len(neckline_candidates) < 2:
                continue
            
            neckline = min(neckline_candidates)
            formation_days = v3 - v1
            
            if formation_days < 10 or formation_days > 120:
                continue
            
            target = neckline + (neckline - vl2)
            
            return {
                "pattern": "头肩底",
                "category": "反转",
                "direction": "看涨",
                "target": round(target, 2),
                "stop_loss": round(vl2 * 0.98, 2),
                "neckline": round(neckline, 2),
                "status": "确认" if self.close[-1] > neckline else "形成中",
                "formation_days": int(formation_days),
                "pivot_indices": [int(v1), int(v2), int(v3)],
                "volume_signature": self._analyze_volume_at_pivots([v1, v2, v3]),
            }
        
        return None
    
    def _detect_double_top(self) -> Optional[Dict]:
        """双顶(M头) — 两个相近高点，中间有回调"""
        if len(self.peaks_idx) < 2:
            return None
        
        for i in range(len(self.peaks_idx) - 1, 0, -1):
            p1, p2 = self.peaks_idx[i-1], self.peaks_idx[i]
            h1, h2 = self.high[p1], self.high[p2]
            
            # 两个高点相近(±5%)
            if abs(h1 - h2) / max(h1, h2, 1e-6) > 0.05:
                continue
            
            # 中间有回调（至少有一个谷底在p1和p2之间）
            mid_valley = None
            for v in self.valleys_idx:
                if p1 < v < p2:
                    mid_valley = self.low[v]
                    break
            
            if mid_valley is None:
                continue
            
            # 回调深度应至少为两峰差的15%
            peak_diff = max(h1, h2) - mid_valley
            if peak_diff / max(h1, h2, 1e-6) < 0.15:
                continue
            
            formation_days = p2 - p1
            if formation_days < 5 or formation_days > 60:
                continue
            
            target = mid_valley - peak_diff
            
            return {
                "pattern": "双顶(M头)",
                "category": "反转",
                "direction": "看跌",
                "target": round(target, 2),
                "stop_loss": round(max(h1, h2) * 1.02, 2),
                "neckline": round(mid_valley, 2),
                "status": "确认" if self.close[-1] < mid_valley else "形成中",
                "formation_days": int(formation_days),
                "pivot_indices": [int(p1), int(p2)],
                "volume_signature": self._analyze_volume_at_pivots([p1, p2]),
            }
        
        return None
    
    def _detect_double_bottom(self) -> Optional[Dict]:
        """双底(W底) — 两个相近低点，中间有反弹"""
        if len(self.valleys_idx) < 2:
            return None
        
        for i in range(len(self.valleys_idx) - 1, 0, -1):
            v1, v2 = self.valleys_idx[i-1], self.valleys_idx[i]
            vl1, vl2 = self.low[v1], self.low[v2]
            
            if abs(vl1 - vl2) / max(vl1, vl2, 1e-6) > 0.05:
                continue
            
            mid_peak = None
            for p in self.peaks_idx:
                if v1 < p < v2:
                    mid_peak = self.high[p]
                    break
            
            if mid_peak is None:
                continue
            
            valley_diff = mid_peak - min(vl1, vl2)
            if valley_diff / max(mid_peak, 1e-6) < 0.15:
                continue
            
            formation_days = v2 - v1
            if formation_days < 5 or formation_days > 60:
                continue
            
            target = mid_peak + valley_diff
            
            return {
                "pattern": "双底(W底)",
                "category": "反转",
                "direction": "看涨",
                "target": round(target, 2),
                "stop_loss": round(min(vl1, vl2) * 0.98, 2),
                "neckline": round(mid_peak, 2),
                "status": "确认" if self.close[-1] > mid_peak else "形成中",
                "formation_days": int(formation_days),
                "pivot_indices": [int(v1), int(v2)],
                "volume_signature": self._analyze_volume_at_pivots([v1, v2]),
            }
        
        return None
    
    def _detect_rounding_bottom(self) -> Optional[Dict]:
        """圆弧底 — U型缓慢筑底"""
        if self.n < 40:
            return None
        
        # 取最近40天
        recent_close = self.close[-40:]
        
        # 找最低点位置
        min_idx = np.argmin(recent_close)
        
        # 最低点应在中间区域（不能太靠边缘）
        if min_idx < 10 or min_idx > 30:
            return None
        
        left = recent_close[:min_idx]
        right = recent_close[min_idx:]
        
        # 左侧缓慢下跌，右侧缓慢上涨
        left_trend = np.polyfit(range(len(left)), left, 1)[0]
        right_trend = np.polyfit(range(len(right)), right, 1)[0]
        
        # 左侧下跌，右侧上涨
        if left_trend > -0.01 or right_trend < 0.01:
            return None
        
        # 曲率检查：底部区域应平滑（非尖底）
        bottom = recent_close[max(0, min_idx-5):min(len(recent_close), min_idx+5)]
        if len(bottom) < 5:
            return None
        
        # 底部标准差应小（平缓）
        bottom_std = np.std(bottom)
        price_range = np.max(recent_close) - np.min(recent_close)
        if price_range == 0:
            return None
        
        if bottom_std / price_range > 0.15:
            return None  # 底部太尖，不像圆弧
        
        target = np.max(recent_close) + (np.max(recent_close) - np.min(recent_close)) * 0.5
        
        return {
            "pattern": "圆弧底",
            "category": "反转",
            "direction": "看涨",
            "target": round(target, 2),
            "stop_loss": round(np.min(recent_close) * 0.98, 2),
            "neckline": round(np.max(recent_close), 2),
            "status": "确认" if self.close[-1] > np.max(recent_close) else "形成中",
            "formation_days": 40,
            "volume_signature": self._analyze_volume_rounding(min_idx),
        }
    
    def _detect_rounding_top(self) -> Optional[Dict]:
        """圆弧顶 — 倒U型缓慢筑顶"""
        if self.n < 40:
            return None
        
        recent_close = self.close[-40:]
        max_idx = np.argmax(recent_close)
        
        if max_idx < 10 or max_idx > 30:
            return None
        
        left = recent_close[:max_idx]
        right = recent_close[max_idx:]
        
        left_trend = np.polyfit(range(len(left)), left, 1)[0]
        right_trend = np.polyfit(range(len(right)), right, 1)[0]
        
        if left_trend < 0.01 or right_trend > -0.01:
            return None
        
        top = recent_close[max(0, max_idx-5):min(len(recent_close), max_idx+5)]
        if len(top) < 5:
            return None
        
        top_std = np.std(top)
        price_range = np.max(recent_close) - np.min(recent_close)
        if price_range == 0:
            return None
        
        if top_std / price_range > 0.15:
            return None
        
        target = np.min(recent_close) - (np.max(recent_close) - np.min(recent_close)) * 0.5
        
        return {
            "pattern": "圆弧顶",
            "category": "反转",
            "direction": "看跌",
            "target": round(target, 2),
            "stop_loss": round(np.max(recent_close) * 1.02, 2),
            "neckline": round(np.min(recent_close), 2),
            "status": "确认" if self.close[-1] < np.min(recent_close) else "形成中",
            "formation_days": 40,
            "volume_signature": self._analyze_volume_rounding(max_idx),
        }
    
    def _detect_v_reversal(self) -> Optional[Dict]:
        """V型反转 — 急涨急跌后快速反向"""
        if self.n < 20:
            return None
        
        recent = self.close[-20:]
        
        # 找极值点
        min_idx = np.argmin(recent)
        max_idx = np.argmax(recent)
        
        # V型底：先急跌后急涨，最低点在中间
        if min_idx > 3 and min_idx < 17:
            left_drop = (recent[min_idx] - recent[0]) / recent[0] if recent[0] != 0 else 0
            right_rise = (recent[-1] - recent[min_idx]) / recent[min_idx] if recent[min_idx] != 0 else 0
            
            # 左侧跌幅 > 10%，右侧涨幅 > 10%
            if left_drop < -0.10 and right_rise > 0.10:
                # 检查是否快速反转（V的右侧应陡峭）
                rise_days = 20 - min_idx
                if rise_days >= 3:
                    steepness = right_rise / rise_days
                    if steepness > 0.015:  # 每天涨1.5%以上算陡峭
                        target = recent[-1] + (recent[-1] - recent[min_idx]) * 0.618
                        return {
                            "pattern": "V型底",
                            "category": "反转",
                            "direction": "看涨",
                            "target": round(target, 2),
                            "stop_loss": round(recent[min_idx] * 0.98, 2),
                            "neckline": round(recent[0], 2),
                            "status": "确认" if recent[-1] > recent[0] else "形成中",
                            "formation_days": 20,
                        }
        
        # V型顶：先急涨后急跌，最高点在中间
        if max_idx > 3 and max_idx < 17:
            left_rise = (recent[max_idx] - recent[0]) / recent[0] if recent[0] != 0 else 0
            right_drop = (recent[-1] - recent[max_idx]) / recent[max_idx] if recent[max_idx] != 0 else 0
            
            if left_rise > 0.10 and right_drop < -0.10:
                drop_days = 20 - max_idx
                if drop_days >= 3:
                    steepness = abs(right_drop) / drop_days
                    if steepness > 0.015:
                        target = recent[-1] - (recent[max_idx] - recent[-1]) * 0.618
                        return {
                            "pattern": "V型顶",
                            "category": "反转",
                            "direction": "看跌",
                            "target": round(target, 2),
                            "stop_loss": round(recent[max_idx] * 1.02, 2),
                            "neckline": round(recent[0], 2),
                            "status": "确认" if recent[-1] < recent[0] else "形成中",
                            "formation_days": 20,
                        }
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════
    # 持续形态检测
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_symmetric_triangle(self) -> Optional[Dict]:
        """对称三角形 — 高点下降，低点上升，振幅收敛"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        high_trend = np.polyfit(range(25), h, 1)[0]
        low_trend = np.polyfit(range(25), l, 1)[0]
        
        # 高点下降 + 低点上升
        if high_trend < -0.005 and low_trend > 0.005:
            amp_start = np.mean(h[:5] - l[:5])
            amp_end = np.mean(h[-5:] - l[-5:])
            
            # 振幅收敛至少30%
            if amp_end < amp_start * 0.7 and amp_start > 0:
                apex = (high_trend * 24 + h[0] + low_trend * 24 + l[0]) / 2
                target = apex + (h[0] - l[0])  # 向上突破目标
                
                return {
                    "pattern": "对称三角形",
                    "category": "持续",
                    "direction": "待突破",
                    "target": round(target, 2),
                    "stop_loss": round(l[-1] * 0.98, 2),
                    "neckline": round(apex, 2),
                    "status": "整理中",
                    "formation_days": 25,
                }
        
        return None
    
    def _detect_ascending_triangle(self) -> Optional[Dict]:
        """上升三角形 — 高点平稳，低点上升"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        high_trend = np.polyfit(range(25), h, 1)[0]
        low_trend = np.polyfit(range(25), l, 1)[0]
        
        # 高点平稳(|斜率|<0.005) + 低点上升
        if abs(high_trend) < 0.005 and low_trend > 0.008:
            resistance = np.mean(h[-5:])
            target = resistance + (resistance - np.min(l))
            
            return {
                "pattern": "上升三角形",
                "category": "持续",
                "direction": "看涨",
                "target": round(target, 2),
                "stop_loss": round(l[-1] * 0.97, 2),
                "neckline": round(resistance, 2),
                "status": "确认" if self.close[-1] > resistance else "整理中",
                "formation_days": 25,
            }
        
        return None
    
    def _detect_descending_triangle(self) -> Optional[Dict]:
        """下降三角形 — 低点平稳，高点下降"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        high_trend = np.polyfit(range(25), h, 1)[0]
        low_trend = np.polyfit(range(25), l, 1)[0]
        
        if high_trend < -0.008 and abs(low_trend) < 0.005:
            support = np.mean(l[-5:])
            target = support - (np.max(h) - support)
            
            return {
                "pattern": "下降三角形",
                "category": "持续",
                "direction": "看跌",
                "target": round(target, 2),
                "stop_loss": round(h[-1] * 1.03, 2),
                "neckline": round(support, 2),
                "status": "确认" if self.close[-1] < support else "整理中",
                "formation_days": 25,
            }
        
        return None
    
    def _detect_bull_flag(self) -> Optional[Dict]:
        """上涨旗形 — 急涨后小幅回调，通道向下倾斜"""
        if self.n < 30:
            return None
        
        recent = self.close[-30:]
        
        # 前10天急涨
        pole_rise = (recent[10] - recent[0]) / recent[0] if recent[0] != 0 else 0
        if pole_rise < 0.15:  # 涨幅至少15%
            return None
        
        # 后20天回调/整理
        flag_high = self.high[-20:]
        flag_low = self.low[-20:]
        
        # 旗形通道：高点和低点都向下倾斜但幅度小
        h_trend = np.polyfit(range(20), flag_high, 1)[0]
        l_trend = np.polyfit(range(20), flag_low, 1)[0]
        
        # 向下倾斜但不超过5%
        if h_trend < -0.001 and l_trend < -0.001:
            flag_drop = (recent[-1] - recent[10]) / recent[10] if recent[10] != 0 else 0
            
            # 回调幅度不超过上涨幅度的50%
            if abs(flag_drop) < pole_rise * 0.5:
                target = recent[-1] + (recent[10] - recent[0])  # 旗杆等长
                
                return {
                    "pattern": "上涨旗形",
                    "category": "持续",
                    "direction": "看涨",
                    "target": round(target, 2),
                    "stop_loss": round(np.min(flag_low) * 0.97, 2),
                    "neckline": round(np.max(flag_high), 2),
                    "status": "整理中",
                    "formation_days": 20,
                    "pole_height_pct": round(pole_rise * 100, 1),
                }
        
        return None
    
    def _detect_bear_flag(self) -> Optional[Dict]:
        """下跌旗形 — 急跌后小幅反弹，通道向上倾斜"""
        if self.n < 30:
            return None
        
        recent = self.close[-30:]
        
        pole_drop = (recent[10] - recent[0]) / recent[0] if recent[0] != 0 else 0
        if pole_drop > -0.15:
            return None
        
        flag_high = self.high[-20:]
        flag_low = self.low[-20:]
        
        h_trend = np.polyfit(range(20), flag_high, 1)[0]
        l_trend = np.polyfit(range(20), flag_low, 1)[0]
        
        if h_trend > 0.001 and l_trend > 0.001:
            flag_rise = (recent[-1] - recent[10]) / recent[10] if recent[10] != 0 else 0
            
            if flag_rise < abs(pole_drop) * 0.5:
                target = recent[-1] + (recent[10] - recent[0])  # 等长下跌
                
                return {
                    "pattern": "下跌旗形",
                    "category": "持续",
                    "direction": "看跌",
                    "target": round(target, 2),
                    "stop_loss": round(np.max(flag_high) * 1.03, 2),
                    "neckline": round(np.min(flag_low), 2),
                    "status": "整理中",
                    "formation_days": 20,
                    "pole_height_pct": round(abs(pole_drop) * 100, 1),
                }
        
        return None
    
    def _detect_rising_wedge(self) -> Optional[Dict]:
        """上升楔形 — 高点和低点都上升，但高点斜率>低点斜率，振幅收敛"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        h_trend = np.polyfit(range(25), h, 1)[0]
        l_trend = np.polyfit(range(25), l, 1)[0]
        
        # 都上升，但高点斜率 > 低点斜率（收敛）
        if h_trend > 0.005 and l_trend > 0.005 and h_trend > l_trend * 1.3:
            amp_start = np.mean(h[:5] - l[:5])
            amp_end = np.mean(h[-5:] - l[-5:])
            
            # 振幅收敛
            if amp_end < amp_start * 0.8 and amp_start > 0:
                # 上升楔形通常看跌（出现在上涨末期）
                target = l[0] - (h[-1] - l[-1])
                
                return {
                    "pattern": "上升楔形",
                    "category": "反转",
                    "direction": "看跌",
                    "target": round(target, 2),
                    "stop_loss": round(h[-1] * 1.02, 2),
                    "neckline": round(l[0], 2),
                    "status": "整理中",
                    "formation_days": 25,
                }
        
        return None
    
    def _detect_falling_wedge(self) -> Optional[Dict]:
        """下降楔形 — 高点和低点都下降，但低点斜率>高点斜率，振幅收敛"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        h_trend = np.polyfit(range(25), h, 1)[0]
        l_trend = np.polyfit(range(25), l, 1)[0]
        
        if h_trend < -0.005 and l_trend < -0.005 and abs(l_trend) > abs(h_trend) * 1.3:
            amp_start = np.mean(h[:5] - l[:5])
            amp_end = np.mean(h[-5:] - l[-5:])
            
            if amp_end < amp_start * 0.8 and amp_start > 0:
                target = h[0] + (h[0] - l[0])
                
                return {
                    "pattern": "下降楔形",
                    "category": "反转",
                    "direction": "看涨",
                    "target": round(target, 2),
                    "stop_loss": round(l[-1] * 0.98, 2),
                    "neckline": round(h[0], 2),
                    "status": "整理中",
                    "formation_days": 25,
                }
        
        return None
    
    def _detect_rectangle(self) -> Optional[Dict]:
        """矩形整理 — 高点和低点都在水平通道内波动"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        h_trend = np.polyfit(range(25), h, 1)[0]
        l_trend = np.polyfit(range(25), l, 1)[0]
        
        # 高点和低点都基本水平
        if abs(h_trend) < 0.003 and abs(l_trend) < 0.003:
            h_range = np.max(h) - np.min(h)
            l_range = np.max(l) - np.min(l)
            
            # 通道宽度应足够（至少3%）
            avg_price = np.mean(self.close[-25:])
            if avg_price == 0:
                return None
            
            channel_width = (np.max(h) - np.min(l)) / avg_price
            if channel_width < 0.03:
                return None
            
            # 至少有3次触及上沿和下沿
            upper_touches = sum(1 for hi in h if hi > np.max(h) * 0.995)
            lower_touches = sum(1 for lo in l if lo < np.min(l) * 1.005)
            
            if upper_touches >= 2 and lower_touches >= 2:
                # 判断突破方向：看最近价格位置
                position = (self.close[-1] - np.min(l)) / (np.max(h) - np.min(l))
                
                if position > 0.7:
                    direction = "看涨"
                    target = np.max(h) + (np.max(h) - np.min(l))
                elif position < 0.3:
                    direction = "看跌"
                    target = np.min(l) - (np.max(h) - np.min(l))
                else:
                    direction = "待突破"
                    target = self.close[-1]
                
                return {
                    "pattern": "矩形整理",
                    "category": "持续",
                    "direction": direction,
                    "target": round(target, 2),
                    "stop_loss": round(np.min(l) * 0.97 if direction == "看涨" else np.max(h) * 1.03, 2),
                    "neckline": round(np.max(h) if direction == "看涨" else np.min(l), 2),
                    "status": "确认" if position > 0.7 or position < 0.3 else "整理中",
                    "formation_days": 25,
                }
        
        return None
    
    def _detect_ascending_channel(self) -> Optional[Dict]:
        """上升通道 — 高点和低点都上升，平行通道"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        h_trend = np.polyfit(range(25), h, 1)[0]
        l_trend = np.polyfit(range(25), l, 1)[0]
        
        # 都上升，且斜率相近（平行）
        if h_trend > 0.005 and l_trend > 0.005:
            slope_ratio = h_trend / l_trend if l_trend != 0 else 1
            if 0.7 < slope_ratio < 1.3:
                # 通道宽度稳定
                channel_widths = h - l
                width_std = np.std(channel_widths)
                width_mean = np.mean(channel_widths)
                
                if width_mean > 0 and width_std / width_mean < 0.3:
                    # 价格在通道内运行
                    upper = h[-1]
                    lower = l[-1]
                    mid = (upper + lower) / 2
                    
                    return {
                        "pattern": "上升通道",
                        "category": "持续",
                        "direction": "看涨",
                        "target": round(upper + (upper - lower) * 0.5, 2),
                        "stop_loss": round(lower * 0.97, 2),
                        "neckline": round(lower, 2),
                        "status": "确认" if self.close[-1] > mid else "靠近下轨",
                        "formation_days": 25,
                    }
        
        return None
    
    def _detect_descending_channel(self) -> Optional[Dict]:
        """下降通道 — 高点和低点都下降，平行通道"""
        if self.n < 25:
            return None
        
        h = self.high[-25:]
        l = self.low[-25:]
        
        h_trend = np.polyfit(range(25), h, 1)[0]
        l_trend = np.polyfit(range(25), l, 1)[0]
        
        if h_trend < -0.005 and l_trend < -0.005:
            slope_ratio = h_trend / l_trend if l_trend != 0 else 1
            if 0.7 < slope_ratio < 1.3:
                channel_widths = h - l
                width_std = np.std(channel_widths)
                width_mean = np.mean(channel_widths)
                
                if width_mean > 0 and width_std / width_mean < 0.3:
                    upper = h[-1]
                    lower = l[-1]
                    mid = (upper + lower) / 2
                    
                    return {
                        "pattern": "下降通道",
                        "category": "持续",
                        "direction": "看跌",
                        "target": round(lower - (upper - lower) * 0.5, 2),
                        "stop_loss": round(upper * 1.03, 2),
                        "neckline": round(upper, 2),
                        "status": "确认" if self.close[-1] < mid else "靠近上轨",
                        "formation_days": 25,
                    }
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════
    # 可靠性评分引擎
    # ═══════════════════════════════════════════════════════════════════
    
    def _score_reliability(self, pattern: Dict) -> Dict:
        """
        5维度可靠性评分引擎
        
        总分 = 结构完整度(0-2) + 成交量配合度(0-1.5) + 突破确认度(0-1.5) + 趋势背景度(0-1) + 时间完整度(0-1)
        满分 = 7分 → 映射到 1-5*
        """
        scores = {
            "structure": self._score_structure(pattern),
            "volume": self._score_volume(pattern),
            "breakout": self._score_breakout(pattern),
            "context": self._score_context(pattern),
            "time": self._score_time(pattern),
        }
        
        total = sum(scores.values())
        
        # 映射到 1-5*
        if total >= 5.5:
            stars = 5
        elif total >= 4.0:
            stars = 4
        elif total >= 2.5:
            stars = 3
        elif total >= 1.5:
            stars = 2
        else:
            stars = 1
        
        pattern["reliability"] = stars
        pattern["reliability_detail"] = {
            "structure": round(scores["structure"], 2),
            "volume": round(scores["volume"], 2),
            "breakout": round(scores["breakout"], 2),
            "context": round(scores["context"], 2),
            "time": round(scores["time"], 2),
            "total": round(total, 2),
            "max": 7.0,
        }
        pattern["confidence_pct"] = round(total / 7.0 * 100, 1)
        
        return pattern
    
    def _score_structure(self, pattern: Dict) -> float:
        """结构完整度评分 (0-2分)"""
        score = 1.0  # 基础分：规则引擎检测到了
        
        # 反转形态检查 neckline
        if pattern["category"] == "反转":
            if pattern.get("neckline") and pattern["neckline"] > 0:
                score += 0.3
            
            # 有明确的极值点索引
            pivots = pattern.get("pivot_indices", [])
            if len(pivots) >= 2:
                score += 0.3
            
            # 形态规模合理（不过大/过小）
            days = pattern.get("formation_days", 0)
            if 10 <= days <= 90:
                score += 0.4
            elif 5 <= days <= 120:
                score += 0.2
        
        # 持续形态检查通道/边界清晰度
        elif pattern["category"] == "持续":
            score += 0.3  # 检测到收敛或平行
            
            # 有明确的方向判断
            if pattern["direction"] != "待突破":
                score += 0.3
            
            days = pattern.get("formation_days", 0)
            if 10 <= days <= 60:
                score += 0.4
        
        return min(2.0, score)
    
    def _score_volume(self, pattern: Dict) -> float:
        """成交量配合度评分 (0-1.5分)"""
        if "volume" not in self.df.columns:
            return 0.5  # 无成交量数据，给基础分
        
        vol_sig = pattern.get("volume_signature", {})
        if not vol_sig:
            return 0.5
        
        score = 0.0
        
        # 反转形态：颈线突破时放量
        if pattern["category"] == "反转":
            if vol_sig.get("breakout_volume_spike", False):
                score += 0.6
            if vol_sig.get("volume_divergence", False):
                score += 0.5  # 量价背离（下跌缩量/上涨放量）
            if vol_sig.get("average_volume_increase", 0) > 0.2:
                score += 0.4
        
        # 持续形态：整理期缩量
        elif pattern["category"] == "持续":
            if vol_sig.get("consolidation_volume_decrease", False):
                score += 0.7
            if vol_sig.get("breakout_volume_spike", False):
                score += 0.5
            if vol_sig.get("average_volume_increase", 0) > 0.1:
                score += 0.3
        
        return min(1.5, max(0.3, score))
    
    def _score_breakout(self, pattern: Dict) -> float:
        """突破确认度评分 (0-1.5分)"""
        status = pattern.get("status", "")
        
        if status == "确认":
            return 1.5  # 已突破，最高分
        elif status == "靠近下轨" or status == "靠近上轨":
            return 0.8  # 接近边界，即将突破
        elif status == "形成中":
            return 0.5  # 还在形成
        elif status == "整理中":
            return 0.3  # 未突破
        
        return 0.3
    
    def _score_context(self, pattern: Dict) -> float:
        """趋势背景度评分 (0-1分)"""
        # 取最近30日趋势
        if self.n < 30:
            return 0.3
        
        recent = self.close[-30:]
        trend = np.polyfit(range(30), recent, 1)[0]
        trend_pct = trend / np.mean(recent) * 100 if np.mean(recent) != 0 else 0
        
        direction = pattern.get("direction", "")
        category = pattern.get("category", "")
        
        score = 0.3  # 基础分
        
        if category == "反转":
            # 反转形态应出现在趋势末端
            # 头肩顶/V型顶应出现在上涨趋势后
            if direction == "看跌" and trend_pct > 0.5:
                score += 0.4  # 上涨后出现顶部反转，合理
            # 头肩底/V型底应出现在下跌趋势后
            elif direction == "看涨" and trend_pct < -0.5:
                score += 0.4  # 下跌后出现底部反转，合理
            else:
                score += 0.1  # 趋势背景不匹配，但不完全排除
        
        elif category == "持续":
            # 持续形态应与当前趋势同向
            if direction == "看涨" and trend_pct > 0.2:
                score += 0.5  # 上涨趋势中的看涨持续形态
            elif direction == "看跌" and trend_pct < -0.2:
                score += 0.5  # 下跌趋势中的看跌持续形态
            elif direction == "待突破":
                score += 0.3  # 方向不明，中性
        
        return min(1.0, score)
    
    def _score_time(self, pattern: Dict) -> float:
        """时间完整度评分 (0-1分)"""
        days = pattern.get("formation_days", 0)
        category = pattern.get("category", "")
        
        # 理想形态形成时间
        ideal_range = {
            "头肩顶": (20, 60),
            "头肩底": (20, 60),
            "双顶(M头)": (10, 40),
            "双底(W底)": (10, 40),
            "圆弧底": (30, 90),
            "圆弧顶": (30, 90),
            "V型底": (5, 20),
            "V型顶": (5, 20),
            "对称三角形": (15, 45),
            "上升三角形": (15, 45),
            "下降三角形": (15, 45),
            "上涨旗形": (5, 20),
            "下跌旗形": (5, 20),
            "上升楔形": (15, 45),
            "下降楔形": (15, 45),
            "矩形整理": (15, 60),
            "上升通道": (20, 90),
            "下降通道": (20, 90),
        }
        
        pattern_name = pattern.get("pattern", "")
        ideal = ideal_range.get(pattern_name, (10, 60))
        
        if ideal[0] <= days <= ideal[1]:
            return 1.0  # 完美
        elif days < ideal[0]:
            # 时间过短
            ratio = days / ideal[0]
            return max(0.2, ratio)
        else:
            # 时间过长
            ratio = ideal[1] / days
            return max(0.2, ratio)
    
    # ═══════════════════════════════════════════════════════════════════
    # 成交量分析辅助方法
    # ═══════════════════════════════════════════════════════════════════
    
    def _analyze_volume_at_pivots(self, pivot_indices: List[int]) -> Dict:
        """分析极值点处的成交量特征"""
        if "volume" not in self.df.columns or len(self.volume) == 0:
            return {}
        
        result = {}
        
        # 极值点成交量 vs 前后5日均量
        vol_at_pivots = []
        vol_before = []
        vol_after = []
        
        for idx in pivot_indices:
            if idx < len(self.volume):
                vol_at_pivots.append(self.volume[idx])
                start = max(0, idx - 5)
                end = min(len(self.volume), idx + 6)
                before = self.volume[start:idx]
                after = self.volume[idx+1:end]
                if len(before) > 0:
                    vol_before.append(np.mean(before))
                if len(after) > 0:
                    vol_after.append(np.mean(after))
        
        if vol_at_pivots and vol_before:
            avg_at = np.mean(vol_at_pivots)
            avg_before = np.mean(vol_before)
            avg_after = np.mean(vol_after) if vol_after else avg_before
            
            result["average_volume_increase"] = round((avg_at - avg_before) / max(avg_before, 1), 2)
            result["volume_at_pivots"] = [round(v, 0) for v in vol_at_pivots]
            
            # 突破时放量（最后一个极值点后成交量放大）
            if avg_after > avg_before * 1.3:
                result["breakout_volume_spike"] = True
            else:
                result["breakout_volume_spike"] = False
            
            # 量价背离检测
            if len(pivot_indices) >= 2:
                first_vol = self.volume[pivot_indices[0]] if pivot_indices[0] < len(self.volume) else 0
                last_vol = self.volume[pivot_indices[-1]] if pivot_indices[-1] < len(self.volume) else 0
                first_price = self.close[pivot_indices[0]]
                last_price = self.close[pivot_indices[-1]]
                
                # 价格创新高但成交量萎缩 = 顶背离
                if last_price > first_price and last_vol < first_vol * 0.8:
                    result["volume_divergence"] = True
                # 价格创新低但成交量萎缩 = 底背离
                elif last_price < first_price and last_vol < first_vol * 0.8:
                    result["volume_divergence"] = True
                else:
                    result["volume_divergence"] = False
        
        return result
    
    def _analyze_volume_rounding(self, pivot_idx: int) -> Dict:
        """分析圆弧形态中的成交量特征"""
        if "volume" not in self.df.columns or len(self.volume) == 0:
            return {}
        
        # 圆弧底：底部区域应缩量，突破时放量
        # 圆弧顶：顶部区域应缩量，破位时放量
        
        result = {}
        
        if pivot_idx >= 5 and pivot_idx < len(self.volume) - 5:
            bottom_vol = np.mean(self.volume[pivot_idx-5:pivot_idx+5])
            before_vol = np.mean(self.volume[max(0, pivot_idx-15):pivot_idx-5])
            after_vol = np.mean(self.volume[pivot_idx+5:min(len(self.volume), pivot_idx+15)])
            
            result["consolidation_volume_decrease"] = bottom_vol < before_vol * 0.8
            result["breakout_volume_spike"] = after_vol > bottom_vol * 1.5
            result["average_volume_increase"] = round((after_vol - before_vol) / max(before_vol, 1), 2)
        
        return result

    # ═══════════════════════════════════════════════════════════════════
    # 多因子技术面评分模型 (用于降级分析)
    # ═══════════════════════════════════════════════════════════════════


