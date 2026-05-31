"""
MASS 股票数据工具 — 全真实数据版本
零容忍虚假数据，所有数据必须来自真实数据源

数据源:
- K线: AkshareCrawler (主源) + SinaCrawler (备用)
- 基本面: EastMoneyCrawler + SinaCrawler + TxCrawler (字段级合并)
- 资金流向: EastMoneyCrawler
- 市场环境: EastMoneyCrawler + SinaCrawler
- 宏观: EastMoneyCrawler
- 情绪/新闻: EastMoneyCrawler + SinaCrawler
"""
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from loguru import logger

from agent.crawlers import (
    CrawlerRegistry,
    EastMoneyCrawler,
    THSCrawler,
    SinaCrawler,
    TxCrawler,
    AkshareCrawler,
)
from agent.core.cache import CacheManager


class StockDataTool:
    """
    股票数据获取工具 — 支持直接实例化（测试友好），也可通过 get_instance() 获取全局单例。
    只返回真实数据，绝不编造。
    """

    _instance: Optional["StockDataTool"] = None
    _lock = threading.Lock()

    def __init__(self):
        # 统一使用全局 CacheManager 单例，避免双重缓存
        self._cache = CacheManager.get_instance()

        # 初始化爬虫注册表 — 使用全局单例，与 SentimentTool 共享
        self._registry = CrawlerRegistry.get_instance()
        self._registry.register(AkshareCrawler())
        self._registry.register(EastMoneyCrawler())
        self._registry.register(SinaCrawler())
        self._registry.register(TxCrawler())
        self._registry.register(THSCrawler())

        # 数据质量统计
        self._data_quality_log: List[Dict[str, Any]] = []
    
    @classmethod
    def get_instance(cls) -> "StockDataTool":
        """获取全局单例（向后兼容）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _is_cache_valid(self, key: str) -> bool:
        """委托给 CacheManager：None 表示未命中或已过期"""
        return self._cache.get(key) is not None

    def _cache_set(self, key: str, value: Any) -> None:
        """委托给 CacheManager：真 LRU + O(1) 淘汰"""
        self._cache.set(key, value)

    # ==================================================================
    # 1. K线数据（Akshare + 新浪K线双源）
    # ==================================================================
    # 常见A股指数代码映射
    _INDEX_CODES = {"000300", "000001", "399001", "399006", "000016", "000905", "000688"}

    def get_kline(
        self,
        stock_code: str,
        days: int = 120,
        period: str = "daily",
    ) -> Optional[pd.DataFrame]:
        """获取K线数据 — 真实数据源，失败返回None
        
        支持个股和常见A股指数（沪深300、上证指数等）。
        指数数据使用 akshare 的 index_zh_a_hist 接口，确保与个股接口分离。
        """
        cache_key = f"kline_{stock_code}_{days}_{period}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        is_index = stock_code in self._INDEX_CODES

        # 主源: akshare
        try:
            import akshare as ak
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)

            if is_index:
                # 指数使用专用接口，确保数据真实可靠
                df = ak.index_zh_a_hist(
                    symbol=stock_code, period=period,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )
            else:
                # 个股接口，前复权
                df = ak.stock_zh_a_hist(
                    symbol=stock_code, period=period,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"), adjust="qfq",
                )
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "振幅": "amplitude",
                    "涨跌幅": "pct_change", "涨跌额": "change",
                    "换手率": "turnover",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df = df.tail(days).reset_index(drop=True)
                self._cache_set(cache_key, df)
                return df
        except Exception as e:
            logger.warning(f"akshare获取{'指数' if is_index else '个股'}K线失败 [{stock_code}]: {e}")

        # 备用源: 通过Registry字段级合并获取（尝试所有支持的爬虫）
        # 注: SinaCrawler 支持个股K线，当 akshare 失败时作为备用
        try:
            merged = self._registry.fetch_merge(stock_code, "kline", days=days)
            if merged and isinstance(merged, dict) and "df" in merged:
                df = merged["df"]
                if df is not None and not df.empty:
                    self._cache_set(cache_key, df)
                    return df
        except Exception as e:
            logger.warning(f"Registry K线获取失败: {e}")

        logger.warning(f"无法获取 {stock_code} 的K线数据，所有真实数据源均不可用")
        return None

    # ==================================================================
    # 1.5 多周期K线获取（日线/周线/60分钟）
    # ==================================================================

    def get_kline_multi_timeframe(
        self,
        stock_code: str,
        daily_days: int = 120,
        weekly_weeks: int = 24,
        hourly_days: int = 5,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        获取多时间框架K线数据
        
        Args:
            stock_code: 股票代码
            daily_days: 日线数据天数
            weekly_weeks: 周线数据周数
            hourly_days: 60分钟线数据天数（获取最近N天的60分钟数据）
        
        Returns:
            {"daily": df, "weekly": df, "hourly": df}
        """
        result = {
            "daily": None,
            "weekly": None,
            "hourly": None,
        }

        # 1. 日线 — 复用现有方法
        result["daily"] = self.get_kline(stock_code, days=daily_days, period="daily")

        # 2. 周线
        try:
            result["weekly"] = self._get_kline_weekly(stock_code, weeks=weekly_weeks)
        except Exception as e:
            logger.warning(f"获取周线数据失败 [{stock_code}]: {e}")

        # 3. 60分钟线
        try:
            result["hourly"] = self._get_kline_hourly(stock_code, days=hourly_days)
        except Exception as e:
            logger.warning(f"获取60分钟线数据失败 [{stock_code}]: {e}")

        return result

    def _get_kline_weekly(self, stock_code: str, weeks: int = 24) -> Optional[pd.DataFrame]:
        """获取周线K线"""
        cache_key = f"kline_weekly_{stock_code}_{weeks}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=weeks * 7 * 2)

            df = ak.stock_zh_a_hist(
                symbol=stock_code, period="weekly",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"), adjust="qfq",
            )
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "振幅": "amplitude",
                    "涨跌幅": "pct_change", "涨跌额": "change",
                    "换手率": "turnover",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df = df.tail(weeks).reset_index(drop=True)
                self._cache_set(cache_key, df)
                return df
        except Exception as e:
            logger.warning(f"akshare获取周线失败 [{stock_code}]: {e}")

        return None

    def _get_kline_hourly(self, stock_code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """获取60分钟K线"""
        cache_key = f"kline_hourly_{stock_code}_{days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)

            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code, period="60",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"), adjust="qfq",
            )
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                # 分钟线字段名可能不同，做通用映射
                col_map = {
                    "时间": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "振幅": "amplitude",
                    "涨跌幅": "pct_change", "涨跌额": "change",
                    "均价": "vwap",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                # 确保时间字段存在
                if "date" not in df.columns and "time" in df.columns:
                    df = df.rename(columns={"time": "date"})
                self._cache_set(cache_key, df)
                return df
        except Exception as e:
            logger.warning(f"akshare获取60分钟线失败 [{stock_code}]: {e}")

        return None

    # ==================================================================
    # 2. 基本面数据（字段级合并）
    # ==================================================================
    def get_fundamentals(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取基本面数据 — 字段级合并多源真实数据"""
        cache_key = f"fund_{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._registry.fetch_merge(stock_code, "fundamentals")
        if result is None:
            logger.warning(f"无法获取 {stock_code} 的基本面数据，所有真实数据源均不可用")
            return None

        # 不再补充任何虚假字段！缺失的字段保持为None或不存在
        # 只确保 stock_code 存在
        result["stock_code"] = stock_code

        self._cache_set(cache_key, result)
        return result

    # ==================================================================
    # 3. 资金流向数据
    # ==================================================================
    def get_fund_flow(self, stock_code: str, days: int = 10) -> Optional[Dict[str, Any]]:
        """获取资金流向数据 — 真实数据源，失败返回None"""
        cache_key = f"flow_{stock_code}_{days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._registry.fetch_merge(stock_code, "fund_flow", days=days)
        if result is None:
            logger.warning(f"无法获取 {stock_code} 的资金流向数据，所有真实数据源均不可用")
            return None

        result["stock_code"] = stock_code
        self._cache_set(cache_key, result)
        return result

    # ==================================================================
    # 4. 市场环境
    # ==================================================================
    def get_market_context(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取市场环境 — 真实数据源，失败返回None"""
        cache_key = f"market_{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._registry.fetch_merge(stock_code, "market_context")
        if result is None:
            logger.warning(f"无法获取市场环境数据，所有真实数据源均不可用")
            return None

        self._cache_set(cache_key, result)
        return result

    # ==================================================================
    # 5. 宏观数据
    # ==================================================================
    def get_macro_data(self) -> Optional[Dict[str, Any]]:
        """获取宏观数据 — 真实数据源，失败返回None"""
        cache_key = "macro_global"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._registry.fetch_merge("GLOBAL", "macro")
        if result is None:
            logger.warning(f"无法获取宏观数据，所有真实数据源均不可用")
            return None

        self._cache_set(cache_key, result)
        return result

    # ==================================================================
    # 6. 情绪数据
    # ==================================================================
    def get_sentiment_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取情绪数据 — 真实数据源，失败返回None"""
        cache_key = f"sentiment_{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._registry.fetch_merge(stock_code, "sentiment")
        if result is None:
            logger.warning(f"无法获取 {stock_code} 的情绪数据")
            # 情绪数据不是关键，返回空结构
            return {"stock_code": stock_code, "_meta": {"sources_succeeded": []}}

        self._cache_set(cache_key, result)
        return result

    # ==================================================================
    # 7. 数据质量报告
    # ==================================================================
    def get_data_quality_report(self, stock_code: str) -> Dict[str, Any]:
        """
        生成数据质量报告
        返回各数据类型的获取状态、来源、覆盖率、缺失字段
        """
        report = {
            "stock_code": stock_code,
            "timestamp": datetime.now().isoformat(),
            "data_types": {},
        }

        # 检查各数据类型的质量
        checks = [
            ("fundamentals", self.get_fundamentals),
            ("fund_flow", self.get_fund_flow),
            ("market_context", self.get_market_context),
            ("sentiment", self.get_sentiment_data),
            ("macro", lambda _: self.get_macro_data()),
        ]

        for data_type, getter in checks:
            try:
                data = getter(stock_code)
                if data is None:
                    report["data_types"][data_type] = {
                        "status": "unavailable",
                        "source": "none",
                        "reason": "所有数据源均不可用",
                    }
                elif isinstance(data, dict) and "_meta" in data:
                    meta = data["_meta"]
                    sources = meta.get("sources_succeeded", [])
                    missing = meta.get("missing_fields", [])
                    completeness = meta.get("data_completeness", 0)

                    if completeness >= 0.8:
                        status = "ok"
                    elif completeness >= 0.5:
                        status = "partial"
                    else:
                        status = "insufficient"

                    report["data_types"][data_type] = {
                        "status": status,
                        "source": "+".join(sources) if sources else "none",
                        "completeness": completeness,
                        "missing_fields": missing,
                    }
                else:
                    report["data_types"][data_type] = {
                        "status": "ok",
                        "source": "unknown",
                    }
            except Exception as e:
                report["data_types"][data_type] = {
                    "status": "error",
                    "reason": str(e),
                }

        # K线单独检查
        try:
            kline = self.get_kline(stock_code, days=5)
            if kline is not None and not kline.empty:
                report["data_types"]["kline"] = {
                    "status": "ok",
                    "source": "akshare/sina",
                    "records": len(kline),
                }
            else:
                report["data_types"]["kline"] = {
                    "status": "unavailable",
                    "source": "none",
                    "reason": "无法获取K线数据",
                }
        except Exception as e:
            report["data_types"]["kline"] = {
                "status": "error",
                "reason": str(e),
            }

        return report

    def get_industry_peers(
        self,
        stock_code: str,
        industry: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        获取同行业市值最接近的 peer 公司列表

        通过 akshare 全市场实时数据，按行业过滤后按市值接近度排序。
        结果缓存 5 分钟，避免重复请求。

        Args:
            stock_code: 当前股票代码
            industry: 行业名称（必须与 akshare 的行业分类一致）
            top_n: 返回的 peer 数量

        Returns:
            [{
                "stock_code": str,
                "stock_name": str,
                "market_cap": float,      # 总市值（元）
                "pe_ttm": float | None,   # 动态市盈率
                "pb": float | None,       # 市净率
            }]
        """
        if not industry or not stock_code:
            return []

        cache_key = f"industry_peers:{industry}:{stock_code}:{top_n}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            # 获取全市场实时数据（包含行业和估值）
            df = ak.stock_zh_a_spot_em()

            # 找到当前股票的市值
            current_row = df[df["代码"] == stock_code]
            if current_row.empty:
                return []

            current_cap = float(current_row["总市值"].values[0])

            # 过滤同行业，排除自己
            peers = df[df["所属行业"] == industry].copy()
            peers = peers[peers["代码"] != stock_code]

            if peers.empty:
                return []

            # 按市值接近度排序
            peers["cap_diff"] = (peers["总市值"] - current_cap).abs()
            peers = peers.nsmallest(top_n, "cap_diff")

            result = []
            for _, row in peers.iterrows():
                result.append({
                    "stock_code": str(row["代码"]),
                    "stock_name": str(row["名称"]),
                    "market_cap": self._safe_float(row.get("总市值")),
                    "pe_ttm": self._safe_float(row.get("市盈率-动态")),
                    "pb": self._safe_float(row.get("市净率")),
                })

            self._cache.set(cache_key, result, ttl=300)
            return result

        except Exception as e:
            logger.warning(f"获取行业 peers 失败({stock_code}, {industry}): {e}")
            return []

    def get_consensus_forecast(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取分析师一致预期数据（盈利预测 + 机构评级分布）

        使用 akshare.stock_profit_forecast_em 获取全市场一致预期汇总表，
        过滤出目标股票的数据。数据包含未来 4 年的预测 EPS、
        研报覆盖数、近 6 个月机构评级分布。

        Args:
            stock_code: 6 位股票代码，如 "600519"

        Returns:
            {
                "forecast_eps_y1": float,      # 当年预测 EPS
                "forecast_eps_y2": float,      # 次年预测 EPS
                "forecast_eps_y3": float,      # 第三年预测 EPS
                "forecast_eps_y4": float,      # 第四年预测 EPS
                "research_report_count": int,  # 研报覆盖数
                "rating_buy": int,             # 近6月"买入"评级数
                "rating_add": int,             # 近6月"增持"评级数
                "rating_neutral": int,         # 近6月"中性"评级数
                "rating_reduce": int,          # 近6月"减持"评级数
                "rating_sell": int,            # 近6月"卖出"评级数
                "rating_buy_pct": float,       # 买入+增持占比
                "rating_total": int,           # 总评级数
            }
        """
        if not stock_code:
            return None

        cache_key = f"consensus_forecast:{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            # stock_profit_forecast_em 必须传空字符串获取全市场数据
            df = ak.stock_profit_forecast_em(symbol="")
            if df is None or df.empty:
                return None

            row = df[df["代码"] == stock_code]
            if row.empty:
                return None

            r = row.iloc[0]

            # 提取评级数据
            rating_buy = int(r.get("机构投资评级(近六个月)-买入", 0) or 0)
            rating_add = int(r.get("机构投资评级(近六个月)-增持", 0) or 0)
            rating_neutral = int(r.get("机构投资评级(近六个月)-中性", 0) or 0)
            rating_reduce = int(r.get("机构投资评级(近六个月)-减持", 0) or 0)
            rating_sell = int(r.get("机构投资评级(近六个月)-卖出", 0) or 0)
            rating_total = rating_buy + rating_add + rating_neutral + rating_reduce + rating_sell

            result = {
                "forecast_eps_y1": self._safe_float(r.get("2025预测每股收益")),
                "forecast_eps_y2": self._safe_float(r.get("2026预测每股收益")),
                "forecast_eps_y3": self._safe_float(r.get("2027预测每股收益")),
                "forecast_eps_y4": self._safe_float(r.get("2028预测每股收益")),
                "research_report_count": int(r.get("研报数", 0) or 0),
                "rating_buy": rating_buy,
                "rating_add": rating_add,
                "rating_neutral": rating_neutral,
                "rating_reduce": rating_reduce,
                "rating_sell": rating_sell,
                "rating_buy_pct": round((rating_buy + rating_add) / rating_total, 2) if rating_total > 0 else 0.0,
                "rating_total": rating_total,
            }

            self._cache.set(cache_key, result, ttl=3600)  # 缓存1小时
            return result

        except Exception as e:
            logger.warning(f"获取一致预期数据失败({stock_code}): {e}")
            return None

    # ==================================================================
    # 8. 机构持仓数据
    # ==================================================================
    def get_institute_hold_summary(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取机构持仓季度汇总数据（新浪财经）

        返回指定股票最新季度的机构持仓概况，包括：
        - 持仓机构数量及变化
        - 持股比例及季度增幅
        - 占流通股比例及季度增幅

        Args:
            stock_code: 6 位股票代码

        Returns:
            {
                "quarter": str,           # 报告期，如 "2024Q1"
                "institution_count": int, # 持仓机构数
                "institution_count_change": int,  # 机构数变化
                "hold_ratio": float,      # 持股比例(%)
                "hold_ratio_change": float,       # 持股比例增幅(%)
                "float_ratio": float,     # 占流通股比例(%)
                "float_ratio_change": float,      # 占流通股比例增幅(%)
            }
        """
        if not stock_code:
            return None

        cache_key = f"inst_hold:{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            from datetime import datetime

            # 计算当前可用的最近季度
            now = datetime.now()
            year = now.year
            month = now.month
            # 财报季度: Q1(3月), Q2(6月), Q3(9月), Q4(12月)
            # 数据通常滞后 1-2 个月，所以用上一季度
            if month <= 4:
                quarter = 4
                year -= 1
            elif month <= 7:
                quarter = 1
            elif month <= 10:
                quarter = 2
            else:
                quarter = 3

            symbol = f"{year}{quarter}"
            quarter_label = f"{year}Q{quarter}"

            df = ak.stock_institute_hold(symbol=symbol)
            if df is None or df.empty:
                return None

            row = df[df["证券代码"] == stock_code]
            if row.empty:
                # 尝试上上个季度
                if quarter == 1:
                    symbol = f"{year - 1}4"
                    quarter_label = f"{year - 1}Q4"
                else:
                    symbol = f"{year}{quarter - 1}"
                    quarter_label = f"{year}Q{quarter - 1}"
                df = ak.stock_institute_hold(symbol=symbol)
                if df is None or df.empty:
                    return None
                row = df[df["证券代码"] == stock_code]
                if row.empty:
                    return None

            r = row.iloc[0]
            result = {
                "quarter": quarter_label,
                "institution_count": int(r.get("机构数", 0) or 0),
                "institution_count_change": int(r.get("机构数变化", 0) or 0),
                "hold_ratio": self._safe_float(r.get("持股比例")),
                "hold_ratio_change": self._safe_float(r.get("持股比例增幅")),
                "float_ratio": self._safe_float(r.get("占流通股比例")),
                "float_ratio_change": self._safe_float(r.get("占流通股比例增幅")),
            }

            self._cache.set(cache_key, result, ttl=3600)  # 缓存1小时
            return result

        except Exception as e:
            logger.warning(f"获取机构持仓汇总失败({stock_code}): {e}")
            return None

    def get_fund_holdings(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取基金重仓数据（东方财富）

        返回指定股票被基金持有的概况：
        - 持有基金家数
        - 持股总数及变化
        - 持股市值
        - 持股变动比例

        Args:
            stock_code: 6 位股票代码

        Returns:
            {
                "quarter": str,           # 报告期
                "fund_count": int,        # 持有基金家数
                "hold_shares": float,     # 持股总数
                "hold_market_value": float,       # 持股市值
                "hold_change": str,       # 持股变化方向（增持/减持/不变/新进）
                "hold_change_shares": float,      # 持股变动数值
                "hold_change_pct": float,         # 持股变动比例(%)
            }
        """
        if not stock_code:
            return None

        cache_key = f"fund_hold:{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            from datetime import datetime

            # 计算最近可用的报告期（基金重仓数据用季度末）
            now = datetime.now()
            year = now.year
            month = now.month
            if month <= 4:
                date_str = f"{year - 1}1231"
            elif month <= 7:
                date_str = f"{year}0331"
            elif month <= 10:
                date_str = f"{year}0630"
            else:
                date_str = f"{year}0930"

            df = ak.stock_report_fund_hold(symbol="基金持仓", date=date_str)
            if df is None or df.empty:
                return None

            row = df[df["股票代码"] == stock_code]
            if row.empty:
                return None

            r = row.iloc[0]
            result = {
                "quarter": date_str,
                "fund_count": int(r.get("持有基金家数", 0) or 0),
                "hold_shares": self._safe_float(r.get("持股总数")),
                "hold_market_value": self._safe_float(r.get("持股市值")),
                "hold_change": str(r.get("持股变化", "") or ""),
                "hold_change_shares": self._safe_float(r.get("持股变动数值")),
                "hold_change_pct": self._safe_float(r.get("持股变动比例")),
            }

            self._cache.set(cache_key, result, ttl=3600)
            return result

        except Exception as e:
            logger.warning(f"获取基金重仓数据失败({stock_code}): {e}")
            return None

    # ==================================================================
    # 9. 融资融券详细数据
    # ==================================================================
    def get_margin_detail(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取个股最新融资融券数据

        从交易所融资融券明细中过滤出目标股票，返回：
        - 融资余额、融资买入额
        - 融券余量、融券余额
        - 融资融券余额合计

        Args:
            stock_code: 6 位股票代码

        Returns:
            {
                "stock_code": str,
                "date": str,              # 数据日期
                "margin_buy_amount": float,       # 融资买入额
                "margin_balance": float,          # 融资余额
                "short_volume": float,            # 融券余量
                "short_balance": float,           # 融券余额
                "total_balance": float,           # 融资融券余额
            }
        """
        if not stock_code:
            return None

        cache_key = f"margin:{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            from datetime import datetime, timedelta

            # 尝试最近 5 个交易日
            for i in range(5):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    # 上海股票
                    if stock_code.startswith("6"):
                        df = ak.stock_margin_detail_sse(date=date)
                    else:
                        df = ak.stock_margin_detail_szse(date=date)

                    if df is None or df.empty:
                        continue

                    row = df[df["证券代码"] == stock_code]
                    if not row.empty:
                        r = row.iloc[0]
                        result = {
                            "stock_code": stock_code,
                            "date": date,
                            "margin_buy_amount": self._safe_float(r.get("融资买入额")),
                            "margin_balance": self._safe_float(r.get("融资余额")),
                            "short_volume": self._safe_float(r.get("融券余量")),
                            "short_balance": self._safe_float(r.get("融券余额")),
                            "total_balance": self._safe_float(r.get("融资融券余额")),
                        }
                        self._cache.set(cache_key, result, ttl=300)  # 缓存5分钟
                        return result
                except Exception:
                    continue

            return None

        except Exception as e:
            logger.warning(f"获取融资融券数据失败({stock_code}): {e}")
            return None

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """安全地将值转为 float，处理 NaN/None/异常"""
        if val is None:
            return None
        try:
            if isinstance(val, float) and (val != val):  # NaN check
                return None
            return float(val)
        except (ValueError, TypeError):
            return None
