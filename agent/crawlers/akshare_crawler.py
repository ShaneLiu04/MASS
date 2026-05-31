"""
MASS akshare封装爬虫
将akshare封装为统一Crawler接口

已验证可用接口:
- ak.stock_zh_a_hist (K线数据)
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from agent.crawlers.base import BaseCrawler


class AkshareCrawler(BaseCrawler):
    """akshare封装爬虫"""

    name = "akshare"
    priority = 95  # 优先级较高，因为K线数据稳定
    request_interval = 1.0

    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        handlers = {
            "kline": self._fetch_kline,
        }
        handler = handlers.get(data_type)
        if not handler:
            return None
        try:
            return handler(stock_code, **kwargs)
        except Exception as e:
            logger.warning(f"[${self.name}] 获取 {data_type} 失败: {e}")
            return None

    def _fetch_kline(self, stock_code: str, days: int = 120, period: str = "daily", **kwargs) -> Optional[Dict[str, Any]]:
        """获取akshare K线数据"""
        try:
            import akshare as ak

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)

            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=period,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq",
            )

            if df is None or df.empty:
                return None

            # 标准化列名
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

            return {
                "df": df,
                "records": len(df),
                "source": self.name,
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"[{self.name}] K线获取失败: {e}")
            return None
