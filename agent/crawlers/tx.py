"""
MASS 腾讯财经爬虫
基于腾讯财经公开API获取A股真实数据

已验证可用接口:
- https://qt.gtimg.cn/q=sh600000 (个股详细信息)
"""
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from loguru import logger

from agent.crawlers.base import BaseCrawler
from agent.crawlers.utils import safe_float, safe_int


class TxCrawler(BaseCrawler):
    """腾讯财经数据爬虫"""

    name = "tx"
    priority = 80
    domain_key = "tx"
    request_interval = 0.5

    # 腾讯返回字段顺序（v_sh600000="1~名称~代码~当前价~昨收~今开~成交量~..."）
    # 参考: https://qt.gtimg.cn/q=sh600000
    TX_FIELDS = [
        "market", "name", "code", "price", "prev_close", "open",
        "volume", "outer_disc", "inner_disc",
        "buy1_v", "buy1_p", "buy2_v", "buy2_p", "buy3_v", "buy3_p",
        "buy4_v", "buy4_p", "buy5_v", "buy5_p",
        "sell1_v", "sell1_p", "sell2_v", "sell2_p", "sell3_v", "sell3_p",
        "sell4_v", "sell4_p", "sell5_v", "sell5_p",
        "recent_deal", "time",
        "change", "pct_change",
        "high", "low",
        "price_per_volume", "pe_ttm", "unknown1", "pe_static", "pb",
        "market_cap", "market_cap2",
        "turnover", "rise_speed", "rise_5min", "rise_15min",
        "unknown2", "amplitude", "circulating_cap", "circulating_cap2",
        "unknown3", "high_52w", "low_52w",
    ]

    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        handlers = {
            "fundamentals": self._fetch_fundamentals,
            "market_context": self._fetch_market_context,
        }
        handler = handlers.get(data_type)
        if not handler:
            return None
        try:
            return handler(stock_code, **kwargs)
        except Exception as e:
            logger.warning(f"[${self.name}] 获取 {data_type} 失败: {e}")
            return None

    def _get_tx_code(self, stock_code: str) -> str:
        code = re.sub(r'[^\d]', '', stock_code)
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"

    def _fetch_fundamentals(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """获取腾讯个股详细信息"""
        tx_code = self._get_tx_code(stock_code)
        url = f"https://qt.gtimg.cn/q={tx_code}"

        try:
            text = self._request_text(url, encoding="gbk")
            if text is None:
                return None

            match = re.search(rf'v_{tx_code}="([^"]*)"', text)
            if not match:
                return None

            values = match.group(1).split("~")
            if len(values) < 10:
                return None

            name = values[1] if len(values) > 1 else ""
            if not name:
                return None

            # 解析关键字段
            result = {
                "stock_code": re.sub(r'[^\d]', '', stock_code),
                "company_name": name,
                "latest_price": safe_float(values[3], default=0.0) if len(values) > 3 else 0,
                "prev_close": safe_float(values[4], default=0.0) if len(values) > 4 else 0,
                "open": safe_float(values[5], default=0.0) if len(values) > 5 else 0,
                "volume": safe_int(values[6], default=0) if len(values) > 6 else 0,
                "high": safe_float(values[33], default=0.0) if len(values) > 33 else 0,
                "low": safe_float(values[34], default=0.0) if len(values) > 34 else 0,
                "change": safe_float(values[31], default=0.0) if len(values) > 31 else 0,
                "pct_change": safe_float(values[32], default=0.0) if len(values) > 32 else 0,
                "pe_ttm": safe_float(values[39]) if len(values) > 39 else None,
                "pe_static": safe_float(values[41]) if len(values) > 41 else None,
                "pb": safe_float(values[42]) if len(values) > 42 else None,
                "market_cap": safe_float(values[43], default=0.0) if len(values) > 43 else 0,
                "turnover": safe_float(values[45], default=0.0) if len(values) > 45 else 0,
                "amplitude": safe_float(values[50], default=0.0) if len(values) > 50 else 0,
                "high_52w": safe_float(values[55], default=0.0) if len(values) > 55 else 0,
                "low_52w": safe_float(values[56], default=0.0) if len(values) > 56 else 0,
                "source": self.name,
                "fetch_time": datetime.now().isoformat(),
            }

            # 过滤无效值
            for k in list(result.keys()):
                if result[k] == 0 and k not in ("stock_code", "company_name", "source", "fetch_time"):
                    result[k] = None

            return result
        except Exception as e:
            logger.warning(f"[{self.name}] 解析个股信息失败: {e}")
            return None

    def _fetch_market_context(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """腾讯暂无独立市场环境接口，返回None"""
        return None
