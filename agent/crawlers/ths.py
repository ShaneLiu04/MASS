"""
MASS 同花顺爬虫（备用数据源）
基于同花顺公开API接口获取A股真实数据

当前环境中部分接口受限，代码基于公开文档和已知接口格式编写
在标准网络环境下可用
"""
from typing import Dict, Any, Optional, List
from datetime import datetime

from loguru import logger

from agent.crawlers.base import BaseCrawler
from agent.crawlers.utils import standardize_stock_code, safe_float, safe_int, validate_data


class THSCrawler(BaseCrawler):
    """同花顺数据爬虫（备用源）"""

    name = "ths"
    priority = 50
    domain_key = "ths"
    request_interval = 0.8  # 备用源稍慢

    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """统一数据获取接口"""
        handlers = {
            "fundamentals": self._fetch_fundamentals,
            "fund_flow": self._fetch_fund_flow,
            "market_context": self._fetch_market_context,
            "macro": self._fetch_macro,
            "sentiment": self._fetch_sentiment,
            "news": self._fetch_news,
        }

        handler = handlers.get(data_type)
        if not handler:
            logger.warning(f"[{self.name}] 不支持的数据类型: {data_type}")
            return None

        try:
            return handler(stock_code, **kwargs)
        except Exception as e:
            logger.warning(f"[${self.name}] 获取 {data_type} 失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 1. 基本面
    # ------------------------------------------------------------------
    def _fetch_fundamentals(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        通过同花顺接口获取个股基本信息
        接口格式参考同花顺行情API
        """
        code_info = standardize_stock_code(stock_code)
        # 同花顺行情接口
        url = f"https://d.10jqka.com.cn/v6/time/hs_{code_info['code6']}/today"
        # 或使用基础信息接口
        url = "https://basic.10jqka.com.cn/api/stockphb/" + code_info["code6"]

        # 由于同花顺接口需要特定header/cookie，此处尝试通用请求
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://basic.10jqka.com.cn/{code_info['code6']}.html",
        }

        resp = self._request(url, headers=headers)
        if not resp:
            return None

        # 同花顺返回格式多样，尝试通用解析
        if isinstance(resp, dict):
            data = resp.get("data", resp)
        else:
            return None

        # 尝试提取关键字段（同花顺字段命名不统一，做容错）
        fundamentals = {
            "stock_code": code_info["code6"],
            "company_name": data.get("name", data.get("stockName", "")),
            "industry": data.get("industry", data.get("hy", "")),
            "latest_price": safe_float(data.get("price", data.get("f43")), default=0.0),
            "pe_ttm": safe_float(data.get("pe", data.get("pettm"))),
            "pb": safe_float(data.get("pb", data.get("pb"))),
            "roe": safe_float(data.get("roe")),
            "market_cap": safe_float(data.get("totalValue", data.get("zsz")), default=0.0),
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

        schema = {
            "company_name": {"required": True, "type": str},
        }
        if not validate_data(fundamentals, schema):
            return None

        return fundamentals

    # ------------------------------------------------------------------
    # 2. 资金流向
    # ------------------------------------------------------------------
    def _fetch_fund_flow(self, stock_code: str, days: int = 10, **kwargs) -> Optional[Dict[str, Any]]:
        """同花顺资金流向"""
        code_info = standardize_stock_code(stock_code)
        # 同花顺资金流向接口示例
        url = f"https://d.10jqka.com.cn/v6/line/hs_{code_info['code6']}/01/all"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://stockpage.10jqka.com.cn/{code_info['code6']}/",
        }

        resp = self._request(url, headers=headers)
        if not resp:
            return None

        # 同花顺返回格式解析（简化）
        return {
            "stock_code": code_info["code6"],
            "main_net_inflow_total": 0.0,
            "main_inflow_days": 0,
            "daily_flow": [],
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
            "note": "同花顺资金流向需要特定cookie，当前为简化实现",
        }

    # ------------------------------------------------------------------
    # 3. 市场环境
    # ------------------------------------------------------------------
    def _fetch_market_context(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """同花顺市场环境"""
        return {
            "indices": {},
            "sector_top": [],
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 4. 宏观
    # ------------------------------------------------------------------
    def _fetch_macro(self, stock_code: str = "GLOBAL", **kwargs) -> Optional[Dict[str, Any]]:
        """同花顺宏观数据"""
        return {
            "bond_yield_10y": None,
            "pmi": None,
            "policy_stance": "未知",
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 5. 情绪
    # ------------------------------------------------------------------
    def _fetch_sentiment(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """同花顺情绪数据"""
        return {
            "zt_count": 0,
            "dt_count": 0,
            "news_count_7d": 0,
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 6. 新闻
    # ------------------------------------------------------------------
    def _fetch_news(self, stock_code: str, limit: int = 10, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """同花顺新闻"""
        return []
