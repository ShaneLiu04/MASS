"""
MASS 东方财富爬虫
基于东方财富公开API接口获取A股真实数据

已验证接口:
- https://push2.eastmoney.com/api/qt/stock/get  (个股基本信息)
- https://searchapi.eastmoney.com/api/suggest/get (搜索)

其他接口基于akshare源码和公开文档实现，在标准网络环境下可用
"""
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from agent.crawlers.base import BaseCrawler
from agent.crawlers.utils import (
    standardize_stock_code,
    parse_jsonp,
    safe_float,
    safe_int,
    format_amount_yi,
    validate_data,
)


class EastMoneyCrawler(BaseCrawler):
    """东方财富数据爬虫"""

    name = "eastmoney"
    priority = 100
    domain_key = "eastmoney"
    request_interval = 0.6

    # 东方财富字段映射 (f字段 -> 含义)
    # 参考 akshare/stock/stock_info_em.py 及公开文档
    FIELD_MAP = {
        "f43": "latest_price",       # 最新价 (需/100 当 fltt=2)
        "f44": "bid_price",          # 竞买价
        "f45": "ask_price",          # 竞卖价
        "f46": "open",               # 今开
        "f47": "volume",             # 成交量(手)
        "f48": "amount",             # 成交额
        "f50": "volume_ratio",       # 量比
        "f51": "highest",            # 最高
        "f52": "lowest",             # 最低
        "f55": "turnover",           # 换手率
        "f57": "code",               # 股票代码
        "f58": "name",               # 股票名称
        "f60": "prev_close",         # 昨收
        "f84": "total_shares",       # 总股本
        "f85": "float_shares",       # 流通股
        "f116": "market_cap",        # 总市值
        "f117": "float_market_cap",  # 流通市值
        "f127": "industry",          # 行业
        "f162": "pe_ttm",            # 市盈率(动)
        "f163": "pe_static",         # 市盈率(静)
        "f167": "pb",                # 市净率
        "f170": "pct_change",        # 涨跌幅
        "f171": "change_amount",     # 涨跌额
        "f173": "roe",               # ROE
        "f177": "turnover_rate",     # 换手率
        "f183": "total_revenue",     # 总营收
        "f184": "gross_margin",      # 毛利率
        "f185": "net_margin",        # 净利率
        "f187": "revenue_yoy",       # 营收同比
        "f188": "profit_yoy",        # 净利同比
        "f189": "list_date",         # 上市日期
    }

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
    # 1. 个股基本信息 / 基本面
    # ------------------------------------------------------------------
    def _fetch_fundamentals(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取个股基本面数据
        接口: https://push2.eastmoney.com/api/qt/stock/get
        """
        code_info = standardize_stock_code(stock_code)
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        fields = (
            "f43,f44,f45,f46,f47,f48,f50,f51,f52,f55,"
            "f57,f58,f60,f84,f85,f116,f117,f127,"
            "f162,f163,f167,f170,f171,f173,f177,"
            "f183,f184,f185,f187,f188,f189"
        )
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": fields,
            "secid": code_info["secid"],
        }

        resp = self._request(url, params=params)
        if not resp:
            return None

        data = resp.get("data")
        if not data:
            return None

        # 解析字段
        # 注意: fltt=2 时，价格字段(f43/f46/f51/f52/f60/f171)已返回正确小数值
        # 涨跌幅(f170)返回的是原始百分点数值，需除以100
        def _parse_val(v):
            if v is None or v == "-" or v == "":
                return 0.0
            return safe_float(v, default=0.0)

        def _parse_pct(v):
            if v is None or v == "-" or v == "":
                return 0.0
            return safe_float(v, default=0.0) / 100.0

        fundamentals = {
            "stock_code": data.get("f57", code_info["code6"]),
            "company_name": data.get("f58", ""),
            "industry": data.get("f127", ""),
            "latest_price": _parse_val(data.get("f43")),
            "open": _parse_val(data.get("f46")),
            "high": _parse_val(data.get("f51")),
            "low": _parse_val(data.get("f52")),
            "prev_close": _parse_val(data.get("f60")),
            "volume": safe_int(data.get("f47"), default=0),  # 单位: 手
            "amount": safe_float(data.get("f48"), default=0.0),
            "turnover": safe_float(data.get("f55"), default=0.0),
            "volume_ratio": safe_float(data.get("f50"), default=0.0),
            "pct_change": _parse_pct(data.get("f170")),
            "change_amount": _parse_val(data.get("f171")),
            # 估值指标
            "pe_ttm": safe_float(data.get("f162")),
            "pe_static": safe_float(data.get("f163")),
            "pb": safe_float(data.get("f167")),
            "roe": safe_float(data.get("f173")),
            # 规模
            "total_shares": safe_float(data.get("f84"), default=0.0),
            "float_shares": safe_float(data.get("f85"), default=0.0),
            "market_cap": safe_float(data.get("f116"), default=0.0),
            "float_market_cap": safe_float(data.get("f117"), default=0.0),
            # 盈利能力
            "total_revenue": safe_float(data.get("f183"), default=0.0),
            "gross_margin": safe_float(data.get("f184"), default=0.0),
            "net_margin": safe_float(data.get("f185"), default=0.0),
            "revenue_yoy": safe_float(data.get("f187"), default=0.0),
            "profit_yoy": safe_float(data.get("f188"), default=0.0),
            "list_date": str(data.get("f189", "")),
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

        # 数据校验
        schema = {
            "company_name": {"required": True, "type": str},
            "pe_ttm": {"required": False, "type": (int, float), "min": 0},
            "pb": {"required": False, "type": (int, float), "min": 0},
        }
        if not validate_data(fundamentals, schema):
            logger.warning(f"[{self.name}] 基本面数据校验失败")
            return None

        return fundamentals

    # ------------------------------------------------------------------
    # 2. 资金流向
    # ------------------------------------------------------------------
    def _fetch_fund_flow(self, stock_code: str, days: int = 10, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取个股资金流向数据
        接口: https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
        返回字段: 日期,主力净流入,小单净流入,中单净流入,大单净流入,超大单净流入,及占比
        """
        code_info = standardize_stock_code(stock_code)
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0",
            "klt": "101",
            "secid": code_info["secid"],
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
            "_": int(time.time() * 1000),
        }

        resp = self._request(url, params=params)
        if not resp:
            return None

        data = resp.get("data")
        if not data or not data.get("klines"):
            return None

        klines = data["klines"]
        daily_flow = []
        total_main = 0.0
        inflow_days = 0

        for line in klines[-days:]:
            parts = line.split(",")
            if len(parts) < 13:
                continue
            # parts: 日期,主力净额,小单净额,中单净额,大单净额,超大单净额,
            #        主力占比,小单占比,中单占比,大单占比,超大单占比,收盘价,涨跌幅
            date_str = parts[0]
            main_net = safe_float(parts[1], default=0.0)
            retail_net = safe_float(parts[2], default=0.0)  # 小单视为散户
            mid_net = safe_float(parts[3], default=0.0)
            big_net = safe_float(parts[4], default=0.0)
            super_big_net = safe_float(parts[5], default=0.0)
            main_pct = safe_float(parts[6], default=0.0)
            close_price = safe_float(parts[11], default=0.0)
            pct_chg = safe_float(parts[12], default=0.0)

            daily_flow.append({
                "date": date_str,
                "main_net_inflow": round(main_net / 10000, 2),      # 万元
                "retail_net_inflow": round(retail_net / 10000, 2),
                "mid_net_inflow": round(mid_net / 10000, 2),
                "big_net_inflow": round(big_net / 10000, 2),
                "super_big_net_inflow": round(super_big_net / 10000, 2),
                "main_inflow_pct": main_pct,
                "close_price": close_price,
                "pct_change": pct_chg,
            })

            total_main += main_net
            if main_net > 0:
                inflow_days += 1

        if not daily_flow:
            return None

        return {
            "stock_code": code_info["code6"],
            "main_net_inflow_total": round(total_main / 10000, 2),
            "main_inflow_days": inflow_days,
            "daily_flow": daily_flow,
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 3. 市场环境 / 大盘 + 板块
    # ------------------------------------------------------------------
    def _fetch_market_context(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取市场环境数据
        包括大盘指数和板块表现
        """
        indices = self._fetch_indices()
        sector = self._fetch_sector_flow()

        return {
            "indices": indices or {},
            "sector_top": sector or [],
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    def _fetch_indices(self) -> Optional[Dict[str, Any]]:
        """获取主要大盘指数"""
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        # 上证、深证、创业板、科创50
        params = {
            "pn": "1",
            "pz": "10",
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "m:1+s:2,m:0+s:3,m:0+t:80,m:1+s:3",
            "fields": "f1,f2,f3,f4,f12,f14,f104,f105",
        }

        resp = self._request(url, params=params)
        if not resp:
            return None

        data = resp.get("data", {})
        diff = data.get("diff", [])

        indices = {}
        name_map = {
            "000001": "上证指数",
            "399001": "深证成指",
            "399006": "创业板指",
            "000688": "科创50",
            "399005": "中小板指",
            "000300": "沪深300",
        }

        for item in diff:
            code = item.get("f12", "")
            name = name_map.get(code)
            if not name:
                continue
            indices[name] = {
                "close": safe_float(item.get("f2"), default=0.0) / 100.0,
                "change": safe_float(item.get("f4"), default=0.0) / 100.0,
                "pct_change": safe_float(item.get("f3"), default=0.0) / 100.0,
                "up_count": safe_int(item.get("f104"), default=0),
                "down_count": safe_int(item.get("f105"), default=0),
            }

        return indices

    def _fetch_sector_flow(self) -> Optional[List[Dict[str, Any]]]:
        """获取板块资金流向排行"""
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "20",
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f20",
            "fs": "m:90+t:2",
            "fields": "f1,f2,f3,f4,f12,f14,f20,f21,f104,f105,f128,f140",
        }

        resp = self._request(url, params=params)
        if not resp:
            return None

        data = resp.get("data", {})
        diff = data.get("diff", [])

        sectors = []
        for item in diff[:10]:
            sectors.append({
                "code": item.get("f12", ""),
                "name": item.get("f14", ""),
                "close": safe_float(item.get("f2"), default=0.0) / 100.0,
                "pct_change": safe_float(item.get("f3"), default=0.0) / 100.0,
                "main_net_inflow": safe_float(item.get("f20"), default=0.0),
                "up_count": safe_int(item.get("f104"), default=0),
                "down_count": safe_int(item.get("f105"), default=0),
            })

        return sectors

    # ------------------------------------------------------------------
    # 4. 宏观数据
    # ------------------------------------------------------------------
    def _fetch_macro(self, stock_code: str = "GLOBAL", **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取宏观数据
        目前通过东方财富数据中心获取国债收益率等
        由于宏观数据接口较为分散，此处获取可得的指标
        """
        # 10年期国债收益率 (通过东方财富债券接口)
        bond_yield = self._fetch_bond_yield()

        return {
            "bond_yield_10y": bond_yield,
            "bond_yield_trend": "未知",
            "pmi": None,           # 预留，可通过其他接口补充
            "policy_stance": "未知",
            "fed_policy": "未知",
            "rmb_trend": "未知",
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    def _fetch_bond_yield(self) -> Optional[float]:
        """获取国债收益率（简化实现）"""
        # 东方财富债券行情接口
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "5",
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "b:MK0406",
            "fields": "f1,f2,f3,f4,f12,f14",
        }
        resp = self._request(url, params=params)
        if not resp:
            return None
        data = resp.get("data", {})
        diff = data.get("diff", [])
        if diff:
            # 取第一个债券的最新价作为近似
            return safe_float(diff[0].get("f2"), default=0.0) / 100.0
        return None

    # ------------------------------------------------------------------
    # 5. 情绪数据
    # ------------------------------------------------------------------
    def _fetch_sentiment(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取市场情绪数据
        包括涨跌停统计等
        """
        zdt = self._fetch_zdt_stats()
        news_count = 0

        # 尝试获取新闻数量（通过搜索接口）
        try:
            search = self._fetch_news_search(stock_code)
            if search:
                news_count = len(search)
        except Exception:
            pass

        return {
            "zt_count": zdt.get("zt_count", 0),
            "dt_count": zdt.get("dt_count", 0),
            "zb_count": zdt.get("zb_count", 0),
            "news_count_7d": news_count,
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    def _fetch_zdt_stats(self) -> Dict[str, int]:
        """获取涨跌停统计（接口可能已下线，兼容处理）"""
        try:
            url = "https://push2ex.eastmoney.com/getTopicZDT"
            params = {
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "dpt": "wz.ztzt",
            }
            resp = self._request(url, params=params)
            if not resp:
                return {}

            data = resp.get("data", {})
            return {
                "zt_count": safe_int(data.get("ztNum"), default=0),
                "dt_count": safe_int(data.get("dtNum"), default=0),
                "zb_count": safe_int(data.get("zgzNum"), default=0),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # 6. 新闻数据
    # ------------------------------------------------------------------
    def _fetch_news(self, stock_code: str, limit: int = 10, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        获取个股相关新闻
        东方财富新闻接口较为分散，此处使用搜索接口获取基本信息
        实际新闻内容可通过其他方式获取
        """
        return self._fetch_news_search(stock_code, limit)

    def _fetch_news_search(self, stock_code: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """通过搜索接口获取新闻线索"""
        url = "https://searchapi.eastmoney.com/api/suggest/get"
        params = {
            "input": stock_code,
            "type": "14",
            "count": str(limit),
        }
        resp = self._request(url, params=params)
        if not resp:
            return None

        data = resp.get("QuotationCodeTable", {}).get("Data", [])
        news = []
        for item in data:
            news.append({
                "code": item.get("Code", ""),
                "title": item.get("Name", ""),
                "source": "东方财富",
                "time": datetime.now().strftime("%Y-%m-%d"),
            })
        return news
