"""
MASS 新浪财经爬虫
基于新浪财经公开API获取A股真实数据

已验证可用接口:
- https://hq.sinajs.cn/list=sh600000 (实时行情)
- https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData (K线)
- https://feed.mix.sina.com.cn/api/roll/get (新闻列表)
"""
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
from loguru import logger

from agent.crawlers.base import BaseCrawler
from agent.crawlers.utils import safe_float, safe_int


class SinaCrawler(BaseCrawler):
    """新浪财经数据爬虫"""

    name = "sina"
    priority = 90
    domain_key = "sina"
    request_interval = 0.5

    # 新浪实时行情字段顺序（沪深A股）
    # var hq_str_sh600000="名称,今日开盘价,昨日收盘价,当前价,今日最高价,今日最低价,
    # 竞买价,竞卖价,成交股数,成交金额,买一量,买一价,买二量,买二价,...,买五量,买五价,
    # 卖一量,卖一价,...,卖五量,卖五价,日期,时间"
    HQ_FIELDS = [
        "name", "open", "prev_close", "price", "high", "low",
        "bid_price", "ask_price", "volume", "amount",
        "bid1_volume", "bid1_price", "bid2_volume", "bid2_price",
        "bid3_volume", "bid3_price", "bid4_volume", "bid4_price",
        "bid5_volume", "bid5_price",
        "ask1_volume", "ask1_price", "ask2_volume", "ask2_price",
        "ask3_volume", "ask3_price", "ask4_volume", "ask4_price",
        "ask5_volume", "ask5_price",
        "date", "time",
    ]

    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """统一数据获取接口"""
        handlers = {
            "fundamentals": self._fetch_fundamentals,
            "kline": self._fetch_kline,
            "market_context": self._fetch_market_context,
            "sentiment": self._fetch_sentiment,
            "news": self._fetch_news,
        }

        handler = handlers.get(data_type)
        if not handler:
            return None

        try:
            return handler(stock_code, **kwargs)
        except Exception as e:
            logger.warning(f"[${self.name}] 获取 {data_type} 失败: {e}")
            return None

    def _get_sina_code(self, stock_code: str) -> str:
        """转换为新浪格式: sh600000 / sz000001"""
        code = re.sub(r'[^\d]', '', stock_code)
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"

    # ------------------------------------------------------------------
    # 1. 实时行情（作为基本面补充）
    # ------------------------------------------------------------------
    def _fetch_fundamentals(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """获取新浪实时行情作为基本面补充"""
        sina_code = self._get_sina_code(stock_code)
        # hq.sinajs.cn 返回纯文本而非JSON，直接使用文本解析
        return self._parse_hq_text(stock_code, sina_code)

    def _parse_hq_text(self, stock_code: str, sina_code: str) -> Optional[Dict[str, Any]]:
        """解析新浪行情文本格式"""
        try:
            url = f"https://hq.sinajs.cn/list={sina_code}"
            headers = {"Referer": "https://finance.sina.com.cn"}
            text = self._request_text(url, headers=headers, encoding="gbk")
            if text is None:
                return None

            if f'var hq_str_{sina_code}=""' in text:
                return None

            # 提取引号内的内容
            match = re.search(rf'var hq_str_{sina_code}="([^"]*)"', text)
            if not match:
                return None

            values = match.group(1).split(",")
            if len(values) < 33:
                return None

            name = values[0]
            if not name or name == "":
                return None

            # 构建买卖盘
            bid_ask = {}
            for i in range(1, 6):
                bid_v_idx = 10 + (i - 1) * 2
                bid_p_idx = 10 + (i - 1) * 2 + 1
                ask_v_idx = 20 + (i - 1) * 2
                ask_p_idx = 20 + (i - 1) * 2 + 1
                if bid_v_idx < len(values) and bid_p_idx < len(values):
                    bid_ask[f"bid{i}_volume"] = safe_int(values[bid_v_idx], default=0)
                    bid_ask[f"bid{i}_price"] = safe_float(values[bid_p_idx], default=0.0)
                if ask_v_idx < len(values) and ask_p_idx < len(values):
                    bid_ask[f"ask{i}_volume"] = safe_int(values[ask_v_idx], default=0)
                    bid_ask[f"ask{i}_price"] = safe_float(values[ask_p_idx], default=0.0)

            open_price = safe_float(values[1], default=0.0)
            prev_close = safe_float(values[2], default=0.0)
            current = safe_float(values[3], default=0.0)
            high = safe_float(values[4], default=0.0)
            low = safe_float(values[5], default=0.0)
            volume = safe_int(values[8], default=0)  # 股数
            amount = safe_float(values[9], default=0.0)  # 元

            pct_change = 0.0
            if prev_close and prev_close > 0:
                pct_change = round((current - prev_close) / prev_close, 4)

            return {
                "stock_code": re.sub(r'[^\d]', '', stock_code),
                "company_name": name,
                "latest_price": current,
                "open": open_price,
                "high": high,
                "low": low,
                "prev_close": prev_close,
                "volume": volume,
                "amount": amount,
                "pct_change": pct_change,
                "bid_ask": bid_ask,
                "update_time": f"{values[30]} {values[31]}" if len(values) > 31 else "",
                "source": self.name,
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"[{self.name}] 解析行情文本失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 2. K线数据
    # ------------------------------------------------------------------
    def _fetch_kline(self, stock_code: str, days: int = 120, **kwargs) -> Optional[Dict[str, Any]]:
        """获取新浪K线数据"""
        sina_code = self._get_sina_code(stock_code)
        # scale: 分钟数, 240=日线; ma: 均线; datalen: 数据条数
        url = (
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "CN_MarketData.getKLineData"
        )
        params = {
            "symbol": sina_code,
            "scale": 240,
            "ma": 5,
            "datalen": min(days, 200),  # 新浪限制
        }

        try:
            headers = {"Referer": "https://finance.sina.com.cn"}
            data = self._request(url, params=params, headers=headers)
            if not data or not isinstance(data, list):
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "day": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            # 新浪返回的是字符串，需转换
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "ma_price5" in df.columns:
                df["ma5"] = pd.to_numeric(df["ma_price5"], errors="coerce")
            if "ma_volume5" in df.columns:
                df["ma_volume5"] = pd.to_numeric(df["ma_volume5"], errors="coerce")

            return {
                "df": df,
                "records": len(df),
                "source": self.name,
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"[{self.name}] K线获取失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 3. 市场环境（通过新浪行情获取大盘指数）
    # ------------------------------------------------------------------
    def _fetch_market_context(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """获取大盘指数行情"""
        indices = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000300": "沪深300",
        }

        result_indices = {}
        for code, name in indices.items():
            try:
                url = f"https://hq.sinajs.cn/list={code}"
                headers = {"Referer": "https://finance.sina.com.cn"}
                text = self._request_text(url, headers=headers, encoding="gbk")
                if text is None:
                    continue
                match = re.search(rf'var hq_str_{code}="([^"]*)"', text)
                if match:
                    values = match.group(1).split(",")
                    if len(values) >= 5:
                        current = safe_float(values[3], default=0.0)
                        prev = safe_float(values[2], default=0.0)
                        pct = round((current - prev) / prev, 4) if prev and prev > 0 else 0.0
                        result_indices[name] = {
                            "close": current,
                            "change": round(current - prev, 2) if current is not None and prev is not None else 0.0,
                            "pct_change": pct,
                        }
            except Exception as e:
                logger.debug(f"[{self.name}] 获取{name}失败: {e}")

        if not result_indices:
            return None

        return {
            "indices": result_indices,
            "source": self.name,
            "fetch_time": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 4. 情绪/新闻
    # ------------------------------------------------------------------
    def _fetch_sentiment(self, stock_code: str, **kwargs) -> Optional[Dict[str, Any]]:
        """新浪暂无独立情绪指标，返回None让其他源补充"""
        return None

    def _fetch_news(self, stock_code: str, limit: int = 10, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """获取新浪财经新闻"""
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2516",
                "k": "",
                "num": str(limit),
                "r": "0.5",
            }
            resp = self._request(url, params=params)
            if not resp:
                return None

            result = resp.get("result", {})
            data = result.get("data", [])

            news = []
            for item in data[:limit]:
                title = item.get("title", "").strip()
                url = item.get("url", "")
                ctime = item.get("ctime", "")
                # 转换时间戳
                pub_time = ""
                try:
                    pub_time = datetime.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pub_time = ""

                if title and url:
                    news.append({
                        "title": title,
                        "url": url,
                        "source": "新浪财经",
                        "time": pub_time,
                    })

            return news if news else None
        except Exception as e:
            logger.warning(f"[{self.name}] 获取新闻失败: {e}")
            return None
