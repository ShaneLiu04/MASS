"""
MASS 爬虫工具模块
UA轮换、JSONP解析、股票代码标准化、数据校验
"""
import math
import re
import json
import random
from typing import Dict, Any, Optional, List

from loguru import logger


class UserAgentRotator:
    """浏览器User-Agent轮换器"""

    _UA_POOL = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    ]

    def get(self) -> str:
        return random.choice(self._UA_POOL)


def parse_jsonp(text: str) -> Optional[Dict[str, Any]]:
    """
    解析JSONP响应，提取JSON对象
    
    支持格式:
    - callback({...})
    - var x = {...};
    - ({...})
    - {...}
    """
    if not text:
        return None

    text = text.strip()

    # 尝试直接解析JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 匹配 callback({...}) 或 ({...})
    match = re.search(r'[\w$]*\((.*)\)\s*;?\s*$', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 匹配 var x = {...};
    match = re.search(r'=\s*(\{.*\})\s*;?\s*$', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 兜底：找第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"JSONP解析失败，无法提取有效JSON")
    return None


def standardize_stock_code(stock_code: str) -> Dict[str, str]:
    """
    标准化股票代码
    
    Args:
        stock_code: 原始代码，如 '600000', '000001', '300001'
        
    Returns:
        {
            "code6": "600000",       # 6位纯数字
            "secid": "1.600000",     # 东方财富格式
            "market": "sh",           # 市场代码
            "symbol": "600000.sh",   # 带后缀格式
        }
    """
    code = re.sub(r'[^\d]', '', stock_code)
    if len(code) != 6:
        raise ValueError(f"股票代码必须是6位数字，收到: {stock_code}")

    # 判断市场
    if code.startswith("6"):
        market = "sh"
        secid = f"1.{code}"
    elif code.startswith("0") or code.startswith("3"):
        market = "sz"
        secid = f"0.{code}"
    else:
        # 北交所等，默认上海
        market = "sh"
        secid = f"1.{code}"

    return {
        "code6": code,
        "secid": secid,
        "market": market,
        "symbol": f"{code}.{market}",
    }


def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    数据质量校验
    
    Args:
        data: 待校验的数据字典
        schema: 校验规则，格式如下:
            {
                "field_name": {
                    "required": True,           # 是否必填
                    "type": (int, float),       # 期望类型
                    "min": 0,                   # 最小值
                    "max": 1000,                # 最大值
                    "nonzero": True,            # 是否必须非零
                }
            }
            
    Returns:
        校验通过返回 True，否则返回 False
    """
    if not isinstance(data, dict):
        logger.warning("数据校验失败: 数据不是字典类型")
        return False

    for field, rules in schema.items():
        # 必填检查
        if rules.get("required", False) and field not in data:
            logger.warning(f"数据校验失败: 缺少必填字段 '{field}'")
            return False

        if field not in data:
            continue

        value = data[field]

        # 类型检查
        expected_types = rules.get("type")
        if expected_types and value is not None:
            if not isinstance(value, expected_types):
                logger.warning(f"数据校验失败: 字段 '{field}' 类型错误，期望 {expected_types}，实际 {type(value)}")
                return False

        # 数值范围检查
        if isinstance(value, (int, float)):
            if "min" in rules and value < rules["min"]:
                logger.warning(f"数据校验失败: 字段 '{field}' 值 {value} 小于最小值 {rules['min']}")
                return False
            if "max" in rules and value > rules["max"]:
                logger.warning(f"数据校验失败: 字段 '{field}' 值 {value} 大于最大值 {rules['max']}")
                return False
            if rules.get("nonzero") and value == 0:
                logger.warning(f"数据校验失败: 字段 '{field}' 不能为零")
                return False

    return True


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    安全转换为 float。

    - 对 None、"-"、空字符串 返回 default（默认 None），便于下游区分"数据缺失"与"真实值为0"
    - 保留真实 0 值，但过滤掉 NaN / Inf
    """
    if value is None or value == "-" or value == "":
        return default
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    安全转换为 int。

    - 对 None、"-"、空字符串 返回 default（默认 None）
    """
    if value is None or value == "-" or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def format_amount_wan(value: float) -> float:
    """将金额转换为万元单位（东方财富某些接口返回元）"""
    return round(value / 10000, 2) if value else 0.0


def format_amount_yi(value: float) -> float:
    """将金额转换为亿元单位"""
    return round(value / 100000000, 2) if value else 0.0
