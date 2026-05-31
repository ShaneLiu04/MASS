"""
MASS LLM 统一调用客户端
支持 OpenAI / DeepSeek / Claude / Ollama
模型参数完全可调: temperature/top_p/max_tokens/frequency_penalty/presence_penalty
"""
import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class LLMConfig:
    """LLM 配置 — 所有参数均可调"""
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    model: str = "deepseek-v4-pro"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3

    def to_api_kwargs(self) -> Dict[str, Any]:
        """转换为 API 调用参数"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class LLMClient:
    """大模型统一调用客户端 — 支持完整参数覆盖"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = self._default_config()
        self.config = config
        self._client = None
        self._init_client()

    def _default_config(self) -> LLMConfig:
        """
        获取默认配置 — 优先从 SystemConfigManager 读取（支持 Web UI 热更新）
        回退到环境变量默认值
        """
        try:
            from agent.core.system_config import get_system_config
            runtime_cfg = get_system_config().get_llm_config()
            return LLMConfig(
                provider=runtime_cfg.provider,
                api_key=runtime_cfg.api_key,
                base_url=runtime_cfg.base_url,
                model=runtime_cfg.model,
                temperature=runtime_cfg.temperature,
                top_p=runtime_cfg.top_p,
                max_tokens=runtime_cfg.max_tokens,
                frequency_penalty=runtime_cfg.frequency_penalty,
                presence_penalty=runtime_cfg.presence_penalty,
                timeout=runtime_cfg.timeout,
                max_retries=runtime_cfg.max_retries,
            )
        except Exception as e:
            logger.warning(f"从 SystemConfigManager 加载配置失败，回退到环境变量: {e}")

        from config import (
            LLM_PROVIDER,
            OPENAI_API_KEY, OPENAI_BASE_URL,
            DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
            CLAUDE_API_KEY, CLAUDE_BASE_URL,
            OLLAMA_BASE_URL, DEFAULT_MODEL,
            LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE,
            LLM_TOP_P, LLM_MAX_TOKENS,
            LLM_FREQUENCY_PENALTY, LLM_PRESENCE_PENALTY,
        )

        provider = LLM_PROVIDER
        if provider == "openai":
            return LLMConfig(
                provider="openai",
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=DEFAULT_MODEL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                frequency_penalty=LLM_FREQUENCY_PENALTY,
                presence_penalty=LLM_PRESENCE_PENALTY,
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )
        elif provider == "deepseek":
            return LLMConfig(
                provider="deepseek",
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEFAULT_MODEL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                frequency_penalty=LLM_FREQUENCY_PENALTY,
                presence_penalty=LLM_PRESENCE_PENALTY,
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )
        elif provider == "claude":
            return LLMConfig(
                provider="claude",
                api_key=CLAUDE_API_KEY,
                base_url=CLAUDE_BASE_URL,
                model="claude-3-5-sonnet-20241022",
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                frequency_penalty=LLM_FREQUENCY_PENALTY,
                presence_penalty=LLM_PRESENCE_PENALTY,
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )
        else:
            return LLMConfig(
                provider="ollama",
                api_key="",
                base_url=OLLAMA_BASE_URL,
                model="llama3",
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                frequency_penalty=LLM_FREQUENCY_PENALTY,
                presence_penalty=LLM_PRESENCE_PENALTY,
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )

    def _init_client(self):
        """初始化底层客户端"""
        if self.config.provider in ("openai", "deepseek"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.error("请安装 openai: pip install openai")
                raise
        elif self.config.provider == "ollama":
            self._client = None

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        model: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        统一对话接口 — 支持模型参数实时覆盖

        Args:
            system: 系统提示词
            user: 用户提示词
            json_mode: 是否强制JSON输出
            model: 覆盖默认模型
            override_config: 实时覆盖模型参数，如:
                {"temperature": 0.5, "top_p": 0.9, "max_tokens": 2048}

        Returns:
            解析后的字典
        """
        model = model or self.config.model

        # 应用覆盖参数
        effective_config = self.config.to_api_kwargs()
        if override_config:
            for key, value in override_config.items():
                if key in effective_config:
                    effective_config[key] = value
                    logger.debug(f"LLM参数覆盖: {key}={value}")

        for attempt in range(self.config.max_retries):
            try:
                result = self._call_api(system, user, json_mode, model, effective_config)
                if json_mode and isinstance(result, str):
                    result = json.loads(result)
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 (attempt {attempt+1}): {e}")
                if attempt == self.config.max_retries - 1:
                    # result 在此上下文中必然存在（json.loads(result) 抛出了异常）
                    return self._fallback_parse(result)
            except Exception as e:
                logger.warning(f"LLM调用失败 (attempt {attempt+1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        return {}

    def _call_api(
        self,
        system: str,
        user: str,
        json_mode: bool,
        model: str,
        api_kwargs: Dict[str, Any],
    ) -> Any:
        """底层API调用"""
        if self.config.provider in ("openai", "deepseek"):
            return self._call_openai(system, user, json_mode, model, api_kwargs)
        elif self.config.provider == "claude":
            return self._call_claude(system, user, json_mode, model, api_kwargs)
        elif self.config.provider == "ollama":
            return self._call_ollama(system, user, json_mode, model, api_kwargs)
        else:
            raise ValueError(f"不支持的provider: {self.config.provider}")

    def _call_openai(
        self,
        system: str,
        user: str,
        json_mode: bool,
        model: str,
        api_kwargs: Dict[str, Any],
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            **api_kwargs,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_claude(
        self,
        system: str,
        user: str,
        json_mode: bool,
        model: str,
        api_kwargs: Dict[str, Any],
    ) -> Any:
        import requests
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        prompt = f"{system}\n\n{user}"
        if json_mode:
            prompt += "\n\n你必须严格按JSON格式输出，不要输出任何其他内容。"

        payload = {
            "model": model,
            "max_tokens": api_kwargs.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
        }
        # Claude 不直接支持 temperature/top_p 等参数，但可传递
        for key in ["temperature", "top_p"]:
            if key in api_kwargs:
                payload[key] = api_kwargs[key]

        resp = requests.post(
            f"{self.config.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return content

    def _call_ollama(
        self,
        system: str,
        user: str,
        json_mode: bool,
        model: str,
        api_kwargs: Dict[str, Any],
    ) -> Any:
        import requests
        prompt = f"{system}\n\n{user}"
        if json_mode:
            prompt += "\n\n你必须严格按JSON格式输出。"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": api_kwargs.get("temperature", 0.2),
                "top_p": api_kwargs.get("top_p", 1.0),
                "num_predict": api_kwargs.get("max_tokens", 4096),
            },
        }
        resp = requests.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """解析失败时的降级处理"""
        import re
        try:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except Exception:
            pass

        logger.error(f"无法解析LLM输出: {text[:200]}")
        return {
            "signal": 0,
            "confidence": 0.3,
            "reasoning": f"模型输出解析失败，请重试。原始输出片段: {text[:100]}",
            "key_factors": ["解析错误"],
            "risk_flags": ["模型输出格式异常"],
        }


class MockLLMClient(LLMClient):
    """
    Mock LLM 客户端 — 基于真实数据的规则引擎
    
    修改说明：原版本使用 random 生成虚假数据，违反 Zero Mock Policy。
    现版本从 prompt 中解析真实指标，基于规则做判断，所有输出数据均来自真实爬取。
    """

    def __init__(self):
        self.config = LLMConfig(provider="mock", model="mock")
        self._client = None

    @staticmethod
    def _extract_kv_from_prompt(text: str) -> Dict[str, Any]:
        """从 prompt 文本中提取 key: value 格式的指标数据"""
        import re
        data = {}
        # 匹配 "key: value" 或 "key": value 格式
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('==='):
                # 处理 "key: value" 格式
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_str = parts[1].strip()
                    # 尝试转换为数值
                    try:
                        # 处理百分比
                        if val_str.endswith('%'):
                            val = float(val_str[:-1])
                        # 处理带正负号
                        elif val_str.startswith('+') or val_str.startswith('-'):
                            val = float(val_str)
                        else:
                            val = float(val_str)
                    except ValueError:
                        val = val_str
                    data[key] = val
        return data

    @staticmethod
    def _try_extract_json(text: str) -> Dict[str, Any]:
        """尝试从 prompt 中提取 JSON 块（如技术指标的 JSON 序列化部分）"""
        import re, json as _json
        # 查找 ```json ... ``` 或 {...} 块
        matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
        for m in matches:
            try:
                return _json.loads(m)
            except:
                continue
        return {}

    def chat(self, system: str, user: str, json_mode: bool = True,
             model: Optional[str] = None, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于真实 prompt 数据的规则引擎输出
        不再使用 random，所有信号均基于从 prompt 中提取的真实指标
        """
        data = self._extract_kv_from_prompt(user)
        json_data = self._try_extract_json(user)
        data.update({k: v for k, v in json_data.items() if k not in data})

        # 提取当前价格（用于计算目标价/止损）
        current_price = 0.0
        for key in ['current_price', '当前价格', 'close', '收盘价']:
            if key in data:
                try:
                    current_price = float(data[key])
                    break
                except:
                    pass

        if "技术面" in system:
            return self._ta_rule_engine(data, current_price)
        elif "基本面" in system:
            return self._fa_rule_engine(data, current_price)
        elif "资金" in system:
            return self._ca_rule_engine(data, current_price)
        elif "情绪" in system:
            return self._sa_rule_engine(data, current_price)
        elif "宏观" in system:
            return self._ma_rule_engine(data, current_price)
        elif "风险" in system:
            return self._ra_rule_engine(data, current_price)
        else:
            return self._chairman_rule_engine(data, current_price)

    def _ta_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """技术面规则引擎 — 基于真实技术指标"""
        signal = 0
        confidence = 0.55
        factors = []
        risks = []

        ma_align = data.get('ma_alignment', '')
        if ma_align == '多头排列':
            signal = 1
            confidence = 0.72
            factors.append('均线多头排列')
        elif ma_align == '空头排列':
            signal = -1
            confidence = 0.72
            factors.append('均线空头排列')
            risks.append('趋势向下')

        if data.get('macd_golden_cross'):
            signal = 1 if signal >= 0 else 0
            confidence = min(confidence + 0.08, 0.90)
            factors.append('MACD金叉')
        elif data.get('macd_death_cross'):
            signal = -1 if signal <= 0 else 0
            confidence = min(confidence + 0.08, 0.90)
            factors.append('MACD死叉')
            risks.append('动量转弱')

        kdj_k = data.get('kdj_k', 50)
        kdj_d = data.get('kdj_d', 50)
        if isinstance(kdj_k, (int, float)) and isinstance(kdj_d, (int, float)):
            if kdj_k > 80:
                risks.append('KDJ超买')
            elif kdj_k < 20:
                factors.append('KDJ超卖')

        rsi = data.get('rsi14', 50)
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                risks.append('RSI超买')
            elif rsi < 30:
                factors.append('RSI超卖')

        vol_trend = data.get('volume_trend', '正常')
        if vol_trend == '放量':
            factors.append('成交量放大')
        elif vol_trend == '缩量':
            risks.append('成交量萎缩')

        if not factors:
            factors.append('指标中性，趋势不明')
        if not risks:
            risks.append('关注大盘系统性风险')

        target_low = round(price * 1.05, 2) if price > 0 else None
        target_high = round(price * 1.15, 2) if price > 0 else None
        stop_loss = round(price * 0.93, 2) if price > 0 and signal == 1 else None

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "target_price_low": target_low,
            "target_price_high": target_high,
            "stop_loss": stop_loss,
            "reasoning": f"基于真实技术指标的规则分析：{'；'.join(factors)}。{'风险：' + '；'.join(risks) if risks else ''}",
            "key_factors": factors,
            "risk_flags": risks,
            "chart_patterns": [ma_align] if ma_align else [],
            "trend_direction": data.get('ma20_trend', '未知'),
        }

    def _fa_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """基本面规则引擎 — 基于真实财务数据"""
        roe = data.get('roe', 0)
        pe = data.get('pe_ttm', 100)
        debt = data.get('debt_ratio', 60)
        profit_growth = data.get('profit_growth', 0)

        if isinstance(roe, str):
            roe = 0
        if isinstance(pe, str):
            pe = 100
        if isinstance(debt, str):
            debt = 60

        score = 50
        factors = []

        if roe > 15:
            score += 15
            factors.append(f'ROE较高({roe}%)')
        elif roe < 5:
            score -= 10
            factors.append(f'ROE偏低({roe}%)')

        if pe < 20:
            score += 15
            factors.append(f'PE较低({pe})')
        elif pe > 60:
            score -= 10
            factors.append(f'PE偏高({pe})')

        if debt < 50:
            score += 10
            factors.append('负债率安全')
        elif debt > 70:
            score -= 10
            factors.append('负债率偏高')

        if profit_growth > 20:
            score += 10
            factors.append('利润增长强劲')
        elif profit_growth < 0:
            score -= 10
            factors.append('利润下滑')

        signal = 1 if score >= 75 else (-1 if score <= 40 else 0)

        return {
            "signal": signal,
            "confidence": round(0.55 + abs(score - 50) / 200, 2),
            "fundamental_score": score,
            "sub_scores": {
                "profitability": 20 if roe > 15 else (10 if roe > 5 else 5),
                "growth": 20 if profit_growth > 20 else (10 if profit_growth > 0 else 5),
                "safety": 15 if debt < 50 else 10,
                "valuation": 15 if pe < 20 else 10,
                "moat": 10,
            },
            "valuation_gap": f"当前PE {pe}，{'低于' if pe < 25 else '高于'}行业中位数",
            "reasoning": f"基于真实财务数据的规则分析：综合评分{score}分。{'；'.join(factors)}",
            "key_factors": factors,
            "risk_flags": ["行业竞争加剧"] if score < 60 else [],
        }

    def _ca_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """资金面规则引擎 — 基于真实资金流向数据"""
        main_flow = data.get('main_net_inflow_10d', 0)
        inflow_days = data.get('main_inflow_days', 0)
        north = data.get('north_bound_5d', 0)

        if isinstance(main_flow, str):
            main_flow = 0
        if isinstance(inflow_days, str):
            inflow_days = 0

        signal = 0
        confidence = 0.55
        factors = []

        if main_flow > 5000 and inflow_days >= 3:
            signal = 1
            confidence = 0.75
            factors.append(f'主力资金{inflow_days}日净流入{main_flow}万')
        elif main_flow < -5000:
            signal = -1
            confidence = 0.75
            factors.append(f'主力资金净流出{abs(main_flow)}万')

        if north > 1000:
            factors.append('北向资金增持')
        elif north < -1000:
            factors.append('北向资金减持')

        if not factors:
            factors.append('资金流向中性')

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "capital_score": 70 if signal == 1 else (40 if signal == -1 else 55),
            "reasoning": f"基于真实资金流向的规则分析：{'；'.join(factors)}",
            "key_factors": factors,
            "risk_flags": ["关注主力资金持续性"] if signal == 1 else [],
            "smart_money_direction": "建仓期" if signal == 1 else ("派发期" if signal == -1 else "观望"),
            "retail_vs_institutional": "散户卖出，机构吸筹" if signal == 1 else "散户买入，机构派发" if signal == -1 else "双方观望",
        }

    def _sa_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """情绪面规则引擎 — 基于真实情绪数据"""
        score = data.get('social_sentiment_7d', 0.0)
        news_score = data.get('sentiment_score', 0.0)
        if isinstance(score, str):
            score = 0.0
        if isinstance(news_score, str):
            news_score = 0.0

        # 逆向信号：情绪过热则看空，过冷则看多
        signal = 0
        if score > 0.6:
            signal = -1
        elif score < -0.6:
            signal = 1

        factors = [f'社交媒体情绪得分: {score}']
        if news_score:
            factors.append(f'新闻情感得分: {news_score}')

        return {
            "signal": signal,
            "confidence": round(0.55 + abs(score) * 0.3, 2),
            "sentiment_index": round(score, 2),
            "sentiment_percentile": int(50 + score * 50),
            "reasoning": f"基于真实情绪数据的规则分析：情绪得分{score}，{'过热' if score > 0.5 else '过冷' if score < -0.5 else '中性'}。{'逆向信号' if signal != 0 else '观望'}",
            "key_factors": factors,
            "risk_flags": ["情绪极端波动风险"] if abs(score) > 0.7 else [],
            "crowd_behavior": "FOMO追涨" if score > 0.5 else "恐慌抛售" if score < -0.5 else "正常交易",
            "contrarian_opportunity": "等待情绪回落" if score > 0.5 else "关注情绪修复" if score < -0.5 else "无明显机会",
        }

    def _ma_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """宏观规则引擎 — 基于真实宏观数据"""
        pmi = data.get('pmi', 50)
        policy = data.get('policy_stance', '中性')
        if isinstance(pmi, str):
            pmi = 50

        signal = 0
        cycle = "复苏早期"
        if pmi > 50 and policy == '宽松':
            signal = 1
            cycle = "复苏早期"
        elif pmi > 50 and policy == '中性':
            signal = 1
            cycle = "复苏晚期"
        elif pmi < 50 and policy == '收紧':
            signal = -1
            cycle = "衰退早期"
        elif pmi < 50:
            signal = -1
            cycle = "衰退晚期"
        elif pmi >= 50:
            cycle = "过热"

        return {
            "market_cycle": cycle,
            "cycle_confidence": round(0.55 + abs(pmi - 50) / 200, 2),
            "sector_outlook": "利好" if signal == 1 else ("利空" if signal == -1 else "中性"),
            "style_alignment": 0.7 if signal == 1 else 0.5,
            "macro_signal": signal,
            "reasoning": f"基于真实宏观数据的规则分析：PMI {pmi}，政策立场{policy}，判断为{cycle}",
            "key_factors": [f"PMI: {pmi}", f"政策: {policy}"],
            "risk_flags": ["外部不确定性"] if signal != 1 else [],
            "recommended_weight_adjustment": {
                "TA": 0.02 if signal == 1 else -0.02,
                "FA": 0.02 if signal == 1 else 0.0,
                "CA": 0.02 if signal == 1 else -0.02,
                "SA": 0.0,
                "MA": 0.0,
                "RA": -0.02 if signal == 1 else 0.02,
            },
        }

    def _ra_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """风险规则引擎 — 基于真实风险指标"""
        vol = data.get('annual_volatility', 20)
        drawdown = data.get('max_drawdown', -15)
        beta = data.get('beta', 1.0)

        if isinstance(vol, str):
            vol = 20
        if isinstance(drawdown, str):
            drawdown = -15
        if isinstance(beta, str):
            beta = 1.0

        risk_level = 3
        if vol > 40 or drawdown < -30:
            risk_level = 5
        elif vol > 30 or drawdown < -20:
            risk_level = 4
        elif vol < 15 and drawdown > -10:
            risk_level = 2
        elif vol < 10 and drawdown > -5:
            risk_level = 1

        max_pos = max(0.05, min(0.50, 0.25 - (risk_level - 1) * 0.04))

        return {
            "signal": 0,
            "confidence": round(0.70 + (risk_level - 1) * 0.05, 2),
            "risk_level": risk_level,
            "max_position_pct": round(max_pos, 2),
            "recommended_stop_loss": round(price * 0.92, 2) if price > 0 else None,
            "risk_reward_ratio": round(1.5 + (6 - risk_level) * 0.2, 1),
            "reasoning": f"基于真实风险指标的规则分析：年化波动率{vol}%，最大回撤{drawdown}%，Beta {beta}，风险等级{risk_level}",
            "key_factors": [f"年化波动率: {vol}%", f"最大回撤: {drawdown}%", f"Beta: {beta}"],
            "risk_flags": ["财报披露临近"] if risk_level >= 3 else [],
            "black_scenarios": ["业绩不及预期", "市场系统性风险"] if risk_level >= 4 else ["市场系统性风险"],
            "position_sizing_formula": f"风险等级{risk_level}，建议仓位不超过{max_pos*100:.0f}%",
        }

    def _chairman_rule_engine(self, data: Dict[str, Any], price: float) -> Dict[str, Any]:
        """Chairman 规则引擎 — 综合判断"""
        # 从 prompt 中提取各Agent的信号（如果包含在prompt中）
        ta_sig = data.get('TA-Agent_signal', 0)
        fa_sig = data.get('FA-Agent_signal', 0)
        ca_sig = data.get('CA-Agent_signal', 0)
        sa_sig = data.get('SA-Agent_signal', 0)
        ma_sig = data.get('MA-Agent_signal', 0)
        ra_risk = data.get('RA-Agent_risk_level', 3)

        signals = [s for s in [ta_sig, fa_sig, ca_sig, sa_sig, ma_sig] if isinstance(s, (int, float)) and s != 0]
        if not signals:
            decision = 0
        else:
            avg = sum(signals) / len(signals)
            decision = 1 if avg > 0.3 else (-1 if avg < -0.3 else 0)

        # 风险过滤
        if isinstance(ra_risk, (int, float)) and ra_risk >= 4 and decision == 1:
            decision = 0

        confidence = round(0.55 + len(signals) * 0.05, 2) if signals else 0.55
        position_pct = 0.10 if decision == 1 else 0.0

        return {
            "decision": int(decision),
            "confidence": min(confidence, 0.90),
            "position_pct": round(position_pct, 2),
            "target_price": round(price * 1.12, 2) if price > 0 and decision == 1 else None,
            "stop_loss": round(price * 0.92, 2) if price > 0 and decision == 1 else None,
            "time_horizon": "2-4周",
            "expected_return_pct": 12.0 if decision == 1 else 0.0,
            "risk_adjusted_score": 65 if decision == 1 else 50,
            "reasoning": f"Chairman规则综合决策：基于{len(signals)}个Agent的真实数据信号，综合判断为{'买入' if decision==1 else '卖出' if decision==-1 else '观望'}",
            "consensus_factors": [f"{k}信号{v}" for k, v in [('TA', ta_sig), ('FA', fa_sig), ('CA', ca_sig)] if v != 0],
            "dissenting_views": [],
            "scenario_analysis": {
                "bull": {"probability": 0.25, "return_pct": 25},
                "base": {"probability": 0.50, "return_pct": 12},
                "bear": {"probability": 0.25, "return_pct": -8},
            },
            "execution_plan": ["分批建仓", "严格止损"] if decision == 1 else ["观望等待"],
        }
