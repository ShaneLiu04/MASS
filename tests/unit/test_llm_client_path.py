"""
LLMClient 真实调用路径单元测试

验证：在 USE_MOCK_LLM=False 时，LLMClient 会正确调用底层 HTTP 客户端，
      构造符合各提供商 API 规范的请求参数。

不依赖真实 API Key，使用 unittest.mock 拦截底层调用。
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.llm_client import LLMClient, LLMConfig


class TestLLMClientOpenAIPath:
    """验证 OpenAI/DeepSeek 调用路径"""

    @patch("openai.OpenAI")
    def test_chat_constructs_correct_messages(self, mock_openai_cls):
        """验证 chat() 构造了正确的 messages 参数"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"result": "ok"}'))]
        )

        client = LLMClient(
            config=LLMConfig(
                provider="deepseek",
                api_key="sk-test",
                base_url="https://api.test.com/v1",
                model="test-model",
                temperature=0.5,
                top_p=0.9,
                max_tokens=2048,
            )
        )
        result = client.chat(
            system="你是一个测试助手",
            user="请回复OK",
            json_mode=True,
        )

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs

        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "你是一个测试助手"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "请回复OK"
        assert result == {"result": "ok"}

    @patch("openai.OpenAI")
    def test_chat_without_json_mode(self, mock_openai_cls):
        """验证 json_mode=False 时不设置 response_format"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="plain text"))]
        )

        client = LLMClient(
            config=LLMConfig(provider="openai", api_key="sk-test", base_url="https://api.openai.com/v1")
        )
        result = client.chat(system="sys", user="usr", json_mode=False)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert result == "plain text"

    @patch("openai.OpenAI")
    def test_chat_with_override_config(self, mock_openai_cls):
        """验证 override_config 会覆盖默认参数"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{}'))]
        )

        client = LLMClient(
            config=LLMConfig(
                provider="deepseek",
                api_key="sk-test",
                temperature=0.2,
                max_tokens=4096,
            )
        )
        client.chat(
            system="sys",
            user="usr",
            override_config={"temperature": 0.8, "max_tokens": 512},
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 512

    @patch("openai.OpenAI")
    def test_chat_retries_on_failure(self, mock_openai_cls):
        """验证失败时会重试"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            ConnectionError("第一次失败"),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"retry": "ok"}'))]),
        ]

        client = LLMClient(
            config=LLMConfig(
                provider="deepseek",
                api_key="sk-test",
                max_retries=2,
            )
        )
        result = client.chat(system="sys", user="usr")

        assert mock_client.chat.completions.create.call_count == 2
        assert result == {"retry": "ok"}

    @patch("openai.OpenAI")
    def test_chat_fallback_parse_on_invalid_json(self, mock_openai_cls):
        """验证 JSON 解析失败时会调用 fallback_parse"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json"))]
        )

        client = LLMClient(
            config=LLMConfig(provider="deepseek", api_key="sk-test", max_retries=1)
        )
        # fallback_parse 可能返回提取的键值对或空 dict
        result = client.chat(system="sys", user="usr", json_mode=True)
        assert isinstance(result, dict)


class TestLLMClientClaudePath:
    """验证 Claude 调用路径"""

    @patch("requests.post")
    def test_chat_uses_requests_for_claude(self, mock_post):
        """验证 Claude provider 使用 requests.post"""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": '{"result": "claude_ok"}'}]
            },
        )

        client = LLMClient(
            config=LLMConfig(
                provider="claude",
                api_key="claude-test-key",
                base_url="https://api.anthropic.com",
                model="claude-test",
                temperature=0.3,
            )
        )
        result = client.chat(system="sys", user="usr", json_mode=True)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args.kwargs["headers"]
        assert headers["x-api-key"] == "claude-test-key"
        assert headers["anthropic-version"] == "2023-06-01"

        payload = call_args.kwargs["json"]
        assert payload["model"] == "claude-test"
        assert payload["temperature"] == 0.3
        assert "你必须严格按JSON格式输出" in payload["messages"][0]["content"]
        assert result == {"result": "claude_ok"}
