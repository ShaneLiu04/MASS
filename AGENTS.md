# MASS 开发指南

## 项目结构

```
agent/              # 多智能体核心
  core/             # 黑板、编排器、决策引擎、辩论、缓存
  agents/           # 6个专业Agent + 基类
  tools/            # LLM客户端、数据工具、指标计算、情感分析
  crawlers/         # 统一爬虫框架（东方财富、同花顺等多源）
  models/           # Pydantic模型、数据库封装
  prompts/system/   # 可热更新的Prompt模板
api/                # Flask Blueprint + 中间件
tests/              # 测试套件
  unit/             # 单元测试
  integration/      # 集成测试
  e2e/              # 端到端测试
mass_cli/           # 命令行工具
```

## 开发规范

### Agent开发

1. **继承BaseAgent**：所有Agent必须继承`BaseAgent`
2. **实现analyze方法**：返回`AgentOpinion`对象
3. **Prompt文件**：在`agent/prompts/system/{agent_id}.md`放置系统提示词
4. **Pydantic校验**：输出必须符合对应的Pydantic模型
5. **降级处理**：LLM调用失败时提供规则引擎回退

### API开发

1. **使用延迟初始化**：`_get_orchestrator()`等函数避免启动时加载
2. **缓存策略**：诊断接口使用5分钟缓存
3. **错误码**：所有错误响应包含`code`字段
4. **输入校验**：股票代码必须是6位数字

### 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 覆盖率报告
pytest tests/ --cov=agent --cov=api --cov-report=html
```

## 配置说明

复制 `.env.example` 为 `.env` 并配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| LLM_PROVIDER | LLM提供商 | openai |
| OPENAI_API_KEY | OpenAI密钥 | - |
| USE_MOCK_LLM | 使用Mock模式 | True |
| FLASK_DEBUG | Flask调试模式 | True |
| FLASK_PORT | 服务端口 | 5000 |

## 扩展指南

### 添加新Agent

1. 在 `agent/agents/` 创建 `{xxx}_agent.py`
2. 继承 `BaseAgent`，实现 `analyze()`
3. 创建 `agent/prompts/system/xxx_agent.md`
4. 在 `agent/agents/__init__.py` 导出
5. 在 `agent/core/orchestrator.py` 注册
6. 在 `agent/models/agent_response.py` 添加输出模型

### 添加新数据源

#### 方式一：通过爬虫框架（推荐）

1. 在 `agent/crawlers/` 创建新的爬虫类，继承 `BaseCrawler`
2. 实现 `fetch(stock_code, data_type)` 方法
3. 在 `StockDataTool.__init__()` 中通过 `CrawlerRegistry.register()` 注册
4. 遵循反爬策略：设置合理的 `request_interval` 和 `retries`

```python
from agent.crawlers.base import BaseCrawler

class MyCrawler(BaseCrawler):
    name = "my_source"
    priority = 80
    
    def fetch(self, stock_code, data_type, **kwargs):
        if data_type == "fundamentals":
            return {...}
        return None
```

#### 方式二：直接扩展 StockDataTool

在 `agent/tools/stock_data_tool.py` 中添加新方法，遵循缓存模式。

## 调试技巧

- 设置 `USE_MOCK_LLM=True` 可免API密钥测试
- 查看 `logs/` 目录获取详细日志
- 使用 `/api/agent/blackboard/clear` 清理缓存
