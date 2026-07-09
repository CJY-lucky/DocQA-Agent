# CLAUDE.md — DocQA Agent 项目编码规范

## 项目概述

DocQA Agent 是一个基于 LLM 的项目文档智能问答系统，能够自动索引代码仓库/文档库，通过自然语言问答帮助开发者查找信息。

## 技术栈

- Python 3.10+
- OpenAI API（兼容）用于 LLM 调用
- ChromaDB 向量数据库
- Sentence-Transformers 嵌入模型
- LangChain 文本分割器
- rank-bm25 关键词检索
- Click CLI 框架
- Rich 终端美化

## 编码规范

### 风格
- 遵循 PEP 8
- 类型注解：所有公开函数必须有完整的类型注解
- 文档字符串：所有公开函数使用 Google 风格的 docstring
- 行宽上限：100 字符

### 项目结构
```
src/
├── agent/       # Agent 核心逻辑（runLoop + 工具）
├── retrieval/   # 检索引擎（向量 + BM25）
├── ingestion/   # 文档加载与分块
├── llm/         # LLM 调用封装
└── cli/         # CLI 入口
tests/           # 单元测试
```

### 命名约定
- 文件名：snake_case
- 类名：PascalCase
- 函数/方法：snake_case
- 常量：UPPER_SNAKE_CASE
- 私有成员：前缀 `_`

### 错误处理
- 使用明确的自定义异常类
- 关键路径上加 try/except 并提供有意义的错误消息
- 不吞异常：要么处理，要么向上抛出

### 日志
- 使用 Python `logging` 模块
- 配置在 `src/__init__.py` 中集中管理
- 关键操作（检索、LLM 调用、工具调用）记录 INFO 级别日志

### 测试
- 使用 `pytest`
- 测试文件放在 `tests/` 目录
- 核心检索和 Agent 逻辑必须有测试覆盖
