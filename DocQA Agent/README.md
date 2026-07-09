<p align="center">
  <h1 align="center">🤖 DocQA Agent</h1>
  <p align="center"><strong>基于 LLM 的项目文档智能问答系统</strong></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/tests-43%20passed-brightgreen.svg" alt="Tests">
</p>

---

## 🎯 项目简介

DocQA Agent 是一个**能够自动"学习"代码仓库或文档库**并回答问题的智能 Agent。

不同于传统的 RAG 系统（检索 → 生成，一次完成），DocQA Agent 的核心理念是：**当检索结果不足以回答问题时，Agent 会主动调用工具（grep / 读文件 / 列目录）去探索项目，直到找到足够的信息再回答**。

> 日常场景：接手一个新项目时，不用再对着代码库盲目搜索——直接问 Agent"这个项目的入口文件在哪""数据库连接怎么配置的"，它会自己翻代码找答案。

---

## ✨ 核心特性

| 特性 | 说明 |
|:---|:---|
| 🔍 **混合检索** | 向量语义检索 + BM25 关键词检索，RRF 融合取 Top-K |
| 🔧 **Agent 工具调用** | `search_code`（grep）· `read_file` · `list_dir` —— 信息不够时自主探索文件系统 |
| 🔄 **多轮对话 + 记忆** | 记住历史对话，支持追问和澄清 |
| 🎯 **置信度评估** | 每次回答自评 0-100，低于阈值主动追问用户 |
| ♻️ **反思机制** | 用户说"不对"时，换检索策略重新回答 |
| ⚙️ **配置文件驱动** | `config.yaml` 一站式配置 LLM、嵌入模型、检索参数 |
| 🧪 **完整测试覆盖** | 43 个单元测试，覆盖检索、工具、Agent 核心逻辑 |

---

## 🏗️ 系统架构

```
                        用户: "这个项目的入口文件是哪个？"
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DocQA Agent                              │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Agent Loop (runLoop)                    │   │
│   │                                                         │   │
│   │  迭代 1: 检索 → 信息不够 → 调 list_dir + search_code     │   │
│   │  迭代 2: 发现 main.ts → 调 read_file 读内容              │   │
│   │  迭代 3: 信息充足 → 生成回答 → 置信度 94/100             │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                    │                    │               │
│        ▼                    ▼                    ▼               │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────┐        │
│  │ 检索引擎  │    │    LLM 客户端    │    │   Agent 工具  │        │
│  │          │    │                 │    │              │        │
│  │ FAISS    │    │ OpenAI 兼容 API │    │ search_code  │        │
│  │ 向量检索  │    │ (千问/GPT/任意) │    │ read_file    │        │
│  │ + BM25   │    │                 │    │ list_dir     │        │
│  │ 混合检索  │    │ Function        │    │              │        │
│  └──────────┘    │ Calling         │    └──────────────┘        │
│                  └─────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    回答 + 引用来源 + 置信度评分
```

---

## 🧱 技术栈

| 组件 | 技术 | 说明 |
|:---|:---|:---|
| **LLM 调用** | `openai` SDK + 阿里云 DashScope | 兼容 OpenAI API，支持千问/GPT/Ollama 等 |
| **嵌入模型** | `sentence-transformers/all-MiniLM-L6-v2` | 384 维轻量模型，本地 CPU 运行 |
| **向量存储** | `FAISS` | 纯 numpy 持久化，不依赖外部服务 |
| **BM25** | `rank_bm25` | 关键词稀疏检索 |
| **文本分块** | LangChain `RecursiveCharacterTextSplitter` | Markdown 语义分块 |
| **CLI** | `click` + `rich` | 终端美化交互 |

---

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/docqa-agent.git
cd docqa-agent

# 2. 安装依赖
pip install -r requirements.txt
```

首次运行时会自动从 HuggingFace 镜像下载嵌入模型（约 90MB），后续无需重新下载。

---

## ⚙️ 配置

编辑 `config.yaml`：

```yaml
llm:
  model: "qwen3.7-max"                              # 模型名称
  api_key: "${DASHSCOPE_API_KEY}"                   # 环境变量传 API Key
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"   # 嵌入模型

retrieval:
  top_k: 5                                          # 每次检索返回片段数

agent:
  max_iterations: 5                                 # 单次回答最大工具调用次数
  confidence_threshold: 60                           # 置信度低于此值会追问

scan:
  extensions: [".md", ".txt", ".py"]                # 要索引的文件类型
```

```bash
# 设置 API Key
export DASHSCOPE_API_KEY="your-api-key"
```

支持的 LLM 后端：阿里云千问、OpenAI、Ollama 等任何 OpenAI 兼容 API。

---

## 🚀 使用

### 索引文档

```bash
python -m src.cli.main index /path/to/your/project --force
```

递归扫描目录，加载匹配的文档/代码，分块、嵌入、建立向量 + BM25 双索引。

```
📚 DocQA Agent — 文档索引
目标目录: /path/to/project
文件类型: .md, .py, .txt
分块大小: 500
嵌入模型: sentence-transformers/all-MiniLM-L6-v2

                   索引完成
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 项目       ┃ 数量                           ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 文档数     │ 85                             │
│ 分块数     │ 4821                           │
│ 向量索引   │ 4821                           │
│ BM25 索引  │ 4821                           │
└────────────┴────────────────────────────────┘
✅ 索引完成！
```

### 单次问答

```bash
python -m src.cli.main ask "这个项目怎么安装？" --workspace /path/to/project
```

### 交互对话模式（推荐）

```bash
python -m src.cli.main chat --workspace /path/to/project
```

```
你: 项目入口文件是什么？
🤖 DocQA Agent: 根据项目结构，入口文件是 ... 置信度 94/100

你: 循环机制是怎么实现的？
🤖 DocQA Agent: ... 置信度 87/100

你: 不对，重新查一下
🤖 DocQA Agent: (触发反思，换策略重新检索) ...

你: /reset    ← 清空对话
你: /quit     ← 退出
```

---

## 📂 项目结构

```
docqa-agent/
├── config.yaml                    # 配置文件
├── requirements.txt               # Python 依赖
├── src/
│   ├── ingestion/
│   │   ├── loader.py              # 文档加载器（自动编码检测）
│   │   └── chunker.py             # 语义分块器
│   ├── retrieval/
│   │   ├── vector_store.py        # FAISS 向量检索
│   │   ├── bm25_store.py          # BM25 关键词检索
│   │   └── hybrid_retriever.py    # RRF 混合检索
│   ├── llm/
│   │   └── client.py              # OpenAI 兼容 LLM 封装
│   ├── agent/
│   │   ├── loop.py                # Agent runLoop（核心）
│   │   └── tools.py               # search_code/read_file/list_dir
│   └── cli/
│       └── main.py                # CLI 入口
├── tests/
│   ├── test_retrieval.py          # 检索模块测试
│   └── test_agent.py              # Agent 模块测试
└── data/faiss/                    # 向量索引持久化目录（自动生成）
```

---

## 🧪 运行测试

```bash
pytest tests/ -v                    # 全部测试
pytest tests/ -v -m "not slow"      # 跳过需要下载模型的慢速测试
```

---

## 🔄 Agent 工作流程

一次典型的问答会经历以下步骤：

```
用户提问
    │
    ▼
① 混合检索 ───── 向量(语义) + BM25(关键词) → RRF 融合 Top-5
    │
    ▼
② LLM 判断 ───── 基于检索结果 + 已获取的工具输出
    │              判断信息是否足够回答问题
    │
    ├─ 足够 → ③ 直接生成回答
    │
    └─ 不足 → 调用工具（search_code / read_file / list_dir）
                │
                ▼
           工具执行 → 结果追加到对话上下文
                │
                ▼
           回到 ②（最多 5 轮）
    │
    ▼
③ 置信度评估 ─── LLM 自评回答可靠性（0-100）
    │
    ├─ ≥ 阈值(60) → 直接返回答案 + 🟢
    └─ < 阈值(60) → 返回答案 + 追问提示 + 🟡/🔴
```

---

## 📝 License

MIT

---

## 💡 设计理念

这个项目串联了构建 AI Agent 的所有核心概念：

- **RAG 检索增强生成** — 向量 + BM25 混合检索
- **Function Calling** — Agent 自主决定何时、如何调用工具
- **runLoop 多轮决策** — 检索不够 → 调工具 → 再判断 → 回答 / 再调工具
- **置信度过滤** — LLM 自评，低置信度追问用户
- **反思机制** — 用户反馈错误时自动切换策略

如果这个项目对你有帮助，欢迎 ⭐ Star！
