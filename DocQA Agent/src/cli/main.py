"""CLI 入口 —— 使用 Click 框架，支持 index / ask / chat 命令。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.loop import DocQAAgent
from src.agent.tools import AgentTools
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.llm.client import LLMClient
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]⚠ 配置文件不存在: {config_path}，使用默认值[/]")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_agent(config: dict, workspace_dir: str) -> DocQAAgent:
    """根据配置构建 Agent 实例。"""
    llm_cfg = config.get("llm", {})
    embed_cfg = config.get("embedding", {})
    retrieval_cfg = config.get("retrieval", {})
    index_cfg = config.get("index", {})
    agent_cfg = config.get("agent", {})

    # LLM 客户端
    llm = LLMClient(
        model=llm_cfg.get("model", "gpt-3.5-turbo"),
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", ""),
        temperature=float(llm_cfg.get("temperature", 0.1)),
        max_tokens=int(llm_cfg.get("max_tokens", 1024)),
    )

    # 向量存储
    vector_store = VectorStore(
        persist_dir=index_cfg.get("persist_dir", "./data/faiss"),
        embedding_model=embed_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=embed_cfg.get("device", "cpu"),
    )

    # BM25 存储 —— 从 ChromaDB 读取已索引的文档来重建
    bm25_store = BM25Store()
    existing_chunks = vector_store.get_all_chunks()
    if existing_chunks:
        bm25_store.index_chunks(existing_chunks)
        vector_store._cached_count = len(existing_chunks)
        console.print(f"[dim]已加载 {len(existing_chunks)} 个块到 BM25 索引[/]")

    # 混合检索器
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        vector_weight=float(retrieval_cfg.get("vector_weight", 0.5)),
        bm25_weight=float(retrieval_cfg.get("bm25_weight", 0.5)),
    )

    # 工具
    tools = AgentTools(workspace_dir)

    # Agent
    agent = DocQAAgent(
        retriever=retriever,
        llm_client=llm,
        tools=tools,
        confidence_threshold=int(agent_cfg.get("confidence_threshold", 60)),
        max_iterations=int(agent_cfg.get("max_iterations", 5)),
        max_history=int(agent_cfg.get("max_history", 10)),
    )

    return agent


# ================================================================
# CLI 命令
# ================================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="DocQA Agent")
def cli():
    """DocQA Agent —— 基于 LLM 的项目文档智能问答系统

    支持对任意代码仓库或文档库进行自动索引，通过自然语言交互回答问题。
    """


@cli.command("index")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config", "-c", "config_path",
    default="config.yaml",
    help="配置文件路径",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="强制重建索引（清空已有数据）",
)
def index_command(directory: str, config_path: str, force: bool):
    """索引指定目录下的文档（.md / .txt / .py）。

    DIRECTORY: 要索引的目录路径
    """
    config = load_config(config_path)
    index_cfg = config.get("index", {})
    scan_cfg = config.get("scan", {})
    embed_cfg = config.get("embedding", {})

    extensions = set(scan_cfg.get("extensions", [".md", ".txt", ".py"]))

    console.print(Panel.fit(
        f"[bold blue]📚 DocQA Agent — 文档索引[/]\n\n"
        f"目标目录: {directory}\n"
        f"文件类型: {', '.join(extensions)}\n"
        f"分块大小: {index_cfg.get('chunk_size', 500)}\n"
        f"嵌入模型: {embed_cfg.get('model', 'all-MiniLM-L6-v2')}",
    ))

    # 1. 加载文档
    loader = DocumentLoader(extensions=extensions)
    documents = loader.load_directory(directory)

    if not documents:
        console.print("[red]❌ 未找到任何可索引的文档[/]")
        return

    # 2. 分块
    chunker = DocumentChunker(
        chunk_size=int(index_cfg.get("chunk_size", 500)),
        chunk_overlap=int(index_cfg.get("chunk_overlap", 50)),
    )
    chunks = chunker.chunk_documents(documents)

    # 3. 向量存储
    from pathlib import Path
    persist_dir = index_cfg.get("persist_dir", "./data/faiss")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)  # 确保目录存在
    vector_store = VectorStore(
        persist_dir=persist_dir,
        embedding_model=embed_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=embed_cfg.get("device", "cpu"),
    )

    if force:
        vector_store.clear()

    v_count = vector_store.index_chunks(chunks)

    # 4. BM25 索引
    bm25_store = BM25Store()
    b_count = bm25_store.index_chunks(chunks)

    # 5. 汇总
    table = Table(title="索引完成")
    table.add_column("项目", style="cyan")
    table.add_column("数量", style="green")

    table.add_row("文档数", str(len(documents)))
    table.add_row("分块数", str(len(chunks)))
    table.add_row("向量索引", str(v_count))
    table.add_row("BM25 索引", str(b_count))
    table.add_row("持久化目录", str(Path(persist_dir).resolve()))

    console.print(table)
    console.print("[green]✅ 索引完成！现在可以用 `docqa ask` 或 `docqa chat` 提问[/]")


@cli.command("ask")
@click.argument("question", type=str)
@click.option(
    "--config", "-c", "config_path",
    default="config.yaml",
    help="配置文件路径",
)
@click.option(
    "--workspace", "-w",
    default=".",
    help="工作区目录（Agent 工具可访问的范围）",
)
def ask_command(question: str, config_path: str, workspace: str):
    """向 Agent 提问（单次问答）。

    QUESTION: 您的问题
    """
    config = load_config(config_path)
    agent = build_agent(config, workspace)

    with console.status("[bold green]Agent 正在思考..."):
        response = agent.ask(question)

    # 渲染回答
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="🤖 DocQA Agent",
        border_style="blue",
    ))

    # 渲染元信息
    meta_table = Table(title="回答元信息", show_header=False)
    meta_table.add_column("项目", style="cyan")
    meta_table.add_column("值", style="white")

    confidence_color = "green" if response.confidence >= 60 else "yellow" if response.confidence >= 30 else "red"
    meta_table.add_row("置信度", f"[{confidence_color}]{response.confidence}/100[/]")
    meta_table.add_row("迭代次数", str(response.iterations))
    meta_table.add_row("使用的工具", ", ".join(response.tool_calls_made) if response.tool_calls_made else "(无)")
    meta_table.add_row("状态", response.finish_reason)

    if response.sources:
        sources_text = "\n".join(f"  • {s}" for s in response.sources[:5])
        meta_table.add_row("引用来源", sources_text)

    console.print(meta_table)


@cli.command("chat")
@click.option(
    "--config", "-c", "config_path",
    default="config.yaml",
    help="配置文件路径",
)
@click.option(
    "--workspace", "-w",
    default=".",
    help="工作区目录",
)
def chat_command(config_path: str, workspace: str):
    """交互式对话模式（多轮对话）。"""
    config = load_config(config_path)
    agent = build_agent(config, workspace)

    console.print(Panel.fit(
        "[bold blue]🤖 DocQA Agent — 交互模式[/]\n\n"
        f"模型: {config.get('llm', {}).get('model', 'gpt-3.5-turbo')}\n"
        f"工作区: {Path(workspace).resolve()}\n"
        f"向量数: {agent.retriever.vector_store.count}\n\n"
        "[dim]输入问题开始对话，输入 /reset 清空历史，输入 /quit 退出[/]",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]你:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/]")
            break

        if not question:
            continue

        if question.lower() in ("/quit", "/exit", "/q"):
            console.print("[yellow]再见！[/]")
            break

        if question.lower() == "/reset":
            agent.reset_history()
            console.print("[green]✅ 对话历史已清空[/]")
            continue

        if question.lower() == "/sources":
            if hasattr(agent, "_last_sources"):
                for s in agent._last_sources:
                    console.print(f"  • {s}")
            else:
                console.print("[dim](无引用来源)[/]")
            continue

        with console.status("[bold green]Agent 正在思考..."):
            response = agent.ask(question)

        console.print()
        console.print(Panel(
            Markdown(response.answer),
            title="🤖 DocQA Agent",
            border_style="blue",
        ))

        # 简洁状态行
        conf_icon = "🟢" if response.confidence >= 60 else "🟡" if response.confidence >= 30 else "🔴"
        tools_info = f" | 🔧 {', '.join(response.tool_calls_made)}" if response.tool_calls_made else ""
        console.print(
            f"  {conf_icon} 置信度 {response.confidence}/100"
            f"{tools_info}"
            f"  [dim](迭代 {response.iterations})[/]"
        )


def main():
    """程序入口。"""
    cli()


if __name__ == "__main__":
    main()
