"""Agent 工具集 —— search_code / read_file / list_dir。

这些工具让 Agent 可以在检索结果不足时主动探索文件系统。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from src import logger


@dataclass
class ToolResult:
    """工具调用结果。"""

    success: bool
    output: str
    tool_name: str
    metadata: dict | None = None


class AgentTools:
    """提供 search_code、read_file、list_dir 三个工具。"""

    def __init__(self, workspace_dir: str | Path):
        """
        Args:
            workspace_dir: 工作区根目录，所有工具操作限制在此目录下
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        if not self.workspace_dir.is_dir():
            raise ValueError(f"工作区目录不存在: {self.workspace_dir}")

    # ---- 工具定义（用于 Function Calling）----

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """返回 OpenAI Function Calling 格式的工具定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": (
                        "在代码/文档文件中搜索关键词（类似 grep）。"
                        "支持正则表达式，返回匹配行及其上下文。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "搜索关键词或正则表达式",
                            },
                            "path": {
                                "type": "string",
                                "description": "搜索的目录或文件路径（相对于工作区根目录，为空则搜索整个工作区）",
                                "default": "",
                            },
                            "file_types": {
                                "type": "string",
                                "description": "逗号分隔的文件扩展名，如 '.py,.md,.txt'",
                                "default": ".py,.md,.txt,.json,.yaml,.yml,.toml,.cfg,.ini,.js,.ts,.html,.css",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "最多返回条数",
                                "default": 20,
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取指定文件的完整内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文件路径（相对于工作区根目录）",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "起始行号（从1开始），为空则从头读取",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "结束行号（含），为空则读到末尾",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "列出目录结构",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "目录路径（相对于工作区根目录，为空则列出根目录）",
                                "default": "",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "递归深度（0=仅当前层）",
                                "default": 1,
                            },
                        },
                    },
                },
            },
        ]

    # ---- 工具实现 ----

    def _resolve_path(self, relative_path: str) -> Path:
        """将相对路径解析为绝对路径，并验证在工作区内。"""
        p = (self.workspace_dir / relative_path).resolve() if relative_path else self.workspace_dir
        # 安全检查：确保路径在工作区范围内
        try:
            p.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(f"路径不在工作区范围内: {relative_path}")
        return p

    def search_code(
        self,
        pattern: str,
        path: str = "",
        file_types: str = ".py,.md,.txt,.json,.yaml,.yml,.toml,.cfg,.ini,.js,.ts,.html,.css",
        max_results: int = 20,
    ) -> ToolResult:
        """在文件中搜索关键词（类似 grep）。

        Args:
            pattern: 搜索关键词或正则表达式
            path: 搜索的目录或文件路径
            file_types: 逗号分隔的文件扩展名
            max_results: 最多返回条数

        Returns:
            ToolResult
        """
        try:
            target = self._resolve_path(path)
            extensions = set(ext.strip() for ext in file_types.split(","))

            compiled = re.compile(pattern, re.IGNORECASE)
            matches: list[str] = []

            if target.is_file():
                files = [target]
            else:
                files = []
                for root, _, filenames in os.walk(target):
                    for fname in filenames:
                        fpath = Path(root) / fname
                        if fpath.suffix.lower() in extensions:
                            files.append(fpath)

            for fpath in files:
                if len(matches) >= max_results:
                    break
                try:
                    lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue

                for i, line in enumerate(lines, start=1):
                    if len(matches) >= max_results:
                        break
                    if compiled.search(line):
                        rel_path = fpath.relative_to(self.workspace_dir)
                        matches.append(f"{rel_path}:{i}: {line.strip()[:200]}")

            output = "\n".join(matches) if matches else "(未找到匹配结果)"
            logger.info("search_code('%s') → %d 条结果", pattern, len(matches))

            return ToolResult(
                success=True,
                output=output,
                tool_name="search_code",
                metadata={"pattern": pattern, "match_count": len(matches)},
            )

        except (PermissionError, ValueError) as e:
            return ToolResult(success=False, output=str(e), tool_name="search_code")

    def read_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        """读取文件内容。

        Args:
            file_path: 文件路径（相对于工作区）
            start_line: 起始行号（从1开始）
            end_line: 结束行号（含）

        Returns:
            ToolResult
        """
        try:
            target = self._resolve_path(file_path)
            if not target.is_file():
                return ToolResult(
                    success=False,
                    output=f"文件不存在: {file_path}",
                    tool_name="read_file",
                )

            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
            total_lines = len(lines)

            s = max(1, start_line or 1) - 1  # 转 0-based
            e = min(total_lines, end_line or total_lines)

            selected = lines[s:e]
            # 加行号
            numbered = [f"{s+i+1:4d}| {line}" for i, line in enumerate(selected)]

            # 限制输出长度（最多250行）
            if len(numbered) > 250:
                numbered = numbered[:250]
                numbered.append(f"... (共 {e - s} 行，仅显示前 250 行)")

            output = "\n".join(numbered)
            logger.info("read_file('%s', L%d-L%d) → %d 行", file_path, s + 1, e, len(numbered))

            return ToolResult(
                success=True,
                output=output,
                tool_name="read_file",
                metadata={
                    "file_path": str(target),
                    "total_lines": total_lines,
                    "shown_lines": len(numbered),
                },
            )

        except (PermissionError, ValueError) as e:
            return ToolResult(success=False, output=str(e), tool_name="read_file")

    def list_dir(self, path: str = "", depth: int = 1) -> ToolResult:
        """列出目录结构。

        Args:
            path: 目录路径（相对于工作区）
            depth: 递归深度

        Returns:
            ToolResult
        """
        try:
            target = self._resolve_path(path) if path else self.workspace_dir
            if not target.is_dir():
                return ToolResult(
                    success=False,
                    output=f"目录不存在: {path}",
                    tool_name="list_dir",
                )

            lines = self._render_tree(target, max_depth=depth, prefix="")
            output = "\n".join(lines) if lines else "(空目录)"
            logger.info("list_dir('%s', depth=%d) → %d 行", path or ".", depth, len(lines))

            return ToolResult(
                success=True,
                output=output,
                tool_name="list_dir",
                metadata={"path": str(target), "depth": depth},
            )

        except (PermissionError, ValueError) as e:
            return ToolResult(success=False, output=str(e), tool_name="list_dir")

    def _render_tree(self, directory: Path, max_depth: int, prefix: str) -> list[str]:
        """递归渲染目录树。"""
        if max_depth < 0:
            return []

        lines: list[str] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return [f"{prefix}[无权限] {directory.name}"]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            name = entry.name
            if entry.is_dir():
                name += "/"

            lines.append(f"{prefix}{connector}{name}")

            if entry.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                lines.extend(self._render_tree(entry, max_depth - 1, next_prefix))

        return lines

    # ---- 工具调用分发 ----

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """根据工具名称和参数执行工具。

        Args:
            tool_name: 工具名称
            arguments: 参数字典

        Returns:
            ToolResult
        """
        if tool_name == "search_code":
            return self.search_code(**arguments)
        elif tool_name == "read_file":
            return self.read_file(**arguments)
        elif tool_name == "list_dir":
            return self.list_dir(**arguments)
        else:
            return ToolResult(
                success=False,
                output=f"未知工具: {tool_name}",
                tool_name=tool_name,
            )
