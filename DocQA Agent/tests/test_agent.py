"""Agent 模块单元测试。

测试范围：工具实现（search_code, read_file, list_dir）、Agent 循环逻辑。
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def workspace_dir():
    """创建模拟的工作区目录。"""
    tmp = tempfile.mkdtemp()

    files = {
        "README.md": "# Test Project\n\nThis is a test.\n\n## Install\n\n`pip install test`\n",
        "src/app.py": (
            "import os\n\n"
            "class App:\n"
            "    def run(self):\n"
            '        print("Running...")\n\n'
            "def main():\n"
            "    app = App()\n"
            "    app.run()\n"
        ),
        "src/utils.py": (
            "def helper():\n"
            '    """A helper function."""\n'
            "    return True\n"
        ),
        "tests/test_app.py": (
            "import pytest\n\n"
            "def test_app():\n"
            "    assert True\n"
        ),
        "docs/guide.md": (
            "# User Guide\n\n"
            "## Getting Started\n\n"
            "Follow these steps...\n"
        ),
    }

    for rel_path, content in files.items():
        full_path = Path(tmp) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    yield tmp
    shutil.rmtree(tmp)


# ================================================================
# 测试：search_code 工具
# ================================================================

class TestSearchCode:
    """测试 search_code 工具。"""

    def test_search_python_keyword(self, workspace_dir):
        """搜索 Python 关键词应返回匹配行。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern="class App", file_types=".py")
        assert result.success
        assert "class App" in result.output
        assert "app.py" in result.output

    def test_search_markdown(self, workspace_dir):
        """搜索 Markdown 文件。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern="Install", file_types=".md")
        assert result.success
        assert "Install" in result.output

    def test_search_regex(self, workspace_dir):
        """支持正则表达式搜索。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern=r"def\s+\w+", file_types=".py")
        assert result.success
        assert "def " in result.output

    def test_search_no_match(self, workspace_dir):
        """无匹配时应返回提示信息。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern="NONEXISTENT_PATTERN_XYZ")
        assert result.success  # 工具本身成功执行
        assert "未找到" in result.output

    def test_search_specific_file(self, workspace_dir):
        """指定文件路径搜索。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern="def", path="src/app.py")
        assert result.success
        assert "app.py" in result.output

    def test_max_results(self, workspace_dir):
        """应遵守 max_results 限制。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern=".", max_results=3)
        lines = result.output.split("\n")
        assert len([l for l in lines if l.strip()]) <= 3

    def test_security_path_traversal(self, workspace_dir):
        """应阻止路径遍历攻击。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.search_code(pattern="test", path="../../../etc")
        assert not result.success


# ================================================================
# 测试：read_file 工具
# ================================================================

class TestReadFile:
    """测试 read_file 工具。"""

    def test_read_entire_file(self, workspace_dir):
        """读取完整文件。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.read_file("README.md")
        assert result.success
        assert "# Test Project" in result.output
        assert "pip install test" in result.output

    def test_read_file_with_line_range(self, workspace_dir):
        """按行号范围读取。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.read_file("src/app.py", start_line=1, end_line=3)
        assert result.success
        lines = result.output.split("\n")
        assert len(lines) <= 3

    def test_read_nonexistent_file(self, workspace_dir):
        """读取不存在的文件应返回失败。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.read_file("nonexistent.py")
        assert not result.success

    def test_read_file_metadata(self, workspace_dir):
        """应包含文件元数据。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.read_file("README.md")
        assert result.metadata is not None
        assert "total_lines" in result.metadata

    def test_security_read_outside_workspace(self, workspace_dir):
        """应阻止读取工作区外的文件。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.read_file("../../../etc/passwd")
        assert not result.success


# ================================================================
# 测试：list_dir 工具
# ================================================================

class TestListDir:
    """测试 list_dir 工具。"""

    def test_list_root(self, workspace_dir):
        """列出根目录。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.list_dir()
        assert result.success
        assert "README.md" in result.output
        assert "src/" in result.output
        assert "docs/" in result.output

    def test_list_subdirectory(self, workspace_dir):
        """列出子目录。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.list_dir(path="src")
        assert result.success
        assert "app.py" in result.output
        assert "utils.py" in result.output

    def test_list_depth(self, workspace_dir):
        """depth=0 应仅列出当前层。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.list_dir(depth=0)
        assert result.success
        # depth=0 仅显示当前层的条目
        lines = result.output.split("\n")
        assert all("│" not in l for l in lines)  # 没有嵌套结构

    def test_list_nonexistent_directory(self, workspace_dir):
        """列出不存在的目录应返回失败。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.list_dir(path="nonexistent")
        assert not result.success

    def test_list_metadata(self, workspace_dir):
        """应包含目录元数据。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.list_dir()
        assert result.metadata is not None
        assert "path" in result.metadata
        assert "depth" in result.metadata


# ================================================================
# 测试：工具执行分发
# ================================================================

class TestToolExecution:
    """测试 execute 分发方法。"""

    def test_execute_search_code(self, workspace_dir):
        """execute('search_code', ...) 应正确分发。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.execute("search_code", {"pattern": "class", "file_types": ".py"})
        assert result.success
        assert result.tool_name == "search_code"

    def test_execute_read_file(self, workspace_dir):
        """execute('read_file', ...) 应正确分发。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.execute("read_file", {"file_path": "README.md"})
        assert result.success
        assert result.tool_name == "read_file"

    def test_execute_list_dir(self, workspace_dir):
        """execute('list_dir', ...) 应正确分发。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.execute("list_dir", {"path": ""})
        assert result.success
        assert result.tool_name == "list_dir"

    def test_execute_unknown_tool(self, workspace_dir):
        """未知工具应返回失败。"""
        from src.agent.tools import AgentTools
        tools = AgentTools(workspace_dir)

        result = tools.execute("unknown_tool", {})
        assert not result.success


# ================================================================
# 测试：工具定义
# ================================================================

class TestToolDefinitions:
    """测试工具定义的格式。"""

    def test_all_tools_have_name_description(self):
        """每个工具定义应有 name 和 description。"""
        from src.agent.tools import AgentTools
        defs = AgentTools.get_tool_definitions()

        for d in defs:
            assert d["type"] == "function"
            func = d["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["name"] in {"search_code", "read_file", "list_dir"}

    def test_search_code_requires_pattern(self):
        """search_code 的 required 参数应包含 pattern。"""
        from src.agent.tools import AgentTools
        defs = AgentTools.get_tool_definitions()

        search_def = next(d for d in defs if d["function"]["name"] == "search_code")
        assert "pattern" in search_def["function"]["parameters"]["required"]


# ================================================================
# 测试：Agent runLoop（Mock LLM）
# ================================================================

class TestDocQAAgent:
    """测试 DocQAAgent 核心逻辑（使用 Mock）。"""

    def _mock_llm_response(self, content="Mock answer", tool_calls=None, finish_reason="stop"):
        """创建模拟的 LLM 响应。"""
        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": {"total_tokens": 100},
        }

    def test_agent_ask_returns_response(self):
        """Agent 应能返回回答。"""
        from src.agent.loop import DocQAAgent

        # 创建 mock 对象
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        # side_effect: 第一次调用返回主回答，第二次调用返回置信度评分
        mock_llm.chat.side_effect = [
            {
                "content": "这是一个模拟的回答。",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {},
            },
            {
                "content": "85",  # 置信度评分 >= 阈值
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {},
            },
        ]

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
            tools=None,
            confidence_threshold=60,
            max_iterations=3,
        )

        response = agent.ask("测试问题")
        assert "这是一个模拟的回答。" in response.answer
        assert response.finish_reason in ("answered", "low_confidence")

    def test_agent_calls_tools_when_needed(self):
        """当 LLM 返回 tool_calls 时，Agent 应执行工具。"""
        from src.agent.loop import DocQAAgent
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        # 第一次调用返回 tool_calls，第二次返回最终回答
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            {
                "content": None,
                "tool_calls": [{"id": "1", "name": "list_dir", "arguments": {"path": ""}}],
                "finish_reason": "tool_calls",
                "usage": {},
            },
            {
                "content": "根据工具结果，这是一个项目。",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {},
            },
        ]

        mock_tools = MagicMock()
        mock_tools.get_tool_definitions.return_value = []
        mock_tools.execute.return_value = MagicMock(
            success=True, output="README.md\nsrc/\n", tool_name="list_dir"
        )

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
            tools=mock_tools,
            confidence_threshold=60,
            max_iterations=5,
        )

        response = agent.ask("这个项目有什么文件？")
        assert response.tool_calls_made
        assert "list_dir" in response.tool_calls_made

    def test_agent_history(self):
        """Agent 应维护对话历史。"""
        from src.agent.loop import DocQAAgent

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "content": "回答",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {},
        }

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
            tools=None,
            max_history=5,
        )

        assert len(agent._history) == 0
        agent.ask("问题1")
        assert len(agent._history) == 2  # user + agent

    def test_agent_reset_history(self):
        """reset_history 应清空历史。"""
        from src.agent.loop import DocQAAgent

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "content": "回答",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {},
        }

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        agent.ask("问题1")
        agent.reset_history()
        assert len(agent._history) == 0

    def test_agent_max_iterations(self):
        """超过最大迭代次数应返回兜底回答。"""
        from src.agent.loop import DocQAAgent

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        # LLM 一直返回 tool_calls，触发迭代耗尽
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "content": None,
            "tool_calls": [{"id": "1", "name": "list_dir", "arguments": {"path": ""}}],
            "finish_reason": "tool_calls",
            "usage": {},
        }

        mock_tools = MagicMock()
        mock_tools.get_tool_definitions.return_value = []
        mock_tools.execute.return_value = MagicMock(
            success=True, output="test", tool_name="list_dir"
        )

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
            tools=mock_tools,
            max_iterations=2,
        )

        response = agent.ask("测试")
        assert response.finish_reason == "max_iterations"
        assert response.iterations == 2

    def test_reflection_detection(self):
        """应能检测反思请求。"""
        from src.agent.loop import DocQAAgent

        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        agent = DocQAAgent(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        assert agent._is_reflection_request("不对，重新查")
        assert agent._is_reflection_request("这个答案不正确")
        assert agent._is_reflection_request("再查一下")
        assert not agent._is_reflection_request("这个项目的安装方法是什么？")
