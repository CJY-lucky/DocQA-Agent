"""Agent 主循环（runLoop）—— 多轮对话、工具调用、置信度评估、反思机制。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import logger
from src.agent.tools import AgentTools, ToolResult
from src.llm.client import LLMClient
from src.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class AgentResponse:
    """Agent 的单次响应。"""

    answer: str
    confidence: int  # 0-100
    sources: list[str]  # 引用来源（文件名+片段）
    tool_calls_made: list[str]  # 本次调用了哪些工具
    iterations: int  # 本次消耗的迭代数
    finish_reason: str  # "answered" | "low_confidence" | "tool_used" | "max_iterations"


@dataclass
class ConversationTurn:
    """对话的一轮。"""

    role: str  # "user" | "assistant"
    content: str


class DocQAAgent:
    """文档问答 Agent —— 核心 runLoop 实现。

    流程：
    1. 检索阶段：混合检索（向量 + BM25）→ Top-K 片段
    2. 工具调用：若检索不足，LLM 可调用 search/read/list 工具
    3. LLM 生成：基于检索结果 + 工具结果生成回答
    4. 置信度评估：LLM 自评 0-100，低于阈值则追问
    5. 反思机制：用户反馈"不对"时，换策略重新回答
    """

    SYSTEM_PROMPT = """你是一个专业的技术文档问答助手（DocQA Agent）。

## 你的能力
1. 基于提供的文档片段回答问题
2. 如果信息不足，可以调用工具探索文件系统：
   - search_code: 在文件中搜索关键词
   - read_file: 读取指定文件内容
   - list_dir: 列出目录结构
3. 对每个答案评估置信度（0-100）

## 规则
- 根据提供的文档上下文回答，不编造信息
- 如果文档不足以回答，优先调用工具获取更多信息
- 用中文回答（除非用户使用其他语言）
- 引用具体的文件名和来源"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        tools: AgentTools | None = None,
        confidence_threshold: int = 60,
        max_iterations: int = 5,
        max_history: int = 10,
    ):
        """
        Args:
            retriever: 混合检索引擎
            llm_client: LLM 客户端
            tools: Agent 工具集
            confidence_threshold: 置信度阈值（低于此值会追问用户）
            max_iterations: 单次回答的最大迭代次数
            max_history: 最大保留的历史对话轮数
        """
        self.retriever = retriever
        self.llm = llm_client
        self.tools = tools
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.max_history = max_history
        self._history: list[ConversationTurn] = []

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> AgentResponse:
        """用户提问，Agent 返回回答。

        Args:
            question: 用户问题

        Returns:
            AgentResponse（包含答案、置信度、来源等）
        """
        logger.info("用户提问: %s", question[:100])

        # 检测是否为反思请求
        if self._is_reflection_request(question):
            return self._handle_reflection(question)

        response = self._run_loop(question)
        self._update_history("user", question)
        self._update_history("assistant", response.answer)
        return response

    def reset_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()
        logger.info("对话历史已清空")

    # ------------------------------------------------------------------
    # 核心 runLoop
    # ------------------------------------------------------------------

    def _run_loop(self, question: str) -> AgentResponse:
        """核心 runLoop：检索 → [工具调用] → 生成 → 置信度评估。

        使用标准的 OpenAI Function Calling 模式：每次工具调用后，
        在同一对话流中把工具结果回传给 LLM，而非重新开始一轮。
        """
        tool_calls_made: list[str] = []
        seen_tool_calls: set[tuple[str, str]] = set()
        file_read_count: dict[str, int] = {}  # 跟踪同一文件被读取的次数

        # Step 1: 混合检索（只执行一次）
        retrieval_results = self.retriever.search(question, top_k=5)
        context_chunks = [r.content for r in retrieval_results]
        sources = [
            f"{r.file_name} (chunk {r.chunk_index}, score={r.score})"
            for r in retrieval_results
        ]

        # Step 2: 构建初始消息（作为持续对话的起点）
        messages = self._build_messages(question, context_chunks)
        tool_defs = self.tools.get_tool_definitions() if self.tools else None

        for iteration in range(1, self.max_iterations + 1):
            logger.info("--- 迭代 %d/%d ---", iteration, self.max_iterations)

            result = self.llm.chat(messages, tools=tool_defs)

            # Step 3: 处理工具调用
            if result.get("tool_calls"):
                # 过滤掉已见过的工具调用（防止死循环）
                new_calls = []
                for tc in result["tool_calls"]:
                    key = (tc["name"], str(sorted(tc.get("arguments", {}).items())))
                    if key not in seen_tool_calls:
                        seen_tool_calls.add(key)
                        new_calls.append(tc)
                    else:
                        logger.warning("跳过重复工具调用: %s(%s)", tc["name"], tc.get("arguments", {}))

                # 如果所有调用都是重复的，追加指令强制 LLM 直接回答
                if not new_calls:
                    messages.append({
                        "role": "user",
                        "content": (
                            "你已经多次调用相同的工具且没有获得新信息。"
                            "请基于已有的检索结果和工具输出直接回答问题，"
                            "如果信息确实不足，请诚实说明并给出你已经了解的部分。"
                        ),
                    })
                    continue

                # 检查是否反复读同一个文件
                for tc in result["tool_calls"]:
                    if tc["name"] == "read_file":
                        fname_key = tc.get("arguments", {}).get("file_path", "")
                        file_read_count[fname_key] = file_read_count.get(fname_key, 0) + 1
                        if file_read_count[fname_key] >= 3:
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"你已经读了 `{fname_key}` {file_read_count[fname_key]} 次。"
                                    "如果还需要找具体信息，请用 search_code 搜索关键模式"
                                    "（如 'class AgentLoop' 或 'run' 或 'async'），"
                                    "而不是继续逐段读同一个文件。"
                                    "如果已有足够信息，请直接回答。"
                                ),
                            })
                            break  # 发送一条就够了

                # 把 LLM 的 tool_calls 消息加入对话
                tool_call_msgs = []
                for tc in result["tool_calls"]:
                    tool_call_msgs.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                        },
                    })

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_call_msgs,
                })

                # 执行工具并把每个结果加入对话（正确匹配 tool_call_id）
                for tc in result["tool_calls"]:
                    tr = self.tools.execute(tc["name"], tc.get("arguments", {}))
                    tool_calls_made.append(tr.tool_name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tr.output[:2000],
                    })

                continue  # 回到 LLM 对话循环

            # Step 4: 无工具调用 → LLM 直接给了回答
            answer = result.get("content", "")
            if not answer or not answer.strip():
                continue

            # Step 5: 置信度评估
            confidence = self._evaluate_confidence(question, answer, context_chunks)

            finish_reason = "answered"
            if confidence < self.confidence_threshold:
                finish_reason = "low_confidence"
                answer = self._build_low_confidence_response(answer, confidence)

            return AgentResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                tool_calls_made=tool_calls_made,
                iterations=iteration,
                finish_reason=finish_reason,
            )

        # 超过最大迭代次数
        return AgentResponse(
            answer="抱歉，我在多次尝试后仍无法找到足够的信息来回答您的问题。"
                   "请尝试更具体地描述您的问题，或者检查索引的文档是否包含相关内容。",
            confidence=0,
            sources=sources,
            tool_calls_made=tool_calls_made,
            iterations=self.max_iterations,
            finish_reason="max_iterations",
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        context_chunks: list[str],
    ) -> list[dict[str, str]]:
        """构建发给 LLM 的消息列表。"""
        context_text = "\n\n---\n\n".join(
            f"[片段 {i+1}] {chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        user_content = f"""## 文档上下文

{context_text if context_text else "(未找到相关文档片段)"}

## 用户问题

{question}

## 指示
1. 先阅读文档上下文，判断是否能回答问题
2. 如果信息充分，请直接回答（引用具体来源）
3. 如果信息不足，请调用合适的工具（search_code / read_file / list_dir）获取更多信息
4. 不要编造任何信息"""

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # 插入历史对话（最近 N 轮）
        recent_history = self._history[-(self.max_history * 2):]
        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _execute_tools(self, tool_calls: list[dict]) -> list[ToolResult]:
        """执行 LLM 请求的工具调用。"""
        if not self.tools:
            return [ToolResult(success=False, output="工具未初始化", tool_name="none")]

        results: list[ToolResult] = []
        for tc in tool_calls:
            logger.info("执行工具: %s(%s)", tc["name"], tc["arguments"])
            result = self.tools.execute(tc["name"], tc.get("arguments", {}))
            results.append(result)
        return results

    def _evaluate_confidence(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
    ) -> int:
        """让 LLM 自评回答的置信度。

        Returns:
            0-100 的置信度分数
        """
        eval_prompt = f"""请对以下问答的置信度进行评分（0-100整数）：

## 问题
{question[:500]}

## 回答
{answer[:500]}

## 评分标准
- 90-100: 答案完全基于文档内容，有明确引用
- 70-89: 答案大部分基于文档，少量合理推断
- 50-69: 答案部分基于文档，有一定不确定性
- 30-49: 答案主要基于常识，文档支持不足
- 0-29: 答案完全是猜测或无法回答

请只返回一个 0-100 的整数，不要包含其他文字。"""

        try:
            result = self.llm.chat([
                {"role": "system", "content": "你是一个评分助手。请只返回一个0-100的整数。"},
                {"role": "user", "content": eval_prompt},
            ])
            content = result.get("content", "").strip()
            # 提取数字
            import re
            match = re.search(r'\d+', content)
            if match:
                score = int(match.group())
                return max(0, min(100, score))
        except Exception as e:
            logger.warning("置信度评估失败: %s", e)

        # 默认：有上下文就给中等分数
        return 60 if context_chunks else 20

    def _build_low_confidence_response(self, answer: str, confidence: int) -> str:
        """构建低置信度回复，主动追问用户。"""
        return (
            f"{answer}\n\n"
            f"---\n"
            f"⚠️ **置信度: {confidence}/100**（较低）\n\n"
            f"我不太确定上述回答是否准确。以下信息可以帮助改进回答：\n"
            f"- 您能否提供更多相关上下文或关键词？\n"
            f"- 您期望的答案是关于哪个具体文件/模块的？\n"
            f"- 是否需要我搜索特定的代码或文档？"
        )

    def _is_reflection_request(self, question: str) -> bool:
        """检测用户是否在反馈"答案不对"。"""
        reflection_keywords = [
            "不对", "错了", "不正确", "再查", "重新", "换一种",
            "no", "wrong", "incorrect", "try again",
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in reflection_keywords)

    def _handle_reflection(self, question: str) -> AgentResponse:
        """反思机制：用户反馈答案不对时，换一种检索策略重试。"""
        logger.info("触发反思机制: %s", question[:50])

        # 策略1: 只用关键词检索 (纯 BM25)
        bm25_results = self.retriever.bm25_store.search(question, top_k=5)
        # 策略2: 只用语义检索 (纯 Vector)
        vector_results = self.retriever.vector_store.search(question, top_k=5)

        # 合并去重
        seen = set()
        all_context: list[str] = []
        all_sources: list[str] = []
        for r in bm25_results + vector_results:
            key = f"{r.file_path}:{r.chunk_index}"
            if key not in seen:
                seen.add(key)
                all_context.append(r.content)
                all_sources.append(f"{r.file_name} (score={r.score}, source={r.source})")

        # 额外：如果工具可用，尝试 search_code
        if self.tools:
            code_result = self.tools.search_code(question)
            if code_result.success and code_result.output != "(未找到匹配结果)":
                all_context.append(f"[代码搜索结果]\n{code_result.output}")

        messages = self._build_messages(question, all_context[:10])
        # 添加反思指令
        messages.append({
            "role": "user",
            "content": (
                "之前的回答可能不准确，请基于**以上新检索结果**重新回答。"
                "请注意：\n"
                "1. 对比之前可能的错误，明确指出差异\n"
                "2. 如果仍然不确定，诚实说明"
            ),
        })

        result = self.llm.chat(messages)
        answer = result.get("content", "无法生成回答")
        confidence = self._evaluate_confidence(question, answer, all_context[:5])

        self._update_history("user", question)
        self._update_history("assistant", answer)

        return AgentResponse(
            answer=answer,
            confidence=confidence,
            sources=all_sources,
            tool_calls_made=["反思-多策略检索"],
            iterations=1,
            finish_reason="answered",
        )

    def _update_history(self, role: str, content: str) -> None:
        """更新对话历史。"""
        self._history.append(ConversationTurn(role=role, content=content))
        # 截断
        max_turns = self.max_history * 2
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]
