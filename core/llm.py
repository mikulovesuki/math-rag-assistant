"""LLM 封装 - DeepSeek API（OpenAI 兼容 SDK）"""
from __future__ import annotations

from typing import Iterator

from .models import Chunk

# 延迟导入 config，避免循环依赖
import config

# 历史消息最多保留的轮数（每轮 2 条），控制上下文长度
MAX_HISTORY_ROUNDS = 6


class DeepSeekLLM:
    """
    通过 OpenAI 兼容 SDK 调用 DeepSeek。

    模型选择:
      - deepseek-chat      : 通用对话（快、便宜）
      - deepseek-reasoner  : 推理模型（适合复杂数学推导）
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or config.DEEPSEEK_API_KEY,
            base_url=base_url or config.DEEPSEEK_BASE_URL,
            timeout=timeout,
        )
        self.model = model or config.DEEPSEEK_MODEL

    def generate(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> str:
        """基于检索上下文生成回答（非流式）"""
        if not context:
            return "未检索到相关论文内容，无法回答。请先上传论文并构建索引。"

        messages = self._build_messages(query, context, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            stream=False,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """流式生成，逐段 yield 增量文本"""
        if not context:
            yield "未检索到相关论文内容，无法回答。请先上传论文并构建索引。"
            return

        messages = self._build_messages(query, context, history)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── 内部方法 ──────────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """组装 messages：system + 最近 N 轮历史 + 当前检索上下文"""
        messages: list[dict] = []
        if history:
            messages.extend(history[-MAX_HISTORY_ROUNDS * 2:])

        context_text = self._format_context(context)
        user_prompt = f"以下是检索到的论文片段：\n\n{context_text}\n\n问题：{query}"
        messages.append({"role": "user", "content": user_prompt})

        return [{"role": "system", "content": config.SYSTEM_PROMPT}] + messages

    @staticmethod
    def _format_context(chunks: list[Chunk]) -> str:
        """将检索结果格式化为 prompt 上下文（含章节级来源）"""
        parts: list[str] = []
        for i, c in enumerate(chunks, 1):
            ref = c.source
            if c.heading:
                ref += f" §{c.heading}"
            parts.append(f"--- 片段 {i} [来源: {ref}] (相关度: {c.score:.3f}) ---\n{c.text}")
        return "\n\n".join(parts)


class TemplateLLM:
    """无 API Key 时的模板 LLM，用于冒烟测试"""

    def generate(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> str:
        if not context:
            return f"未检索到与「{query}」相关的内容。"

        refs = "\n\n".join(
            f"[{i + 1}] 来源: {c.source}"
            + (f" §{c.heading}" if c.heading else "")
            + f" | 相关度: {c.score:.3f}\n{c.text[:200]}..."
            for i, c in enumerate(context)
        )
        return (
            f"问题: {query}\n"
            f"{'─' * 50}\n"
            f"检索到 {len(context)} 条相关论文片段:\n\n{refs}\n"
            f"{'─' * 50}\n"
            f"[模板输出] 配置 DEEPSEEK_API_KEY 后将调用 DeepSeek 生成回答"
        )

    def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """模板流式输出：一次性 yield 完整结果"""
        yield self.generate(query, context, history)