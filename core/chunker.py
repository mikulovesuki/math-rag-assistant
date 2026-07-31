"""Markdown 感知分块器 - 保护公式块和代码块不被切断"""
from __future__ import annotations

import re

from .models import Chunk


class MarkdownChunker:
    """
    将文本拆为原子单元后分组，确保公式块和代码块完整。

    原子单元类型:
      - $$...$$      数学公式块（不可分割）
      - ```...```    代码块（不可分割）
      - # / ## / ###  标题行（单独一个单元，并触发分块边界）
      - 空行分隔的段落（一个段落一个单元）

    每个 chunk 记录所属章节标题路径（heading），供检索引用到章节级。
    """

    _heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        units = self._split_into_units(text)
        if not units:
            return []
        return self._build_chunks(units, source)

    # ── 分块主逻辑 ──────────────────────────────────────

    def _build_chunks(self, units: list[str], source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        chunk_heading = ""
        heading_stack: list[tuple[int, str]] = []
        idx = 0

        def flush(with_overlap: bool = False) -> None:
            """封存当前块；with_overlap 时从尾部截取重叠单元"""
            nonlocal current, current_len, idx
            if not current:
                return
            chunks.append(self._make_chunk(current, source, idx, chunk_heading))
            idx += 1
            if with_overlap:
                current, current_len = self._take_overlap(current)
            else:
                current, current_len = [], 0

        for unit in units:
            # 标题单元：更新标题栈，标题作为新块的开头，保证 heading 准确
            m = self._heading_re.match(unit.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                heading_stack = [h for h in heading_stack if h[0] < level]
                heading_stack.append((level, title))
                flush(with_overlap=False)
                chunk_heading = " / ".join(t for _, t in heading_stack)
                current.append(unit)
                current_len += len(unit) + 1
                continue

            # 单个单元本身超长时，独立成块（不截断公式/代码）
            if len(unit) > self.chunk_size and current:
                flush(with_overlap=False)

            if current_len + len(unit) > self.chunk_size and current:
                flush(with_overlap=True)

            current.append(unit)
            current_len += len(unit) + 1  # +1 for newline

        if current:
            chunks.append(self._make_chunk(current, source, idx, chunk_heading))

        return chunks

    # ── 原子单元拆分 ──────────────────────────────────────

    @staticmethod
    def _split_into_units(text: str) -> list[str]:
        """将文本拆为不可分割的原子单元列表"""
        units: list[str] = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 数学公式块 $$...$$
            if stripped.startswith("$$"):
                block, i = MarkdownChunker._collect_block(lines, i, "$$")
                units.append(block)
                continue

            # 代码块 ```...```
            if stripped.startswith("```"):
                block, i = MarkdownChunker._collect_block(lines, i, "```")
                units.append(block)
                continue

            # 标题行
            if re.match(r"^#{1,6}\s", stripped):
                units.append(line)
                i += 1
                continue

            # 空行 - 跳过
            if not stripped:
                i += 1
                continue

            # 普通段落 - 累积到空行或特殊块为止
            para_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                if (not next_stripped
                        or next_stripped.startswith("$$")
                        or next_stripped.startswith("```")
                        or re.match(r"^#{1,6}\s", next_stripped)):
                    break
                para_lines.append(next_line)
                i += 1
            units.append("\n".join(para_lines))

        return units

    @staticmethod
    def _collect_block(lines: list[str], start: int, delimiter: str) -> tuple[str, int]:
        """收集从 start 开始的封闭块，返回 (块文本, 下一行索引)"""
        block_lines = [lines[start]]
        i = start + 1
        # 单行块: $$...$$ 同行结束
        if lines[start].strip().count(delimiter) >= 2:
            return block_lines[0], i

        while i < len(lines):
            block_lines.append(lines[i])
            if delimiter in lines[i]:
                i += 1
                break
            i += 1

        return "\n".join(block_lines), i

    # ── 辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _make_chunk(units: list[str], source: str, idx: int, heading: str = "") -> Chunk:
        return Chunk(
            text="\n\n".join(units),
            source=source,
            index=idx,
            heading=heading,
        )

    def _take_overlap(self, units: list[str]) -> tuple[list[str], int]:
        """
        从尾部保留不超过 overlap 字符的单元。
        至少保留一个单元，避免因单个超长单元导致 overlap 完全失效。
        """
        kept: list[str] = []
        total = 0
        for u in reversed(units):
            if kept and total + len(u) > self.overlap:
                break
            kept.insert(0, u)
            total += len(u) + 1
        return kept, total