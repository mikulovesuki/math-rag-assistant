"""数据模型 - 贯穿 RAG 管线的核心数据结构"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Document:
    """原始文档 - 转换器输出，分块器输入"""
    text: str
    source: str = ""               # 文件名或标识


@dataclass
class Chunk:
    """文档分块 - 检索和生成的基本单元"""
    text: str
    source: str = ""
    index: int = 0                 # 在原文档中的块序号
    heading: str = ""              # 所属章节标题路径，如 "3.1 Attention"
    embedding: np.ndarray | None = None   # 由 Embedder 填充
    score: float = 0.0             # 检索相似度，由 Retriever 填充

    def to_dict(self) -> dict[str, Any]:
        """序列化为可持久化的字典（不含 embedding）"""
        return {
            "text": self.text,
            "source": self.source,
            "index": self.index,
            "heading": self.heading,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Chunk:
        """从字典反序列化（兼容缺少 heading 的旧数据）"""
        return Chunk(
            text=d["text"],
            source=d["source"],
            index=d["index"],
            heading=d.get("heading", ""),
        )