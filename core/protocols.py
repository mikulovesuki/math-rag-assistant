"""组件契约 - RAG 各模块的可替换接口（typing.Protocol，结构子类型）"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .models import Chunk, Document


@runtime_checkable
class Embedder(Protocol):
    """向量化组件：文本 -> L2 归一化向量"""
    model_name: str

    def embed(self, text: str, query: bool = False) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库：存储与检索（含磁盘持久化）"""
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query_vec: np.ndarray, top_k: int) -> list[Chunk]: ...
    def clear(self) -> None: ...
    def save(self, dir_path, meta: dict | None = None) -> None: ...
    def load(self, dir_path, expected: dict | None = None) -> tuple[bool, str]: ...
    @property
    def size(self) -> int: ...
    @property
    def sources(self) -> set[str]: ...


@runtime_checkable
class LLM(Protocol):
    """生成组件：基于检索上下文生成回答（支持多轮历史与流式）"""
    def generate(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ) -> str: ...
    def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        history: list[dict] | None = None,
    ): ...


@runtime_checkable
class Chunker(Protocol):
    """分块组件：文档文本 -> chunk 列表"""
    def chunk(self, text: str, source: str = "") -> list[Chunk]: ...


@runtime_checkable
class Converter(Protocol):
    """文档转换组件：文件 -> Document"""
    def convert(self, file_path) -> Document: ...
    def convert_directory(self, dir_path) -> list[Document]: ...