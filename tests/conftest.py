"""pytest 共享 fixture - 离线可跑,不依赖模型下载与外部服务"""
from __future__ import annotations

import numpy as np
import pytest

from core.models import Chunk, Document


class StubEmbedder:
    """确定性向量:按文本字符映射到固定维度,保证相同文本得到相同向量"""
    model_name = "stub-embedder"
    _dim = 8

    def embed(self, text: str, query: bool = False) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) % 7 + 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim


class StubConverter:
    """把 txt 文件内容读为 Document,无需 MarkItDown"""

    def convert(self, file_path) -> Document:
        from pathlib import Path
        path = Path(file_path)
        return Document(text=path.read_text(encoding="utf-8"), source=path.name)

    def convert_directory(self, dir_path) -> list[Document]:
        from pathlib import Path
        docs = []
        for path in sorted(Path(dir_path).iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                docs.append(self.convert(path))
        return docs


class StubLLM:
    """固定回答 + 流式分段输出,用于验证管线编排"""

    def generate(self, query: str, context: list[Chunk], history: list[dict] | None = None) -> str:
        return f"答案[{query}] 基于 {len(context)} 个片段"

    def generate_stream(self, query: str, context: list[Chunk], history: list[dict] | None = None):
        full = self.generate(query, context, history)
        for i in range(0, len(full), 4):
            yield full[i:i + 4]


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """带 heading 的示例分块"""
    return [
        Chunk(text="transformer 使用多头注意力机制 self-attention", source="a.txt", index=0, heading="3.1"),
        Chunk(text="位置编码 positional encoding 加入序列信息", source="a.txt", index=1, heading="3.2"),
        Chunk(text="前馈网络 feed-forward 逐位置变换", source="b.txt", index=0, heading="4"),
    ]