"""向量存储 - numpy 内存检索 + 磁盘持久化"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .models import Chunk


class PersistentVectorStore:
    """
    numpy 向量库，支持磁盘 save/load。

    持久化文件:
      - {dir}/vectors.npy  : numpy 矩阵 (N, dim)
      - {dir}/chunks.pkl   : chunk 元数据列表
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """添加带 embedding 的 chunk 到向量库"""
        self._chunks.extend(chunks)
        new_vecs = np.array([c.embedding for c in chunks], dtype=np.float32)
        self._matrix = new_vecs if self._matrix is None else np.vstack([self._matrix, new_vecs])

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[Chunk]:
        """cosine 相似度检索，返回 Top-K 结果（带 score）"""
        if self._matrix is None or not self._chunks:
            return []

        scores = self._matrix @ query_vec
        top_k = min(top_k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: list[Chunk] = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            results.append(Chunk(
                text=self._chunks[i].text,
                source=self._chunks[i].source,
                index=self._chunks[i].index,
                score=float(scores[i]),
            ))
        return results

    def clear(self) -> None:
        """清空向量库"""
        self._chunks.clear()
        self._matrix = None

    def save(self, dir_path: str | Path) -> None:
        """持久化到磁盘"""
        directory = Path(dir_path)
        directory.mkdir(parents=True, exist_ok=True)

        if self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)

        # 只存元数据，不存 embedding（向量单独存）
        chunk_dicts = [c.to_dict() for c in self._chunks]
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(chunk_dicts, f)

    def load(self, dir_path: str | Path) -> bool:
        """从磁盘加载，返回是否成功"""
        directory = Path(dir_path)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.pkl"

        if not vectors_path.exists() or not chunks_path.exists():
            return False

        self._matrix = np.load(vectors_path)
        with open(chunks_path, "rb") as f:
            chunk_dicts = pickle.load(f)
        self._chunks = [Chunk.from_dict(d) for d in chunk_dicts]
        return True

    @property
    def size(self) -> int:
        return len(self._chunks)