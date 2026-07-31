"""向量存储 - numpy 内存检索 + 磁盘持久化"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Chunk


class PersistentVectorStore:
    """
    numpy 向量库，支持磁盘 save/load。

    持久化文件:
      - {dir}/vectors.npy     : numpy 矩阵 (N, dim)
      - {dir}/chunks.json     : chunk 元数据列表（JSON，纯文本）
      - {dir}/index_meta.json : 索引元数据（模型名/维度/来源文件/时间）

    加载时校验元数据，模型或维度变化时拒绝加载，避免静默出错。
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self.meta: dict = {}

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

        # 先过滤非正相似度，再取 Top-K，保证返回条数完整
        valid_idx = np.flatnonzero(scores > 0)
        if len(valid_idx) == 0:
            return []

        top_k = min(top_k, len(valid_idx))
        top_valid = valid_idx[np.argsort(scores[valid_idx])[::-1][:top_k]]

        results: list[Chunk] = []
        for i in top_valid:
            results.append(Chunk(
                text=self._chunks[i].text,
                source=self._chunks[i].source,
                index=self._chunks[i].index,
                heading=self._chunks[i].heading,
                score=float(scores[i]),
            ))
        return results

    def cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """返回所有 chunk 与查询向量的 cosine 分数（N,）"""
        if self._matrix is None or not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return self._matrix @ query_vec

    def get_chunk(self, index: int) -> Chunk:
        """按内部索引取 chunk 副本（检索打分用，不改动库内数据）"""
        c = self._chunks[index]
        return Chunk(
            text=c.text,
            source=c.source,
            index=c.index,
            heading=c.heading,
        )

    def clear(self) -> None:
        """清空向量库"""
        self._chunks.clear()
        self._matrix = None
        self.meta = {}

    # ── 持久化 ──────────────────────────────────────────

    def save(self, dir_path: str | Path, meta: dict | None = None) -> None:
        """持久化到磁盘。meta 用于记录索引来源（模型名/文件列表等）"""
        directory = Path(dir_path)
        directory.mkdir(parents=True, exist_ok=True)

        if self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)

        # 只存元数据，不存 embedding（向量单独存）
        chunk_dicts = [c.to_dict() for c in self._chunks]
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=1)

        # 索引元数据：维度/模型/来源/时间，供加载时校验
        index_meta = {
            "chunk_count": len(self._chunks),
            "dim": self._matrix.shape[1] if self._matrix is not None else 0,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if meta:
            index_meta.update(meta)
        self.meta = index_meta
        with open(directory / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(index_meta, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str | Path, expected: dict | None = None) -> tuple[bool, str]:
        """
        从磁盘加载，返回 (是否成功, 说明/错误原因)。

        expected 可提供期望约束，如 {"model_name": "...", "dim": 512}，
        与索引元数据不一致时拒绝加载（提示重建索引）。
        """
        directory = Path(dir_path)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.json"
        meta_path = directory / "index_meta.json"

        if not vectors_path.exists() or not chunks_path.exists():
            return False, "索引文件不存在"

        self._matrix = np.load(vectors_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        self._chunks = [Chunk.from_dict(d) for d in chunk_dicts]

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        # 版本校验：模型或维度与当前配置不一致时拒绝加载
        if expected:
            meta_model = self.meta.get("model_name")
            if meta_model and expected.get("model_name") and meta_model != expected["model_name"]:
                return False, (
                    f"索引由模型 {meta_model} 构建，当前配置为 {expected['model_name']}。"
                    "请重新构建索引"
                )
            meta_dim = self.meta.get("dim")
            if meta_dim and expected.get("dim") and meta_dim != expected["dim"]:
                return False, (
                    f"索引维度 {meta_dim} 与当前模型维度 {expected['dim']} 不一致。"
                    "请重新构建索引"
                )

        return True, "索引加载成功"

    # ── 属性 ────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> set[str]:
        """索引覆盖的来源文件集合"""
        return {c.source for c in self._chunks}