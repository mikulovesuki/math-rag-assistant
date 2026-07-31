"""向量库测试:增删查、JSON 持久化、版本校验"""
from __future__ import annotations

import json
import numpy as np
import pytest

from core.models import Chunk
from core.vector_store import PersistentVectorStore


def _chunk(text: str, source: str = "a.txt", heading: str = "") -> Chunk:
    return Chunk(text=text, source=source, heading=heading)


def test_add_search_roundtrip():
    store = PersistentVectorStore()
    chunks = [
        _chunk("hello world", heading="1"),
        _chunk("foo bar", heading="2"),
    ]
    chunks[0].embedding = np.array([1.0, 0.0], dtype=np.float32)
    chunks[1].embedding = np.array([0.0, 1.0], dtype=np.float32)
    store.add(chunks)

    assert store.size == 2
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].text == "hello world"
    assert results[0].score > 0.99
    assert results[0].heading == "1"


def test_search_filters_negative_scores_first():
    """先过滤非正相似度再取 top-k,保证返回条数完整"""
    store = PersistentVectorStore()
    chunks = [
        _chunk("positive"),
        _chunk("negative"),
        _chunk("positive2"),
    ]
    chunks[0].embedding = np.array([1.0, 0.0], dtype=np.float32)
    chunks[1].embedding = np.array([-1.0, 0.0], dtype=np.float32)
    chunks[2].embedding = np.array([0.8, 0.0], dtype=np.float32)
    store.add(chunks)

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
    assert len(results) == 2, "负相似度 chunk 应被过滤"


def test_save_load_json(tmp_path):
    """持久化为 JSON 纯文本,加载后检索一致"""
    store = PersistentVectorStore()
    chunks = [
        _chunk("hello world", heading="3.1"),
        _chunk("foo bar", heading="3.2"),
    ]
    chunks[0].embedding = np.array([1.0, 0.0], dtype=np.float32)
    chunks[1].embedding = np.array([0.0, 1.0], dtype=np.float32)
    store.add(chunks)
    store.save(tmp_path, meta={"model_name": "stub", "files": ["a.txt"]})

    # 持久化文件齐全
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "index_meta.json").exists()

    # chunks.json 是纯文本(无 pickle)
    with open(tmp_path / "chunks.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw[0]["heading"] == "3.1"
    assert "embedding" not in raw[0]

    store2 = PersistentVectorStore()
    ok, reason = store2.load(tmp_path)
    assert ok, reason
    assert store2.size == 2
    assert store2.sources == {"a.txt"}
    results = store2.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0].text == "hello world"


def test_load_rejects_model_mismatch(tmp_path):
    """模型名不一致时拒绝加载并给出提示"""
    store = PersistentVectorStore()
    chunks = [_chunk("data")]
    chunks[0].embedding = np.array([1.0, 0.0], dtype=np.float32)
    store.add(chunks)
    store.save(tmp_path, meta={"model_name": "old-model", "dim": 2})

    store2 = PersistentVectorStore()
    ok, reason = store2.load(tmp_path, expected={"model_name": "new-model", "dim": 2})
    assert not ok
    assert "重新构建索引" in reason


def test_load_rejects_dim_mismatch(tmp_path):
    store = PersistentVectorStore()
    chunks = [_chunk("data")]
    chunks[0].embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store.add(chunks)
    store.save(tmp_path, meta={"model_name": "m", "dim": 3})

    store2 = PersistentVectorStore()
    ok, reason = store2.load(tmp_path, expected={"model_name": "m", "dim": 512})
    assert not ok
    assert "维度" in reason


def test_load_missing_index(tmp_path):
    store = PersistentVectorStore()
    ok, reason = store.load(tmp_path)
    assert not ok
    assert "不存在" in reason


def test_sources_property():
    store = PersistentVectorStore()
    chunks = [_chunk("a", source="a.txt"), _chunk("b", source="b.txt"), _chunk("c", source="a.txt")]
    for c in chunks:
        c.embedding = np.array([1.0, 0.0], dtype=np.float32)
    store.add(chunks)
    assert store.sources == {"a.txt", "b.txt"}