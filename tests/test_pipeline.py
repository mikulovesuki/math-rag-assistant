"""管线端到端测试:建库 - 检索 - 问答 - 持久化重载,全部走临时目录与 stub 组件"""
from __future__ import annotations

import json

import pytest

import config
from core.pipeline import MathRAGPipeline


PAPER_TXT = """# Transformer 综述

## 3.1 Attention

注意力机制是 transformer 的核心,self-attention 计算查询与键的相似度。

## 3.2 Feed-Forward

前馈网络对每个位置独立做非线性变换。

## 3.3 Positional Encoding

位置编码为序列注入位置信息。
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    """隔离的 papers/index 目录 + 注入 stub 组件的工厂"""
    papers = tmp_path / "papers"
    index = tmp_path / "index"
    papers.mkdir()
    (papers / "survey.txt").write_text(PAPER_TXT, encoding="utf-8")

    monkeypatch.setattr(config, "PAPERS_DIR", papers)
    monkeypatch.setattr(config, "INDEX_DIR", index)

    def make_pipeline() -> MathRAGPipeline:
        from tests.conftest import StubConverter, StubEmbedder, StubLLM
        return MathRAGPipeline(
            converter=StubConverter(),
            embedder=StubEmbedder(),
            llm=StubLLM(),
        )

    return make_pipeline


def test_build_index_creates_all_files(env):
    p = env()
    msg = p.build_index()

    assert "索引构建完成" in msg
    assert p.vector_store.size > 0
    assert p.stats["paper_count"] == 1

    from pathlib import Path
    index_dir = Path(config.INDEX_DIR)
    for name in ("vectors.npy", "chunks.json", "index_meta.json", "bm25_corpus.json"):
        assert (index_dir / name).exists(), f"缺少 {name}"
    assert p.stats["index_loaded"]


def test_retrieve_returns_scored_chunks(env):
    p = env()
    p.build_index()

    chunks = p.retrieve("self-attention 注意力", top_k=2)
    assert len(chunks) == 2
    assert all(c.score > 0 for c in chunks)
    # 首个 chunk 应含注意力相关内容
    assert any("attention" in c.text.lower() or "注意力" in c.text for c in chunks)
    # 混合检索补充的章节引用
    assert any(c.heading for c in chunks)


def test_query_with_history(env):
    p = env()
    p.build_index()

    history = [
        {"role": "user", "content": "什么是 transformer?"},
        {"role": "assistant", "content": "基于注意力的模型。"},
    ]
    answer, sources = p.query("那注意力怎么算?", history=history)
    assert answer
    assert len(sources) > 0
    assert "答案" in answer  # StubLLM 输出


def test_stream_answer(env):
    p = env()
    p.build_index()

    chunks, error = p.prepare_query("位置编码")
    assert error == ""
    parts = list(p.stream_answer("位置编码", chunks))
    assert parts, "流式输出不应为空"
    assert "".join(parts) == p.llm.generate("位置编码", chunks)


def test_persistence_reload(env):
    p = env()
    p.build_index()

    p2 = env()
    assert p2.stats["index_loaded"]
    assert p2.vector_store.size == p.vector_store.size
    results = p2.retrieve("feed-forward 前馈网络", top_k=1)
    assert results


def test_reload_rejects_old_model(env):
    p = env()
    p.build_index()

    # 篡改索引元数据模拟旧模型构建的索引
    meta_path = config.INDEX_DIR / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["model_name"] = "old-model"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    p2 = env()
    assert not p2.stats["index_loaded"]
    assert "重新构建索引" in p2.stats["load_message"]


def test_index_fresh_detects_changes(env, tmp_path):
    p = env()
    p.build_index()

    ok, msg = p.index_fresh()
    assert ok

    # 新增论文文件后应提示需重建
    (config.PAPERS_DIR / "new_paper.txt").write_text("新论文", encoding="utf-8")
    ok, msg = p.index_fresh()
    assert not ok
    assert "新增" in msg


def test_retrieve_empty_index(env):
    p = env()
    assert p.retrieve("任何问题") == []
    chunks, error = p.prepare_query("任何问题")
    assert chunks is None
    assert "索引为空" in error