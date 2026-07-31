"""混合检索测试:分词、融合排序、归一化"""
from __future__ import annotations

import numpy as np

from core.hybrid_retriever import HybridRetriever


def test_tokenize_mixed():
    """中英混合分词:中文分词 + 英文小写 + 纯符号过滤"""
    toks = HybridRetriever.tokenize("注意力机制 transformer 的公式 $E(x)$ 与 attention")
    assert "注意力" in toks
    assert "transformer" in toks
    assert "attention" in toks
    # 纯符号 token 应被过滤
    assert not any(not tok.isalnum() for tok in toks if not any(c.isalpha() for c in tok))


def test_fuse_blends_scores():
    """融合分数 = alpha * cosine + (1-alpha) * bm25_norm"""
    retriever = HybridRetriever(alpha=0.5)
    corpus = [
        "transformer 的 self-attention 多头注意力机制",
        "feed-forward 前馈网络逐位置变换",
        "位置编码 positional encoding",
    ]
    retriever.build(corpus)

    cosine = np.array([0.9, 0.5, 0.2], dtype=np.float32)
    results = retriever.fuse(cosine, "transformer attention", top_k=2)

    assert len(results) == 2
    # 第一个 chunk 向量分最高且含关键词,应排第一
    assert results[0][0] == 0
    # 分数应介于两通道分数之间
    idx, score = results[0]
    assert 0.2 <= score <= 0.95


def test_fuse_keyword_boosts_relevant():
    """关键词强命中的 chunk 应通过 BM25 提升排名"""
    retriever = HybridRetriever(alpha=0.3)  # BM25 权重更高
    corpus = [
        "关于深度学习模型的综述内容",
        "self-attention 多头注意力机制详解",
        "self-attention 注意力 self-attention 应用",
    ]
    retriever.build(corpus)

    # 向量分数无区分度
    cosine = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    results = retriever.fuse(cosine, "self-attention", top_k=3)

    # BM25 应把含 self-attention 的 chunk 排到前面
    assert results[0][0] != 0
    assert results[0][1] > results[2][1]


def test_normalize_flat_scores():
    """分数无区分度时归一化为全 0,不产生 NaN"""
    retriever = HybridRetriever(alpha=0.5)
    corpus = ["文本一", "文本二"]
    retriever.build(corpus)
    cosine = np.array([0.1, 0.1], dtype=np.float32)
    results = retriever.fuse(cosine, "无匹配关键词", top_k=2)
    assert len(results) == 2
    assert all(np.isfinite(s) for _, s in results)


def test_save_load_roundtrip(tmp_path):
    retriever = HybridRetriever()
    retriever.build(["你好 world", "数学公式 $E(x)$"])
    retriever.save(tmp_path)

    retriever2 = HybridRetriever()
    assert retriever2.load(tmp_path)
    assert retriever2.size == 2
    results = retriever2.fuse(np.array([0.5, 0.5], dtype=np.float32), "world", top_k=1)
    assert results[0][0] == 0, "含 world 的 chunk 应排第一"


def test_fuse_length_mismatch_raises():
    retriever = HybridRetriever()
    retriever.build(["a", "b"])
    with np.testing.assert_raises(ValueError):
        retriever.fuse(np.array([0.5], dtype=np.float32), "a", top_k=1)