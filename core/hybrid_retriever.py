"""混合检索器 - BM25 关键词检索 + 向量余弦 加权融合"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


class HybridRetriever:
    """
    融合 BM25 关键词检索与向量余弦检索。

    数学论文中精确术语（transformer、self-attention）依赖关键词匹配，
    语义相近的表述依赖向量检索，加权融合互补：

        final_score = alpha * cosine + (1 - alpha) * bm25_norm

    其中 bm25_norm 将无界的 BM25 分数 min-max 归一化到 [0, 1]。

    - BM25 语料随索引持久化（纯文本 json），加载时重建
    - 中文用 jieba 分词，英文小写按 token 切分，纯符号 token 过滤
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._texts: list[str] = []
        self._bm25 = None  # BM25Okapi, 延迟导入

    # ── 索引构建与持久化 ──────────────────────────────────

    def build(self, texts: list[str]) -> None:
        """全量构建 BM25 索引（与向量库 chunk 顺序一一对应）"""
        from rank_bm25 import BM25Okapi
        self._texts = list(texts)
        self._bm25 = BM25Okapi([self.tokenize(t) for t in self._texts])
        self._repair_zero_idf()

    def _repair_zero_idf(self) -> None:
        """
        rank_bm25 的 ATIRE 变体 idf 在词出现在恰好一半文档时退化为 0，
        小语料（几篇论文、几十个 chunk）下关键词检索会完全失效。
        将 idf<=0 的词补一个正下限，保证 BM25 通道始终有区分度。
        """
        idf = self._bm25.idf
        avg = self._bm25.average_idf
        floor = max(avg, 0.0) * 0.1 + 0.1
        for word, value in idf.items():
            if value <= 0:
                idf[word] = floor

    def save(self, dir_path: str | Path) -> None:
        """持久化 BM25 语料"""
        directory = Path(dir_path)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "bm25_corpus.json", "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False)

    def load(self, dir_path: str | Path) -> bool:
        """从磁盘加载语料并重建索引，成功返回 True"""
        corpus_path = Path(dir_path) / "bm25_corpus.json"
        if not corpus_path.exists():
            return False
        with open(corpus_path, "r", encoding="utf-8") as f:
            self._texts = json.load(f)
        if not self._texts:
            return False
        self.build(self._texts)
        return True

    # ── 融合检索 ─────────────────────────────────────────

    def fuse(self, cosine_scores: np.ndarray, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        融合向量分数与 BM25 分数，返回 (chunk_index, final_score) 降序列表。

        cosine_scores: 所有 chunk 的向量相似度（与语料顺序一致）。
        """
        if self._bm25 is None or not self._texts:
            return []
        if len(cosine_scores) != len(self._texts):
            raise ValueError("cosine_scores 长度与 BM25 语料不一致")

        bm25_raw = np.array(self._bm25.get_scores(self.tokenize(query)), dtype=np.float64)
        bm25_norm = self._normalize(bm25_raw)
        final = self.alpha * np.asarray(cosine_scores, dtype=np.float64) + (1 - self.alpha) * bm25_norm

        order = np.argsort(final)[::-1][:top_k]
        return [(int(i), float(final[i])) for i in order]

    @property
    def size(self) -> int:
        return len(self._texts)

    # ── 工具 ─────────────────────────────────────────────

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """min-max 归一化；分数无区分度时返回全 0"""
        max_s, min_s = float(scores.max()), float(scores.min())
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores, dtype=np.float64)
        return (scores - min_s) / (max_s - min_s)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        中英混合分词：
        - 中文：jieba 精确模式
        - 英文/数字：小写按空白切分
        - 纯符号 token（$、(、= 等）过滤，保留如 frac/partial/attention 等有语义的 token
        """
        import jieba

        tokens: list[str] = []
        for seg in jieba.cut(text, cut_all=False):
            for sub in re.split(r"\s+", seg):
                sub = sub.strip().lower()
                if not sub or not re.search(r"[a-z0-9\u4e00-\u9fff]", sub):
                    continue
                tokens.append(sub)
        return tokens