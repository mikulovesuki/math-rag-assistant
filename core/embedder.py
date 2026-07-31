"""向量化模块 - SentenceTransformer 封装，输出 L2 归一化向量"""
from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """
    封装 sentence-transformers，输出 L2 归一化向量。

    归一化后 dot product 等价于 cosine similarity，
    VectorStore 可以直接用矩阵乘法做检索。

    模型懒加载：构造时不下载/加载模型，首次 embed 时才加载，
    避免 UI 启动或未建索引时阻塞。
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
    # bge-zh 系列官方推荐：检索 query 前加指令前缀可显著提升效果
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章:"

    def __init__(self, model_name: str = DEFAULT_MODEL, query_prefix: str | None = QUERY_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self._model = None
        self._dim: int | None = None

    def _ensure_loaded(self):
        """惰性加载模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """向量维度（触发模型加载）"""
        if self._dim is None:
            model = self._ensure_loaded()
            if hasattr(model, "get_embedding_dimension"):
                self._dim = model.get_embedding_dimension()
            else:
                self._dim = model.get_sentence_embedding_dimension()
        return self._dim

    def embed(self, text: str, query: bool = False) -> np.ndarray:
        """单条文本 -> 归一化向量。query=True 时附加检索指令前缀"""
        if query:
            text = self.query_prefix + text
        model = self._ensure_loaded()
        vec = model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量文本 -> 归一化向量列表（文档侧，不加指令前缀）"""
        model = self._ensure_loaded()
        vecs = model.encode(texts, normalize_embeddings=True)
        return [np.array(v, dtype=np.float32) for v in vecs]