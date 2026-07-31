"""向量化模块 - 用 SentenceTransformer 将文本转为向量"""
from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """
    封装 sentence-transformers，输出 L2 归一化向量。

    归一化后 dot product 等价于 cosine similarity，
    VectorStore 可以直接用矩阵乘法做检索。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # 兼容新旧版本 API
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """单条文本 -> 归一化向量"""
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量文本 -> 归一化向量列表"""
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [np.array(v, dtype=np.float32) for v in vecs]