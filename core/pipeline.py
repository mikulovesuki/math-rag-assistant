"""RAG 管线编排器 - 串联转换 -> 分块 -> 向量化 -> 检索 -> 生成"""
from __future__ import annotations

from pathlib import Path

import config
from .models import Document, Chunk
from .converter import DocumentConverter
from .chunker import MarkdownChunker
from .embedder import SentenceTransformerEmbedder
from .vector_store import PersistentVectorStore
from .llm import DeepSeekLLM, TemplateLLM


class MathRAGPipeline:
    """
    数学论文 RAG 管线。

    数据流:
        PDF ──converter──> Document ──chunker──> Chunk
             ──embedder──> VectorStore ──save──> 磁盘

        query ──embedder──> VectorStore ──search──> Top-K
             ──llm──> 回答 + 来源
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = MarkdownChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        self.embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL)
        self.vector_store = PersistentVectorStore()

        # 根据是否配置 API Key 选择 LLM
        if config.has_api_key():
            self.llm = DeepSeekLLM()
            self._llm_type = "DeepSeek"
        else:
            self.llm = TemplateLLM()
            self._llm_type = "Template（未配置 API Key）"

        # 启动时尝试加载已有索引
        self._index_loaded = self.vector_store.load(config.INDEX_DIR)

    def reinit_llm(self) -> None:
        """API Key 更新后重新初始化 LLM"""
        if config.has_api_key():
            self.llm = DeepSeekLLM()
            self._llm_type = "DeepSeek"
        else:
            self.llm = TemplateLLM()
            self._llm_type = "Template（未配置 API Key）"

    def build_index(self) -> str:
        """扫描 data/papers/，转换 + 分块 + 向量化 + 持久化。返回状态消息。"""
        papers_dir = Path(config.PAPERS_DIR)
        if not papers_dir.exists():
            return f"论文目录不存在: {papers_dir}"

        # 1. 转换文档
        documents = self.converter.convert_directory(papers_dir)
        if not documents:
            return f"未在 {papers_dir} 中找到可转换的文档"

        # 2. 分块
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunker.chunk(doc.text, doc.source))

        if not all_chunks:
            return "文档分块结果为空，请检查文档内容"

        # 3. 向量化
        embeddings = self.embedder.embed_batch([c.text for c in all_chunks])
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # 4. 存入向量库并持久化
        self.vector_store.clear()
        self.vector_store.add(all_chunks)
        self.vector_store.save(config.INDEX_DIR)

        return (
            f"索引构建完成: {len(documents)} 篇文档, "
            f"{len(all_chunks)} 个分块, 已保存到 {config.INDEX_DIR}"
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[Chunk]:
        """仅检索，不生成。返回 Top-K 文档块。"""
        if self.vector_store.size == 0:
            return []
        query_vec = self.embedder.embed(question)
        return self.vector_store.search(query_vec, top_k or config.TOP_K)

    def query(self, question: str, top_k: int | None = None) -> tuple[str, list[Chunk]]:
        """
        端到端问答：检索 + 生成。

        返回 (回答文本, 来源 chunk 列表)
        """
        if self.vector_store.size == 0:
            return "索引为空，请先上传论文并构建索引。", []

        chunks = self.retrieve(question, top_k)

        if not chunks:
            return "未检索到相关内容。", []

        answer = self.llm.generate(question, chunks)
        return answer, chunks

    @property
    def stats(self) -> dict:
        """索引统计信息"""
        sources = set()
        if self.vector_store.size > 0:
            # 通过 chunks 的 source 字段统计来源数
            for chunk in self.vector_store._chunks:
                sources.add(chunk.source)

        return {
            "chunk_count": self.vector_store.size,
            "paper_count": len(sources),
            "sources": sorted(sources),
            "index_loaded": self._index_loaded,
            "llm_type": self._llm_type,
        }