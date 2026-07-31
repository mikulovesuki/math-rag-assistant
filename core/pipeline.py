"""RAG 管线编排器 - 串联转换 -> 分块 -> 向量化 -> 混合检索 -> 生成"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import config
from .models import Document, Chunk
from .converter import DocumentConverter
from .chunker import MarkdownChunker
from .embedder import SentenceTransformerEmbedder
from .vector_store import PersistentVectorStore
from .hybrid_retriever import HybridRetriever
from .llm import DeepSeekLLM, TemplateLLM


class MathRAGPipeline:
    """
    数学论文 RAG 管线。

    数据流:
        PDF ──converter──> Document ──chunker──> Chunk
             ──embedder──> VectorStore + HybridRetriever ──save──> 磁盘

        query ──embedder──> 向量分数 + BM25 ──融合──> Top-K
             ──llm(流式, 多轮)──> 回答 + 来源
    """

    def __init__(
        self,
        converter: DocumentConverter | None = None,
        chunker: MarkdownChunker | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        vector_store: PersistentVectorStore | None = None,
        hybrid: HybridRetriever | None = None,
        llm: DeepSeekLLM | TemplateLLM | None = None,
    ):
        """组件可注入，便于测试替换（如 stub embedder）"""
        self.converter = converter or DocumentConverter()
        self.chunker = chunker or MarkdownChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        self.embedder = embedder or SentenceTransformerEmbedder(config.EMBEDDING_MODEL)
        self.vector_store = vector_store or PersistentVectorStore()
        self.hybrid = hybrid or HybridRetriever(alpha=config.BM25_ALPHA)
        self.llm = llm or self._make_llm()

        # 启动时尝试加载已有索引（含模型/维度版本校验）
        self._index_loaded = False
        self._load_message = ""
        self._try_load_index()

    # ── 组件与索引生命周期 ──────────────────────────────────

    def _make_llm(self) -> DeepSeekLLM | TemplateLLM:
        """根据是否配置 API Key 选择 LLM"""
        return DeepSeekLLM() if config.has_api_key() else TemplateLLM()

    def _try_load_index(self) -> None:
        index_dir = Path(config.INDEX_DIR)
        try:
            expected = {"model_name": self.embedder.model_name}
            # 仅当索引存在时才需要模型维度（避免无模型缓存时误触发下载）
            if (index_dir / "vectors.npy").exists():
                expected["dim"] = self.embedder.dim

            loaded, reason = self.vector_store.load(index_dir, expected=expected)
            if loaded:
                hybrid_ok = self.hybrid.load(index_dir)
                if not hybrid_ok or self.hybrid.size != self.vector_store.size:
                    loaded = False
                    reason = "索引数据不完整（BM25 语料与向量数量不一致），请重新构建索引"
            self._index_loaded = loaded
            self._load_message = reason
        except Exception as e:
            self._index_loaded = False
            self._load_message = f"索引加载失败: {e}"

    def reinit_llm(self) -> None:
        """API Key 更新后重新初始化 LLM"""
        self.llm = self._make_llm()

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

        # 4. 存入向量库 + BM25 索引并持久化
        self.vector_store.clear()
        self.vector_store.add(all_chunks)
        self.vector_store.save(config.INDEX_DIR, meta={
            "model_name": self.embedder.model_name,
            "files": [d.source for d in documents],
        })
        self.hybrid.build([c.text for c in all_chunks])
        self.hybrid.save(config.INDEX_DIR)
        self._index_loaded = True
        self._load_message = "索引加载成功"

        return (
            f"索引构建完成: {len(documents)} 篇文档, "
            f"{len(all_chunks)} 个分块, 已保存到 {config.INDEX_DIR}"
        )

    # ── 检索与问答 ─────────────────────────────────────────

    def retrieve(self, question: str, top_k: int | None = None) -> list[Chunk]:
        """混合检索（向量 + BM25 融合），返回 Top-K 文档块（带 score）"""
        if self.vector_store.size == 0:
            return []

        query_vec = self.embedder.embed(question, query=True)
        cosine = self.vector_store.cosine_scores(query_vec)
        fused = self.hybrid.fuse(cosine, question, top_k or config.TOP_K)

        results: list[Chunk] = []
        for idx, score in fused:
            chunk = self.vector_store.get_chunk(idx)
            chunk.score = score
            results.append(chunk)
        return results

    def prepare_query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[Chunk] | None, str]:
        """
        流式问答前的检索准备。
        返回 (chunks, 错误信息)；chunks 为 None 表示检索阶段已失败。
        """
        if self.vector_store.size == 0:
            return None, "索引为空，请先上传论文并构建索引。"

        chunks = self.retrieve(question, top_k)
        if not chunks:
            return None, "未检索到相关内容。"
        return chunks, ""

    def query(
        self,
        question: str,
        top_k: int | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, list[Chunk]]:
        """端到端问答（非流式）：检索 + 生成。返回 (回答, 来源)"""
        chunks, error = self.prepare_query(question, top_k)
        if chunks is None:
            return error, []

        answer = self.llm.generate(question, chunks, history)
        return answer, chunks

    def stream_answer(
        self,
        question: str,
        chunks: list[Chunk],
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """基于已检索结果流式生成回答"""
        yield from self.llm.generate_stream(question, chunks, history)

    # ── 状态 ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """索引统计信息"""
        return {
            "chunk_count": self.vector_store.size,
            "paper_count": len(self.vector_store.sources),
            "sources": sorted(self.vector_store.sources),
            "index_loaded": self._index_loaded,
            "load_message": self._load_message,
            "llm_type": type(self.llm).__name__,
        }

    def index_fresh(self) -> tuple[bool, str]:
        """检查索引与论文目录是否同步，返回 (是否同步, 说明)"""
        files_in_meta = set(self.vector_store.meta.get("files", []))
        if not files_in_meta:
            return True, ""

        papers_dir = Path(config.PAPERS_DIR)
        current_files = {
            p.name for p in papers_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DocumentConverter.SUPPORTED_EXTENSIONS
        } if papers_dir.exists() else set()

        if files_in_meta != current_files:
            added = sorted(current_files - files_in_meta)
            removed = sorted(files_in_meta - current_files)
            detail = []
            if added:
                detail.append(f"新增 {len(added)} 个文件")
            if removed:
                detail.append(f"移除 {len(removed)} 个文件")
            return False, "论文目录已变化（" + "，".join(detail) + "），请重新构建索引"

        return True, ""