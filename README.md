# Math RAG Assistant

基于 RAG（检索增强生成）架构的数学论文智能问答系统。上传 PDF 论文，系统自动完成文档转换、语义分块、向量化索引，用户通过 Web 界面提问，AI 基于论文内容生成带来源引用的回答。

## Tech Stack

| 层级 | 技术 | 用途 |
|------|------|------|
| **文档处理** | MarkItDown (Microsoft) | PDF / Word / PPT -> Markdown 文本转换 |
| **分块策略** | 自研 MarkdownChunker | 原子单元拆分，保护 `$$公式$$` 和 `` `代码块` `` 不被切断，记录章节标题路径 |
| **向量化** | Sentence-Transformers (`BAAI/bge-small-zh-v1.5`) | 本地双语 Embedding，L2 归一化，懒加载，无需 API |
| **混合检索** | NumPy Cosine + BM25 (jieba 分词) | 向量语义 + 关键词加权融合，公式与精确术语命中率高 |
| **LLM** | DeepSeek (OpenAI-compatible API) | 多轮上下文增强生成，流式输出，支持 `deepseek-chat` / `deepseek-reasoner` |
| **前端** | Gradio Blocks | 可折叠侧边栏 + 深色聊天界面 + 面板切换 + KaTeX 公式渲染 |
| **工程化** | Protocol 接口协议 / dataclass / .env 配置 / pytest | 模块解耦，组件可替换，测试离线可跑 |

## Architecture

```mermaid
graph LR
    subgraph "阶段1: 离线建库 Indexing"
        A[PDF 论文] -->|MarkItDown| B[Markdown 文本]
        B -->|MarkdownChunker + 标题路径| C[Chunk 列表]
        C -->|bge-small-zh 懒加载| D[向量矩阵]
        C -->|jieba 分词| E[BM25 索引]
        D -->|save .npy + chunks.json + index_meta.json| F[(磁盘持久化)]
        E -->|bm25_corpus.json| F
    end

    subgraph "阶段2: 在线检索 Retrieval"
        G[用户提问 + 多轮历史] -->|embed + 检索指令前缀| D
        G -->|jieba 分词| E
        D -->|cosine 相似度| H[分数融合 alpha * cosine + (1-alpha) * bm25]
        E -->|BM25 分数| H
        H -->|Top-K| I[相关 Chunk]
    end

    subgraph "阶段3: 增强生成 Generation"
        I -->|system + 历史 + 上下文| J[DeepSeek API 流式]
        J --> K[回答 + 章节级引用]
        K -->|Gradio + KaTeX| L[用户界面]
    end
```

## Key Features

- **Markdown 感知分块** -- 自动识别 `$$...$$` 公式块、`` `...` `` 代码块、标题行和段落作为原子单元，保证语义完整性不被截断
- **章节级引用** -- 分块器维护标题栈，每个 chunk 记录所属章节路径（如 `3.1 Attention`），回答引用精确到章节
- **混合检索** -- 向量余弦 + BM25 关键词加权融合（默认权重 0.7/0.3），中文 jieba 分词，英文小写 token 化，公式符号自动过滤
- **索引版本校验** -- 索引携带模型名/维度元数据，更换模型后自动检测并要求重建，杜绝静默维度错误
- **索引同步检测** -- 启动时对比论文目录与索引文件清单，文件增删后提示重建
- **多轮对话** -- 历史消息（最近 6 轮）参与生成，支持追问
- **流式输出** -- DeepSeek 流式生成逐段渲染，长推导不再白屏等待
- **组件全可替换** -- Embedder / VectorStore / LLM / Chunker / Converter 均通过 `typing.Protocol` 定义契约，零继承耦合
- **模型懒加载** -- 未构建索引时不触发 embedding 模型下载/加载，UI 秒开
- **API Key 界面内配置** -- Web UI 直接填写密钥，自动写入 `.env`，无需命令行操作
- **可折叠侧边栏** -- 深色主题侧边栏支持一键收起，聊天区域自适应占满全屏

## Quick Start

```bash
# 1. 安装依赖
git clone https://github.com/mikulovesuki/math-rag-assistant.git
cd math-rag-assistant
pip install -r requirements.txt

# 2. 启动（首次启动会在界面内引导配置 API Key）
python main.py
```

打开浏览器访问 `http://localhost:7860`，按界面引导三步完成：

1. **配置密钥** -- 粘贴 DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）
2. **上传论文** -- 拖入 PDF/Word 文件，点击「构建索引」（首次会下载 embedding 模型，约 95MB）
3. **智能问答** -- 输入问题，获取基于论文内容的回答

> 未配置 API Key 时系统自动降级为模板模式，可正常体验检索流程。

## 运行测试

```bash
python -m pytest tests/ -v
```

测试使用 stub 组件（确定性向量/固定回答），不依赖模型下载与外部 API，离线可跑。

## Project Structure

```
math_rag_assistant/
├── main.py                  Gradio Web UI（流式聊天 + 侧边栏 + KaTeX 渲染）
├── config.py                全局配置（API Key / 模型 / 分块参数 / System Prompt）
├── requirements.txt
├── .env.example             环境变量模板
├── tests/
│   ├── conftest.py          stub 组件（离线可跑）
│   ├── test_chunker.py      分块器测试（公式/代码块保护、标题路径）
│   ├── test_hybrid.py       混合检索测试（分词、融合排序）
│   ├── test_pipeline.py     管线端到端测试（建库/检索/问答/持久化）
│   └── test_vector_store.py 向量库测试（JSON 持久化、版本校验）
├── core/
│   ├── models.py            数据模型 (Document, Chunk)
│   ├── protocols.py         组件契约 (typing.Protocol)
│   ├── converter.py         MarkItDown 封装 -- 多格式文档转文本
│   ├── chunker.py           Markdown 感知分块 -- 原子单元保护 + 章节标题路径
│   ├── embedder.py          bge-small-zh 封装 -- L2 归一化 + 懒加载 + 检索指令前缀
│   ├── hybrid_retriever.py  混合检索 -- BM25(jieba) + Cosine 加权融合
│   ├── vector_store.py      NumPy 向量库 -- 检索 + JSON 持久化 + 版本校验
│   ├── llm.py               DeepSeek LLM 封装 -- 多轮历史 + 流式 + 超时
│   └── pipeline.py          RAG 管线编排 -- 串联三阶段
└── data/
    ├── papers/              论文存放目录
    └── index/               向量索引持久化（自动生成）
```

## Configuration

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key，界面内配置或写入 `.env` |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 可切换 `deepseek-reasoner` 进行深度推理 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地双语 Embedding 模型，可替换为 `BAAI/bge-m3` 等 |
| `BM25_ALPHA` | `0.7` | 混合检索融合权重：`alpha * 向量分数 + (1-alpha) * BM25` |
| `CHUNK_SIZE` | `800` | 每个分块最大字符数 |
| `CHUNK_OVERLAP` | `100` | 相邻块重叠字符数 |
| `TOP_K` | `5` | 检索返回的文档块数量 |

## Module Design

每个核心模块通过 [core/protocols.py](core/protocols.py) 中的 `typing.Protocol` 定义契约，结构子类型（鸭子类型），可独立替换：

```python
from typing import Protocol
import numpy as np
from core.models import Chunk

class Embedder(Protocol):
    model_name: str
    def embed(self, text: str, query: bool = False) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

class LLM(Protocol):
    def generate(self, query: str, context: list[Chunk],
                 history: list[dict] | None = None) -> str: ...
    def generate_stream(self, query: str, context: list[Chunk],
                        history: list[dict] | None = None): ...
```

替换实现只需满足方法签名，无需继承：

```python
class MyEmbedder:
    """满足 Embedder Protocol 的自定义向量化实现"""
    model_name = "my-model"
    def embed(self, text, query=False): ...
    def embed_batch(self, texts): ...
```

管线构造器支持组件注入，测试与替换均无需改动编排逻辑：

```python
pipeline = MathRAGPipeline(embedder=MyEmbedder(), llm=MyLLM())
```

## License

MIT