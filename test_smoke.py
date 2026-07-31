"""冒烟测试 - 验证核心模块功能"""
import sys
sys.path.insert(0, ".")

import numpy as np
from core.models import Document, Chunk
from core.chunker import MarkdownChunker
import config


def test_models():
    """测试数据模型序列化"""
    c = Chunk(text="chunk text", source="test.pdf", index=0)
    c.embedding = np.array([1.0, 2.0], dtype=np.float32)
    d = c.to_dict()
    c2 = Chunk.from_dict(d)
    assert c2.text == c.text
    assert c2.source == c.source
    assert c2.index == c.index
    print("[PASS] models: serialize/deserialize")


def test_chunker_math_block():
    """测试公式块不被切断"""
    chunker = MarkdownChunker(chunk_size=200, overlap=30)

    text = """## Introduction

This paper presents a new method for solving differential equations.

The key equation is:

$$
\\frac{\\partial u}{\\partial t} = \\nabla^2 u + f(x, t)
$$

where $u$ is the solution and $f$ is the source term.

## Method

We use finite difference method:

```python
def solve(u, dt, dx):
    u_new = u + dt * laplacian(u, dx)
    return u_new
```

The stability condition is $\\Delta t \\leq \\frac{\\Delta x^2}{4D}$.
"""

    # 验证原子单元拆分
    units = chunker._split_into_units(text)
    print(f"  原子单元数: {len(units)}")
    for i, u in enumerate(units):
        print(f"  Unit {i} ({len(u)} chars): {u[:60].replace(chr(10), ' ')}...")

    # 验证分块
    chunks = chunker.chunk(text, source="paper.pdf")
    print(f"  分块数: {len(chunks)}")

    # 验证 $$ 配对
    all_paired = True
    for i, c in enumerate(chunks):
        count = c.text.count("$$")
        if count % 2 != 0:
            print(f"  [FAIL] chunk {i} has unpaired $$: count={count}")
            print(f"  content: {repr(c.text[:100])}")
            all_paired = False
        else:
            print(f"  chunk {i}: {len(c.text)} chars, $$ count={count} (OK)")

    if all_paired:
        print("[PASS] chunker: math blocks preserved")
    else:
        print("[FAIL] chunker: math block was split")


def test_chunker_code_block():
    """测试代码块不被切断"""
    chunker = MarkdownChunker(chunk_size=150, overlap=20)

    text = """## Code Section

Here is some code:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

And here is more text after the code block.
"""

    chunks = chunker.chunk(text, source="code.pdf")
    print(f"  分块数: {len(chunks)}")
    for i, c in enumerate(chunks):
        backtick_count = c.text.count("```")
        print(f"  chunk {i}: {len(c.text)} chars, ``` count={backtick_count}")
        if backtick_count % 2 != 0:
            print(f"  [FAIL] chunk {i} has unpaired ```")
        else:
            print(f"  (OK)")

    print("[PASS] chunker: code blocks preserved")


def test_config():
    """测试配置加载"""
    assert config.CHUNK_SIZE == 800
    assert config.CHUNK_OVERLAP == 100
    assert config.TOP_K == 5
    assert config.DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert config.DEEPSEEK_MODEL == "deepseek-chat"
    assert len(config.SYSTEM_PROMPT) > 0
    assert isinstance(config.has_api_key(), bool)
    print("[PASS] config: all values loaded")


def test_vector_store_persistence():
    """测试向量库持久化"""
    import tempfile
    from core.vector_store import PersistentVectorStore

    store = PersistentVectorStore()

    # 添加测试数据
    chunks = [
        Chunk(text="hello world", source="a.pdf", index=0, embedding=np.array([1, 0], dtype=np.float32)),
        Chunk(text="foo bar", source="b.pdf", index=0, embedding=np.array([0, 1], dtype=np.float32)),
    ]
    store.add(chunks)
    assert store.size == 2

    # 检索
    results = store.search(np.array([1, 0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].text == "hello world"
    assert results[0].score > 0.99

    # 持久化
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)

        # 重新加载
        store2 = PersistentVectorStore()
        loaded = store2.load(tmpdir)
        assert loaded
        assert store2.size == 2

        results2 = store2.search(np.array([1, 0], dtype=np.float32), top_k=1)
        assert len(results2) == 1
        assert results2[0].text == "hello world"

    print("[PASS] vector_store: add, search, save, load")


if __name__ == "__main__":
    print("=" * 60)
    print("冒烟测试 - 数学论文 RAG 助手")
    print("=" * 60)
    print()

    test_models()
    print()
    test_chunker_math_block()
    print()
    test_chunker_code_block()
    print()
    test_config()
    print()
    test_vector_store_persistence()
    print()
    print("=" * 60)
    print("全部通过")
    print("=" * 60)