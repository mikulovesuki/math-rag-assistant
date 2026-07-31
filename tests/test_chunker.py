"""分块器测试:公式/代码块保护、heading 标题路径、overlap"""
from __future__ import annotations

from core.chunker import MarkdownChunker

MATH_DOC = r"""## 3.1 Attention

The key equation is:

$$
\frac{\partial u}{\partial t} = \nabla^2 u + f(x, t)
$$

where $u$ is the solution.

## 3.2 Method

We use code:

```python
def solve(u, dt):
    return u + dt * laplacian(u)
```

The stability condition is $\Delta t \leq \frac{\Delta x^2}{4D}$.
"""


def test_math_block_not_split():
    """公式块保持完整,不跨块切断"""
    chunker = MarkdownChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk(MATH_DOC, source="paper.pdf")

    assert chunks
    for c in chunks:
        assert c.text.count("$$") % 2 == 0, f"chunk {c.index} 公式未配对"


def test_code_block_not_split():
    """代码块保持完整,不跨块切断"""
    chunker = MarkdownChunker(chunk_size=150, overlap=20)
    chunks = chunker.chunk(MATH_DOC, source="paper.pdf")

    for c in chunks:
        assert c.text.count("```") % 2 == 0, f"chunk {c.index} 代码块未配对"


def test_heading_path_tracking():
    """标题路径:嵌套标题归并为完整路径,chunk 记录所属章节"""
    chunker = MarkdownChunker(chunk_size=1000, overlap=0)
    text = """# Transformer

## 3.1 Attention

multi-head attention 的内容。

### 3.1.1 Scaled Dot-Product

公式推导内容。

## 3.2 FFN

前馈网络内容。
"""
    chunks = chunker.chunk(text, source="t.pdf")

    # 标题栈归并:3.1 下嵌套 3.1.1,heading 由标题文本组成
    headings = {c.heading for c in chunks}
    assert "Transformer / 3.1 Attention" in headings
    assert "Transformer / 3.1 Attention / 3.1.1 Scaled Dot-Product" in headings
    assert "Transformer / 3.2 FFN" in headings


def test_heading_reset_on_sibling():
    """同级标题替换栈顶,不会错误叠加"""
    chunker = MarkdownChunker(chunk_size=1000, overlap=0)
    text = """# A

## B1

内容一。

## B2

内容二。
"""
    chunks = chunker.chunk(text, source="t.pdf")
    headings = {c.heading for c in chunks}
    assert "A / B1" in headings
    assert "A / B2" in headings
    assert "A / B1 / B2" not in headings


def test_overlap_keeps_at_least_one_unit():
    """overlap 时至少保留一个尾部单元,不会完全丢失重叠"""
    chunker = MarkdownChunker(chunk_size=40, overlap=10)
    units = ["短段落一", "短段落二", "这是一个很长的段落超过了 overlap 设定值"]
    kept, total = chunker._take_overlap(units)

    assert len(kept) >= 1, "overlap 不应为空"
    assert total >= len(kept[0]), "overlap 长度应覆盖保留的单元"


def test_oversized_unit_isolated():
    """单个超长单元独立成块,不截断也不影响后续分块"""
    chunker = MarkdownChunker(chunk_size=50, overlap=0)
    big = "x" * 200
    text = f"## H\n\n开头段落。\n\n{big}\n\n结尾段落。"
    chunks = chunker.chunk(text, source="t.pdf")

    texts = [c.text for c in chunks]
    assert any(big in t for t in texts), "超长单元应完整存在"
    assert all(("$$" not in c.text or c.text.count("$$") % 2 == 0) for c in chunks)