"""端到端测试 - 验证完整 RAG 管线"""
import sys
sys.path.insert(0, ".")

print("1. 初始化管线...")
from core.pipeline import MathRAGPipeline
pipeline = MathRAGPipeline()
print(f"   LLM: {pipeline._llm_type}")
print(f"   索引已加载: {pipeline._index_loaded}")
print()

print("2. 构建索引...")
msg = pipeline.build_index()
print(f"   {msg}")
print()

print("3. 检索测试...")
chunks = pipeline.retrieve("What is scaled dot-product attention?", top_k=3)
for i, c in enumerate(chunks, 1):
    preview = c.text[:100].replace("\n", " ")
    print(f"   [{i}] 相关度={c.score:.4f} | {preview}...")
print()

print("4. 端到端问答（模板 LLM，无 API Key）...")
answer, sources = pipeline.query("What is the attention formula?")
print(answer)
print()

print("5. 验证持久化（重新加载）...")
pipeline2 = MathRAGPipeline()
print(f"   索引已加载: {pipeline2._index_loaded}")
print(f"   分块数: {pipeline2.vector_store.size}")
chunks2 = pipeline2.retrieve("multi-head attention", top_k=2)
for i, c in enumerate(chunks2, 1):
    preview = c.text[:80].replace("\n", " ")
    print(f"   [{i}] 相关度={c.score:.4f} | {preview}...")
print()

print("=" * 50)
print("完整管线测试通过")
print("=" * 50)