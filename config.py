"""全局配置 - 所有可配置项集中管理"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ── 路径配置 ──────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
INDEX_DIR = BASE_DIR / "data" / "index"

# ── DeepSeek LLM 配置 ─────────────────────────────────────

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── Embedding 配置 ────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── 分块参数 ──────────────────────────────────────────────

CHUNK_SIZE = 800       # 每个块的最大字符数
CHUNK_OVERLAP = 100    # 相邻块之间的重叠字符数

# ── 检索参数 ──────────────────────────────────────────────

TOP_K = 5              # 检索返回的文档块数量

# ── System Prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """你是一个数学论文研究助手。请基于检索到的论文片段回答用户的数学问题。

要求：
1. 优先使用检索资料中的内容回答，在关键论点后标注来源 [来源: 文件名]
2. 数学公式用 LaTeX 格式输出，行内公式用 $...$，独立公式用 $$...$$
3. 解释公式时给出逐步推导，不要跳步
4. 如果检索资料中没有足够信息，明确说明"未在已索引的论文中找到相关内容"
5. 回答使用中文，专业术语保留英文原文
"""


def has_api_key() -> bool:
    """检查是否配置了有效的 API Key"""
    return bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-api-key-here")


def save_api_key(api_key: str) -> None:
    """将 API Key 写入 .env 文件并更新全局变量"""
    global DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY = api_key.strip()

    env_path = BASE_DIR / ".env"
    lines: list[str] = []
    found = False

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DEEPSEEK_API_KEY="):
                    lines.append(f"DEEPSEEK_API_KEY={api_key.strip()}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"DEEPSEEK_API_KEY={api_key.strip()}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)