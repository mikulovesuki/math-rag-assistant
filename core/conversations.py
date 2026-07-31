"""历史会话存储 - 每个会话一个 JSON 文件，消息格式与聊天组件互通"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionMeta:
    """会话列表元信息（列表页只读标题与时间，不加载全量消息）"""

    id: str
    title: str
    updated_at_ns: int


class ConversationStore:
    """
    会话持久化：data/conversations/{sid}.json

    文件结构: {"id", "title", "updated_at_ns", "messages": [{"role", "content"}, ...]}
    messages 直接复用聊天组件的 dict 格式，加载后无需转换。
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    # ── 文件路径 ─────────────────────────────────────────

    def _ensure_dir(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def _path(self, sid: str) -> Path:
        return self._ensure_dir() / f"{sid}.json"

    # ── 列表 / 读取 ──────────────────────────────────────

    def list(self) -> list[SessionMeta]:
        """全部会话按更新时间倒序；损坏文件跳过"""
        metas: list[SessionMeta] = []
        for f in self._ensure_dir().glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                metas.append(
                    SessionMeta(
                        id=data["id"],
                        title=data.get("title", "未命名"),
                        updated_at_ns=int(data.get("updated_at_ns", 0)),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                continue
        return sorted(metas, key=lambda m: m.updated_at_ns, reverse=True)

    def load(self, sid: str) -> dict | None:
        """读取会话全量数据（含消息），不存在或损坏返回 None"""
        path = self._path(sid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ── 写入 / 删除 ─────────────────────────────────────

    def save(self, messages: list[dict], sid: str = "") -> str:
        """保存会话；sid 为空时新建并返回生成的 id。已有会话保留原标题。"""
        sid = sid or uuid.uuid4().hex[:12]
        path = self._path(sid)
        title = self._load_title(path) or self._derive_title(messages)
        payload = {
            "id": sid,
            "title": title,
            "updated_at_ns": time.time_ns(),
            "messages": messages,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return sid

    def delete(self, sid: str) -> None:
        self._path(sid).unlink(missing_ok=True)

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _load_title(path: Path) -> str | None:
        """读取已有会话标题；文件缺失或损坏返回 None"""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("title")
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _derive_title(messages: list[dict]) -> str:
        """取第一条用户消息的前 20 字作为标题"""
        for m in messages:
            if m.get("role") == "user":
                text = m.get("content", "").strip().replace("\n", " ")
                return text[:20] if text else "未命名"
        return "未命名"