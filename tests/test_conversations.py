"""历史会话存储测试 - 纯文件 IO，无需 Gradio"""
from core.conversations import ConversationStore


def _history():
    return [
        {"role": "user", "content": "什么是 scaled dot-product attention？"},
        {"role": "assistant", "content": "它是一种注意力打分方式…\n\n---\n**参考来源:**\n1. a.txt §3.1"},
    ]


def test_save_load_roundtrip(tmp_path):
    store = ConversationStore(tmp_path / "conv")
    sid = store.save(_history())
    assert sid

    data = store.load(sid)
    assert data is not None
    assert data["id"] == sid
    assert data["title"] == "什么是 scaled dot-produ"  # 前 20 字符
    assert len(data["messages"]) == 2


def test_save_with_given_sid_keeps_title(tmp_path):
    store = ConversationStore(tmp_path)
    sid = store.save([{"role": "user", "content": "第一个问题"}])
    store.save([{"role": "user", "content": "追问第二问"}], sid)

    data = store.load(sid)
    assert data["title"] == "第一个问题"
    assert data["messages"][-1]["content"] == "追问第二问"


def test_auto_title_truncates_and_falls_back(tmp_path):
    store = ConversationStore(tmp_path)
    long_title = store.save([{"role": "user", "content": "这个问题非常长" * 20}])
    assert len(store.load(long_title)["title"]) == 20

    empty = store.save([{"role": "assistant", "content": "只有回答"}])
    assert store.load(empty)["title"] == "未命名"


def test_list_orders_by_updated_at(tmp_path):
    store = ConversationStore(tmp_path)
    a = store.save([{"role": "user", "content": "AAA"}])
    store.save([{"role": "user", "content": "BBB"}])
    store.save([{"role": "user", "content": "CCC"}], a)  # 更新 a 使其排最前

    ids = [m.id for m in store.list()]
    assert ids == [a, *[i for i in ids if i != a]]


def test_delete_and_missing(tmp_path):
    store = ConversationStore(tmp_path)
    sid = store.save([{"role": "user", "content": "x"}])
    store.delete(sid)
    assert store.load(sid) is None
    assert store.list() == []
    store.delete("nonexistent")  # 幂等


def test_corrupted_file_skipped(tmp_path):
    store = ConversationStore(tmp_path)
    sid = store.save([{"role": "user", "content": "ok"}])
    bad = store.directory / "broken.json"
    bad.write_text("{not json", encoding="utf-8")

    assert store.load(sid) is not None
    assert all(m.id != "broken" for m in store.list())