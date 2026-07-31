"""数学论文智能问答助手 - Gradio Web UI"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import gradio as gr

import config
from core.conversations import ConversationStore
from core.pipeline import MathRAGPipeline

_pipeline: MathRAGPipeline | None = None
_store = ConversationStore(config.CONVERSATIONS_DIR)


def get_pipeline() -> MathRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MathRAGPipeline()
    return _pipeline


# ── 事件处理 ─────────────────────────────────────────────

def handle_save_api_key(api_key: str) -> str:
    if not api_key or not api_key.strip():
        return "请输入 API Key"
    try:
        config.save_api_key(api_key.strip())
        if _pipeline is not None:
            _pipeline.reinit_llm()
        return "密钥已保存。下次提问时将使用 DeepSeek 生成回答。"
    except Exception as e:
        return f"保存失败: {e}"


def handle_upload(files: list) -> str:
    if not files:
        return "请先选择文件"
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for file in files:
        src = Path(file.name) if hasattr(file, "name") else Path(file)
        dest = config.PAPERS_DIR / src.name
        shutil.copy2(src, dest)
        saved.append(src.name)
    return f"已保存 {len(saved)} 个文件: {', '.join(saved)}"


def handle_build_index() -> tuple[str, str]:
    try:
        p = get_pipeline()
        msg = p.build_index()
        stats = p.stats
        return (
            f"{msg}\n\n当前: {stats['paper_count']} 篇论文, {stats['chunk_count']} 个分块",
            index_status_html(),
        )
    except Exception as e:
        return f"构建失败: {e}", index_status_html()


def _append(history: list[dict], role: str, content: str) -> None:
    """向聊天记录追加一条消息"""
    history.append({"role": role, "content": content})


# ── 历史会话 ─────────────────────────────────────────────

def _session_choices() -> list[tuple[str, str]]:
    """侧边栏下拉选项: "标题 · MM-DD HH:MM" -> 会话 id"""
    return [
        (
            f"{m.title} · {time.strftime('%m-%d %H:%M', time.localtime(m.updated_at_ns / 1e9))}",
            m.id,
        )
        for m in _store.list()
    ]


def handle_load_session(sid: str):
    """加载历史会话：恢复消息并切换到问答面板"""
    data = _store.load(sid) if sid else None
    history = list(data["messages"]) if data else []
    return list(history), *show_chat(), sid if data else ""


def handle_delete_session(sid: str) -> tuple[list, str, gr.update]:
    """删除会话并清空当前聊天区"""
    if sid:
        _store.delete(sid)
    return [], "", gr.update(choices=_session_choices(), value=None)


def handle_new_chat() -> tuple[list, str]:
    """开始新对话：清空聊天区与当前会话 id"""
    return [], ""


def handle_chat(message: str, history: list[dict], session_id: str):
    """
    流式问答：逐段产出回答，最后附加参考来源；
    结束时自动保存会话（失败消息也保存，保证历史完整）。
    """
    if not message.strip():
        yield gr.update(), history, session_id, gr.update()
        return

    if not config.has_api_key():
        _append(history, "user", message)
        _append(history, "assistant",
                "尚未配置 API Key，无法生成回答。\n\n请点击左上角菜单，进入「配置密钥」完成配置。")
        session_id = _store.save(history, session_id)
        yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())
        return

    try:
        p = get_pipeline()
    except Exception as e:
        _append(history, "user", message)
        _append(history, "assistant", f"系统初始化失败: {e}")
        session_id = _store.save(history, session_id)
        yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())
        return

    # 检索（失败时不加载/调用 LLM）
    try:
        chunks, error = p.prepare_query(message)
    except Exception as e:
        _append(history, "user", message)
        _append(history, "assistant", f"检索失败: {e}")
        session_id = _store.save(history, session_id)
        yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())
        return

    if chunks is None:
        _append(history, "user", message)
        _append(history, "assistant", error)
        session_id = _store.save(history, session_id)
        yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())
        return

    # LLM 侧历史：仅最近 6 轮，保持上下文长度可控
    llm_history = [
        {"role": h["role"], "content": h["content"]}
        for h in history[-12:]
        if h.get("role") in ("user", "assistant")
    ]

    _append(history, "user", message)
    _append(history, "assistant", "")
    assistant_msg = ""
    try:
        for delta in p.stream_answer(message, chunks, llm_history):
            assistant_msg += delta
            history[-1]["content"] = assistant_msg
            yield gr.update(value=""), list(history), gr.update(), gr.update()
    except Exception as e:
        history[-1]["content"] = f"生成失败: {e}"
        session_id = _store.save(history, session_id)
        yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())
        return

    # 生成完成后追加参考来源
    if chunks:
        source_lines = ["\n\n---\n**参考来源:**"]
        for i, s in enumerate(chunks, 1):
            ref = s.source
            if s.heading:
                ref += f" §{s.heading}"
            source_lines.append(f"{i}. {ref} (相关度: {s.score:.3f})")
        history[-1]["content"] = assistant_msg + "\n".join(source_lines)

    session_id = _store.save(history, session_id)
    yield gr.update(value=""), list(history), session_id, gr.update(choices=_session_choices())


def handle_clear() -> list[dict]:
    return []


def index_status_text() -> str:
    """索引状态文案（防御性获取，不抛异常）"""
    try:
        p = get_pipeline()
        st = p.stats
        if st["chunk_count"] == 0:
            return "未构建"
        if not st["index_loaded"]:
            return f"不可用: {st['load_message']}"
        fresh, msg = p.index_fresh()
        if not fresh:
            return f"需重建: {msg}"
        return f"已构建 ({st['chunk_count']} 块 / {st['paper_count']} 篇)"
    except Exception as e:
        return f"加载失败: {e}"


# ── 状态徽章 ─────────────────────────────────────────────

def _badge(text: str, ok: bool | None = None) -> str:
    """状态徽章 HTML；ok=None 时按文字是否含「已」推断"""
    if ok is None:
        ok = "已" in text
    cls = "ok" if ok else "warn"
    return f"<span class=\"badge {cls}\">● {text}</span>"


def index_status_html() -> str:
    """索引状态徽章（Markdown 兼容 HTML）"""
    return f"**索引** {_badge(index_status_text())}"


def api_status_html() -> str:
    """API 密钥状态徽章"""
    ok = config.has_api_key()
    return f"**密钥** {_badge('已配置' if ok else '未配置', ok)}"


# ── 侧边栏切换 ───────────────────────────────────────────

def toggle_sidebar(visible: bool) -> tuple[gr.update, bool, gr.update]:
    """切换侧边栏显示/隐藏，同时更新按钮文字"""
    new_visible = not visible
    btn_text = "<< 收起菜单" if new_visible else ">> 展开菜单"
    return gr.update(visible=new_visible), new_visible, gr.update(value=btn_text)


def show_config() -> tuple[gr.update, gr.update, gr.update]:
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def show_upload() -> tuple[gr.update, gr.update, gr.update]:
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def show_chat() -> tuple[gr.update, gr.update, gr.update]:
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


# ── CSS ──────────────────────────────────────────────────

CUSTOM_CSS = """
/* ═══════════ 全局：纸张质感 ═══════════ */
.gradio-container {
    max-width: 1280px !important;
    background: #faf9f6 !important;
    font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    color: #1f2430;
    letter-spacing: 0.01em;
}
footer { display: none !important; }

/* ═══════════ 顶部：期刊眉题 ═══════════ */
#topbar { padding: 18px 4px 8px; margin-bottom: 8px; }
.masthead { padding: 2px 6px; }
.eyebrow {
    font-size: 11px;
    letter-spacing: 0.35em;
    color: #b08d57;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.masthead-title {
    font-family: Georgia, 'Noto Serif SC', 'Songti SC', serif;
    font-size: 26px;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0 0 4px;
    letter-spacing: 0.02em;
}
.masthead-sub {
    font-size: 12.5px;
    color: #8a8f98;
    margin: 0;
    letter-spacing: 0.05em;
}
#toggle-btn {
    max-width: 110px;
    font-size: 12.5px;
    background: transparent !important;
    border: 1px solid #ddd8cc !important;
    color: #6b7280 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
#toggle-btn:hover { border-color: #1e3a5f !important; color: #1e3a5f !important; }

/* ═══════════ 侧边栏：白卡 + 墨蓝书脊 ═══════════ */
#sidebar {
    background: #ffffff !important;
    border: 1px solid #e7e3d8;
    border-left: 3px solid #1e3a5f;
    border-radius: 4px;
    padding: 20px 16px;
    min-height: 560px;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.04);
}
#sidebar h2 {
    font-family: Georgia, 'Noto Serif SC', serif;
    font-size: 13px;
    color: #b08d57;
    letter-spacing: 0.25em;
    font-weight: 600;
    text-transform: uppercase;
    margin: 4px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #efece3;
}
#sidebar h3 {
    font-family: Georgia, 'Noto Serif SC', serif;
    font-size: 13px;
    color: #1e3a5f;
    letter-spacing: 0.2em;
    font-weight: 600;
    text-transform: uppercase;
    margin: 22px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #f0ede4;
}
#sidebar .nav-btn {
    width: 100%;
    text-align: left;
    margin-bottom: 4px;
    background: transparent !important;
    color: #4a5261 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 9px 12px !important;
    font-size: 13.5px;
    font-weight: 400;
    box-shadow: none !important;
    border-left: 2px solid transparent !important;
    transition: all 0.15s ease;
}
#sidebar .nav-btn:hover {
    background: #f4f1ea !important;
    color: #1e3a5f !important;
    border-left: 2px solid #b08d57 !important;
}

/* 状态徽章 */
.badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 400;
    padding: 2px 10px;
    border-radius: 999px;
    margin-left: 4px;
}
.badge.ok { color: #2f7d4f; background: #e9f5ec; }
.badge.warn { color: #9a6b1f; background: #faf3e2; }

/* ═══════════ 主区面板 ═══════════ */
#main-area .panel { animation: fadeIn 0.25s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

#main-area h2 {
    font-family: Georgia, 'Noto Serif SC', serif;
    font-size: 20px;
    color: #1e3a5f;
    font-weight: 700;
    margin: 8px 0 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #efece3;
    position: relative;
}
#main-area h2::after {
    content: '';
    position: absolute;
    left: 0; bottom: -2px;
    width: 52px;
    height: 2px;
    background: #b08d57;
}
#main-area > div > .markdown p {
    color: #6b7280;
    font-size: 13.5px;
    line-height: 1.75;
}

/* 输入控件：细边框 */
#main-area input, #main-area textarea, #main-area .file-preview, .gradio-dropdown {
    border-color: #ddd8cc !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    background: #ffffff !important;
}
#main-area textarea:focus, #main-area input:focus {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 0 2px rgba(30, 58, 95, 0.08) !important;
}

/* 按钮：克制的学术色 */
#main-area .primary, .gradio-button.primary {
    background: #1e3a5f !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    box-shadow: none !important;
}
#main-area .primary:hover { background: #16314f !important; }
.gradio-button.secondary, .gradio-button.sm {
    background: transparent !important;
    border: 1px solid #d5d0c2 !important;
    border-radius: 6px !important;
    color: #5a6270 !important;
    box-shadow: none !important;
}
.gradio-button.secondary:hover { border-color: #1e3a5f !important; color: #1e3a5f !important; }

/* ═══════════ 聊天区：论文评注风 ═══════════ */
#chat-area {
    background: #ffffff !important;
    border: 1px solid #e7e3d8 !important;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.04);
}
#chat-area .chat-window { background: #fdfcfa !important; }
#chat-area .message {
    font-family: Georgia, 'Noto Serif SC', 'Songti SC', serif;
    color: #2a3040 !important;
    line-height: 1.75;
    font-size: 14.5px;
}
#chat-area .message-user {
    background: #1e3a5f !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
    color: #f2f4f8 !important;
}
#chat-area .message-bot {
    background: #ffffff !important;
    border: 1px solid #e9e5da !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
#chat-area .message-bot pre { background: #f6f4ee !important; border-radius: 4px; }

/* 滚动条 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #d8d3c5; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #b8b2a0; }
::-webkit-scrollbar-track { background: transparent; }

/* 分隔线 */
hr { border-color: #efece3 !important; }
"""


# ── UI 布局 ───────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="数学论文智能问答助手") as app:
        # 侧边栏状态 / 当前会话状态
        sidebar_state = gr.State(True)
        session_state = gr.State("")

        # 顶部栏：切换按钮 + 期刊风格标题
        with gr.Row(elem_id="topbar"):
            toggle_btn = gr.Button("≪ 收起", elem_id="toggle-btn", variant="secondary")
            gr.HTML(
                """
                <div class="masthead">
                    <div class="eyebrow">Math RAG Assistant · Hybrid Retrieval</div>
                    <h1 class="masthead-title">数学论文智能问答助手</h1>
                    <p class="masthead-sub">对任意论文建库 · 带章节级引用的多轮问答</p>
                </div>
                """
            )

        with gr.Row():
            # ── 侧边栏（可隐藏）──
            with gr.Column(scale=1, elem_id="sidebar", visible=True) as sidebar:
                gr.Markdown("## 导航")

                nav_config = gr.Button("01 · 配置密钥", elem_classes=["nav-btn"])
                nav_upload = gr.Button("02 · 上传论文", elem_classes=["nav-btn"])
                nav_chat = gr.Button("03 · 智能问答", elem_classes=["nav-btn"])

                gr.Markdown("---")
                gr.Markdown("### 状态")
                api_status = gr.Markdown(api_status_html())
                index_status = gr.Markdown(index_status_html())

                # 历史对话
                gr.Markdown("---")
                gr.Markdown("### 历史对话")
                session_drop = gr.Dropdown(
                    label="选择会话",
                    choices=_session_choices(),
                    interactive=True,
                )
                with gr.Row():
                    load_sess_btn = gr.Button("加载", size="sm")
                    del_sess_btn = gr.Button("删除", size="sm")
                new_chat_btn = gr.Button("新对话", size="sm", variant="secondary")

            # ── 主内容区 ──
            with gr.Column(scale=4, elem_id="main-area"):
                # 面板 1: 配置密钥
                with gr.Column(visible=True, elem_classes=["panel"]) as panel_config:
                    gr.Markdown("## 配置 DeepSeek API 密钥")
                    gr.Markdown(
                        "AI 生成回答需要调用 DeepSeek API。密钥只需配置一次，会自动保存到本地。\n\n"
                        "没有密钥？前往 [platform.deepseek.com](https://platform.deepseek.com/) 注册获取。"
                    )
                    api_key_input = gr.Textbox(
                        label="DeepSeek API Key",
                        placeholder="在此粘贴你的 API Key（以 sk- 开头）",
                        type="password",
                        lines=1,
                    )
                    save_btn = gr.Button("保存密钥", variant="primary")
                    save_result = gr.Textbox(label="结果", interactive=False)

                    save_btn.click(fn=handle_save_api_key, inputs=[api_key_input], outputs=[save_result])

                # 面板 2: 上传论文
                with gr.Column(visible=False, elem_classes=["panel"]) as panel_upload:
                    gr.Markdown("## 上传论文文件")
                    gr.Markdown("支持 PDF、Word、TXT、Markdown 格式。上传后点击「构建索引」。")

                    file_upload = gr.File(
                        label="选择论文文件",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                    )
                    with gr.Row():
                        upload_btn = gr.Button("保存文件", variant="secondary")
                        index_btn = gr.Button("构建索引", variant="primary")

                    upload_status = gr.Textbox(label="保存结果", interactive=False, lines=2)
                    index_result = gr.Textbox(label="索引结果", interactive=False, lines=3)

                    upload_btn.click(fn=handle_upload, inputs=[file_upload], outputs=[upload_status])
                    index_btn.click(fn=handle_build_index, outputs=[index_result, index_status])

                # 面板 3: 智能问答
                with gr.Column(visible=False, elem_classes=["panel"]) as panel_chat:
                    gr.Markdown("## 智能问答")
                    gr.Markdown("基于已索引的论文内容提问，AI 会给出带来源引用的回答。")

                    with gr.Column(elem_id="chat-area"):
                        chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            elem_classes=["chat-window"],
                            show_label=False,
                            # Gradio 6 原生 KaTeX 渲染块级公式
                            latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="",
                                placeholder="输入你的问题，按回车发送...",
                                lines=2,
                                elem_id="chat-input",
                                show_label=False,
                                scale=5,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)

                    clear_btn = gr.Button("清空对话", variant="secondary", size="sm")

                    send_btn.click(
                        fn=handle_chat,
                        inputs=[msg_input, chatbot, session_state],
                        outputs=[msg_input, chatbot, session_state, session_drop],
                    )
                    msg_input.submit(
                        fn=handle_chat,
                        inputs=[msg_input, chatbot, session_state],
                        outputs=[msg_input, chatbot, session_state, session_drop],
                    )
                    clear_btn.click(fn=handle_clear, outputs=[chatbot])

            # ── 事件绑定 ──
            toggle_btn.click(
                fn=toggle_sidebar,
                inputs=[sidebar_state],
                outputs=[sidebar, sidebar_state, toggle_btn],
            )
            nav_config.click(fn=show_config, outputs=[panel_config, panel_upload, panel_chat])
            nav_upload.click(fn=show_upload, outputs=[panel_config, panel_upload, panel_chat])
            nav_chat.click(fn=show_chat, outputs=[panel_config, panel_upload, panel_chat])

            # 历史会话
            load_sess_btn.click(
                fn=handle_load_session,
                inputs=[session_drop],
                outputs=[chatbot, panel_config, panel_upload, panel_chat, session_state],
            )
            del_sess_btn.click(
                fn=handle_delete_session,
                inputs=[session_drop],
                outputs=[chatbot, session_state, session_drop],
            )
            new_chat_btn.click(fn=handle_new_chat, outputs=[chatbot, session_state])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )