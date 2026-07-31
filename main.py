"""数学论文智能问答助手 - Gradio Web UI"""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr

import config
from core.pipeline import MathRAGPipeline

_pipeline: MathRAGPipeline | None = None


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


def handle_build_index() -> str:
    try:
        p = get_pipeline()
        msg = p.build_index()
        stats = p.stats
        return f"{msg}\n\n当前: {stats['paper_count']} 篇论文, {stats['chunk_count']} 个分块"
    except Exception as e:
        return f"构建失败: {e}"


def handle_chat(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not message.strip():
        return "", history

    if not config.has_api_key():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content":
            "尚未配置 API Key，无法生成回答。\n\n请点击左上角菜单，进入「配置密钥」完成配置。"
        })
        return "", history

    try:
        p = get_pipeline()
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"系统初始化失败: {e}"})
        return "", history

    if p.vector_store.size == 0:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content":
            "尚未构建索引，无法检索论文内容。\n\n请点击左上角菜单，进入「上传论文」上传文件并构建索引。"
        })
        return "", history

    try:
        answer, sources = p.query(message)
        if sources:
            source_lines = ["\n\n---\n**参考来源:**"]
            for i, s in enumerate(sources, 1):
                source_lines.append(f"{i}. {s.source} (相关度: {s.score:.3f})")
            answer = answer + "\n".join(source_lines)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"处理失败: {e}"})

    return "", history


def handle_clear() -> list[dict]:
    return []


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
/* 全局 */
.gradio-container {
    font-family: 'Georgia', 'Times New Roman', serif;
    max-width: 1200px !important;
}

/* 顶部栏 */
#topbar {
    padding: 8px 0;
    margin-bottom: 5px;
}
#toggle-btn {
    max-width: 120px;
    font-size: 14px;
}

/* 侧边栏 - 紧凑 */
#sidebar {
    background-color: #1a1a2e;
    border-radius: 10px;
    padding: 15px;
    min-height: 550px;
}
#sidebar h2 {
    color: #e0e0e0;
    font-size: 16px;
    border-bottom: 1px solid #2c3e50;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
#sidebar .nav-btn {
    width: 100%;
    text-align: left;
    margin-bottom: 5px;
    background-color: #16213e !important;
    color: #a0a0b0 !important;
    border: 1px solid #2c3e50 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-size: 13px;
}
#sidebar .nav-btn:hover {
    background-color: #0f3460 !important;
    color: #ffffff !important;
}
#sidebar .status-box {
    background-color: #16213e !important;
    border: 1px solid #2c3e50 !important;
    border-radius: 6px;
    color: #a0a0b0;
    font-size: 12px;
    padding: 6px 10px;
}

/* 聊天框深色 */
#chat-area {
    background-color: #1a1a2e !important;
    border: 1px solid #2c3e50 !important;
    border-radius: 10px;
    overflow: hidden;
}
#chat-area .chat-window { background-color: #0d1117 !important; }
#chat-area .message {
    font-family: 'Georgia', serif;
    color: #e0e0e0 !important;
    line-height: 1.6;
}
#chat-area .message-user {
    background-color: #0f3460 !important;
    border: 1px solid #1a5276 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
#chat-area .message-bot {
    background-color: #1a1a2e !important;
    border: 1px solid #2c3e50 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
"""


# ── UI 布局 ───────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="数学论文智能问答助手") as app:
        # 侧边栏状态
        sidebar_state = gr.State(True)

        # 顶部栏：切换按钮 + 标题
        with gr.Row(elem_id="topbar"):
            toggle_btn = gr.Button("<< 收起菜单", elem_id="toggle-btn", variant="secondary")
            gr.Markdown("### 数学论文智能问答助手")

        with gr.Row():
            # ── 侧边栏（可隐藏）──
            with gr.Column(scale=1, elem_id="sidebar", visible=True) as sidebar:
                gr.Markdown("## 导航")

                nav_config = gr.Button("配置密钥", elem_classes=["nav-btn"])
                nav_upload = gr.Button("上传论文", elem_classes=["nav-btn"])
                nav_chat = gr.Button("智能问答", elem_classes=["nav-btn"])

                gr.Markdown("---")
                gr.Markdown("### 状态")
                api_status = gr.Textbox(
                    label="API 密钥",
                    value="已配置" if config.has_api_key() else "未配置",
                    interactive=False,
                    elem_classes=["status-box"],
                )
                index_status = gr.Textbox(
                    label="论文索引",
                    value="未构建",
                    interactive=False,
                    elem_classes=["status-box"],
                )

            # ── 主内容区 ──
            with gr.Column(scale=4):

                # 面板 1: 配置密钥
                with gr.Column(visible=True) as panel_config:
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
                with gr.Column(visible=False) as panel_upload:
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
                    index_btn.click(fn=handle_build_index, outputs=[index_result])

                # 面板 3: 智能问答
                with gr.Column(visible=False) as panel_chat:
                    gr.Markdown("## 智能问答")
                    gr.Markdown("基于已索引的论文内容提问，AI 会给出带来源引用的回答。")

                    with gr.Column(elem_id="chat-area"):
                        chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            elem_classes=["chat-window"],
                            show_label=False,
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

                    send_btn.click(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
                    msg_input.submit(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
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

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )