"""
AI-HomeFlow Streamlit 原型：双列表 + Agent 工具调用。
启动（仓库根目录）：streamlit run frontend/app.py
需已配置 PYTHONPATH 包含 backend，或使用下方自动注入。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 保证可导入 backend 下的 agent / core / services
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import streamlit as st
from langchain_core.messages import BaseMessage

from agent.runner import run_chat_turn
from core.state import SessionState
from services.furniture_api import warmup_rag

def _resolve_image(image_url: str) -> str | None:
    """将相对路径（相对于 backend/）解析为绝对路径；在线 URL 直接返回；空值返回 None。"""
    if not image_url:
        return None
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url
    abs_path = _BACKEND / image_url
    return str(abs_path) if abs_path.exists() else None


@st.cache_resource(show_spinner="正在预热向量检索模型…")
def _warmup_rag_once() -> bool:
    """仅执行一次的 RAG 预热；cache_resource 保证跨 rerun 只调用一次，且在正确的 Streamlit 上下文中执行。"""
    try:
        warmup_rag()
    except Exception:
        pass
    return True


def _init_session_state() -> None:
    if "homeflow_session" not in st.session_state:
        st.session_state.homeflow_session = SessionState()
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages: list[BaseMessage] = []
    if "ui_chat_log" not in st.session_state:
        st.session_state.ui_chat_log: list[tuple[str, str]] = []


def _render_show_item(it, idx: int) -> None:
    """候选池：紧凑横排卡片（左小图 + 右文字）。"""
    cat = it.category.value if hasattr(it.category, "value") else it.category
    dims = it.dimensions
    dim_str = f"{dims.w:.0f}×{dims.d:.0f}×{dims.h:.0f} cm"
    style_str = "、".join(it.style_tags) if it.style_tags else "—"
    colors_str = "、".join(it.colors) if it.colors else "—"
    name_display = it.name or it.id

    with st.container(border=True):
        col_img, col_info = st.columns([1, 2])
        with col_img:
            img = _resolve_image(it.image_url)
            if img:
                st.image(img, use_container_width=True)
        with col_info:
            st.markdown(f"**{name_display}**")
            st.caption(f"`{it.id}` · `{cat}` · **¥{it.price:,.0f}**")
            st.caption(f"📐 {dim_str}　🪵 {it.material or '—'}")
            st.caption(f"🎨 {colors_str}　🏷️ {style_str}")
            if it.description:
                st.caption(it.description)


def _render_user_item(it, idx: int) -> None:
    """决策池：紧凑横排（小缩略图 + 文字 + 删除）。"""
    cat = it.category.value if hasattr(it.category, "value") else it.category
    name_display = it.name or it.id
    with st.container(border=True):
        col_img, col_info, col_btn = st.columns([1, 5, 1])
        with col_img:
            img = _resolve_image(it.image_url)
            if img:
                st.image(img, use_container_width=True)
        with col_info:
            st.caption(f"**{name_display}**　`{cat}` · ¥{it.price:,.0f}")
        with col_btn:
            if st.button("🗑️", key=f"del_user_{it.id}_{idx}", help="从决策池移除"):
                st.session_state.homeflow_session.user_list.pop(idx)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI-HomeFlow", layout="wide")
    _warmup_rag_once()
    _init_session_state()
    session: SessionState = st.session_state.homeflow_session

    with st.sidebar:
        st.header("连接")
        api_key = st.text_input(
            "API Key",
            type="password",
            help="OpenAI 或兼容服务的密钥；留空时尝试环境变量 OPENAI_API_KEY",
        )
        base_url = st.text_input(
            "Base URL（可选）",
            value=os.getenv("OPENAI_BASE_URL") or "",
            placeholder="默认 OpenAI；兼容网关请填完整 base URL",
        )
        model = st.text_input("模型", value="gpt-4o-mini")
        st.divider()
        st.header("偏好")
        style = st.text_input("风格偏好（写入会话）", value=session.style_preference or "")
        if style != (session.style_preference or ""):
            session.style_preference = style or None
        st.divider()
        if st.button("重置会话", type="secondary"):
            st.session_state.homeflow_session = SessionState()
            st.session_state.lc_messages = []
            st.session_state.ui_chat_log = []
            st.rerun()

    col_chat, col_lists = st.columns([1, 1], gap="large")

    with col_chat:
        st.subheader("对话")
        for role, text in st.session_state.ui_chat_log:
            with st.chat_message(role):
                st.markdown(text)

        user_input = st.chat_input(
            "描述需求，Agent 检索后将推荐追加到候选池",
            disabled=session.workflow_phase == "finished",
        )
        if session.workflow_phase == "finished":
            st.caption("已确认清单完成；对话已关闭。可在侧栏重置会话。")

        key = (api_key or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
        if user_input and key:
            st.session_state.ui_chat_log.append(("user", user_input))
            with st.spinner("Agent 思考中…"):
                try:
                    reply, tail = run_chat_turn(
                        session,
                        list(st.session_state.lc_messages),
                        user_input,
                        api_key=key,
                        model=model.strip() or "gpt-4o-mini",
                        base_url=base_url.strip() or os.getenv("OPENAI_BASE_URL") or None,
                    )
                    st.session_state.lc_messages.extend(tail)
                except Exception as e:
                    reply = f"调用失败：{type(e).__name__}: {e}"
            st.session_state.ui_chat_log.append(("assistant", reply or "（无文本回复）"))
            st.rerun()
        elif user_input and not key:
            st.warning("请填写侧栏 API Key 或设置环境变量 OPENAI_API_KEY。")

    with col_lists:
        # ── 候选池 ──────────────────────────────────────────────
        show_count = len(session.show_list)
        st.subheader(f"Show-List（候选池）{'  ' + str(show_count) + ' 件' if show_count else ''}")

        if not session.show_list:
            st.info("暂无候选；发送对话后由 Agent 追加写入。")
        else:
            picked_ids = []
            for idx, it in enumerate(session.show_list):
                col_cb, col_card = st.columns([0.05, 0.95])
                with col_cb:
                    checked = st.checkbox("", key=f"pick_{it.id}_{idx}", label_visibility="collapsed")
                    if checked:
                        picked_ids.append(it.id)
                with col_card:
                    _render_show_item(it, idx)

            if st.button("将勾选加入 User-List ✅", type="primary", disabled=not picked_ids):
                have = {x.id for x in session.user_list}
                for it in session.show_list:
                    if it.id in picked_ids and it.id not in have:
                        session.user_list.append(it)
                        have.add(it.id)
                session.clear_show_list()
                st.rerun()

        st.divider()

        # ── 决策池 ──────────────────────────────────────────────
        user_count = len(session.user_list)
        st.subheader(f"User-List（决策池）{'  ' + str(user_count) + ' 件' if user_count else ''}")
        st.metric("决策池总价", f"¥{session.user_list_total:,.2f}")

        if not session.user_list:
            st.info("决策池为空。")
        else:
            for idx, it in enumerate(session.user_list):
                _render_user_item(it, idx)

        st.divider()
        if session.workflow_phase == "browsing":
            if st.button("确认清单完成（Finish）", type="secondary"):
                session.workflow_phase = "finished"
                st.session_state.ui_chat_log.append(
                    ("assistant", "已记录：你确认了当前决策池清单。后续将不再向候选池写入推荐（可重置会话重新开始）。")
                )
                st.rerun()
        else:
            st.success("当前阶段：finished（已确认清单）")


main()
