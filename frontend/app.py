"""
AI-HomeFlow Streamlit 原型：双列表 + Agent 工具调用。
启动（仓库根目录）：streamlit run frontend/app.py
需已配置 PYTHONPATH 包含 backend，或使用下方自动注入。
"""

from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
import warnings
from pathlib import Path

# 保证可导入 backend 下的 agent / core / services
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The `device` argument is deprecated and will be removed in v5 of Transformers\.",
    category=FutureWarning,
)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from langchain_core.messages import BaseMessage

from agent.config import (
    DEFAULT_ARK_IMAGE_API_KEY,
    DEFAULT_ARK_IMAGE_BASE_URL,
    DEFAULT_ARK_IMAGE_MODEL,
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_MODEL,
)
from agent.runner import (
    generate_room_image_from_multiple_inputs,
    run_chat_turn,
)
from core.state import SessionState
from services.furniture_api import warmup_rag

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"]

def _resolve_image(image_url: str) -> str | None:
    """将相对路径（相对于 backend/）解析为绝对路径；在线 URL 直接返回；空值返回 None。"""
    if not image_url:
        return None
    if image_url.startswith(("http://", "https://", "data:")):
        return image_url
    local_path = Path(image_url)
    if local_path.is_absolute() and local_path.exists():
        return str(local_path)
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
    if "render_uploads" not in st.session_state:
        st.session_state.render_uploads: list[dict] = []
    if "render_room_image_id" not in st.session_state:
        st.session_state.render_room_image_id = None
    if "generated_room_image_url" not in st.session_state:
        st.session_state.generated_room_image_url = None
    if "generated_room_image_error" not in st.session_state:
        st.session_state.generated_room_image_error = None
    if "generated_room_image_inputs" not in st.session_state:
        st.session_state.generated_room_image_inputs: list[str] = []


def _upload_cache_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "ai-homeflow" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_uploaded_file(uploaded_file) -> dict:
    data = uploaded_file.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    suffix = Path(uploaded_file.name).suffix.lower() or ".img"
    stored_path = _upload_cache_dir() / f"{digest}{suffix}"
    if not stored_path.exists():
        stored_path.write_bytes(data)
    return {
        "id": f"{digest}:{uploaded_file.name}",
        "name": uploaded_file.name,
        "path": str(stored_path),
        "size": len(data),
        "mime": getattr(uploaded_file, "type", "") or "",
    }


def _sync_render_uploads(uploaded_files) -> None:
    previous_ids = [item["id"] for item in st.session_state.render_uploads]
    current = [_persist_uploaded_file(file) for file in uploaded_files or []]
    current_ids = [item["id"] for item in current]
    st.session_state.render_uploads = current

    selected_id = st.session_state.render_room_image_id
    if selected_id not in set(current_ids):
        st.session_state.render_room_image_id = current[0]["id"] if current else None

    if current_ids != previous_ids:
        st.session_state.generated_room_image_url = None
        st.session_state.generated_room_image_error = None
        st.session_state.generated_room_image_inputs = []


def _run_image_generation(
    session: SessionState,
) -> str:
    uploads = st.session_state.render_uploads
    if not uploads:
        raise ValueError("请先上传至少 1 张房间或参考图片。")
    if not session.user_list:
        raise ValueError("请先将至少 1 件家具加入 User-List，再进入图生图阶段。")

    room_id = st.session_state.render_room_image_id
    room_entry = next((item for item in uploads if item["id"] == room_id), None)
    if room_entry is None:
        raise ValueError("请从上传图片中选择 1 张作为房间照片。")

    extra_refs = [item["path"] for item in uploads if item["id"] != room_id]
    result = generate_room_image_from_multiple_inputs(
        session=session,
        api_key=DEFAULT_ARK_IMAGE_API_KEY,
        room_image_url=room_entry["path"],
        uploaded_image_urls=extra_refs,
        base_url=DEFAULT_ARK_IMAGE_BASE_URL,
        model=DEFAULT_ARK_IMAGE_MODEL,
    )
    st.session_state.generated_room_image_url = result["image_url"]
    st.session_state.generated_room_image_error = None
    st.session_state.generated_room_image_inputs = result["input_images"]
    return result["image_url"]


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
                st.image(img, width="stretch")
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
                st.image(img, width="stretch")
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
    session.style_preference = None

    st.title("AI-HomeFlow")
    st.caption("先通过对话筛选家具，再上传房间图片并在确认清单后生成效果图。")

    with st.sidebar:
        st.header("当前配置")
        st.caption(f"对话模型：`{DEFAULT_CHAT_MODEL}`")
        st.caption(f"图像模型：`{DEFAULT_ARK_IMAGE_MODEL}`")
        if not DEFAULT_CHAT_API_KEY:
            st.warning("未检测到 `OPENAI_API_KEY`，当前无法进行对话。")
        if not DEFAULT_ARK_IMAGE_API_KEY:
            st.warning("未检测到 `ARK_API_KEY`，当前无法进行图生图。")
        st.divider()
        if st.button("重置会话", type="secondary"):
            st.session_state.homeflow_session = SessionState()
            st.session_state.lc_messages = []
            st.session_state.ui_chat_log = []
            st.session_state.render_uploads = []
            st.session_state.render_room_image_id = None
            st.session_state.generated_room_image_url = None
            st.session_state.generated_room_image_error = None
            st.session_state.generated_room_image_inputs = []
            st.rerun()

    col_chat, col_lists = st.columns([1.1, 0.9], gap="large")

    with col_chat:
        st.subheader("对话与上传")
        uploaded_files = st.file_uploader(
            "上传房间/参考图片",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="建议至少上传 1 张房间照片；这些图片会在确认清单完成后与 User-List 家具图一起用于图生图。",
        )
        _sync_render_uploads(uploaded_files)

        if st.session_state.render_uploads:
            room_options = [item["id"] for item in st.session_state.render_uploads]
            selected_room_id = st.selectbox(
                "选择房间照片",
                options=room_options,
                index=room_options.index(st.session_state.render_room_image_id)
                if st.session_state.render_room_image_id in room_options
                else 0,
                format_func=lambda item_id: next(
                    item["name"] for item in st.session_state.render_uploads if item["id"] == item_id
                ),
            )
            st.session_state.render_room_image_id = selected_room_id

            preview_cols = st.columns(min(4, len(st.session_state.render_uploads)))
            for idx, item in enumerate(st.session_state.render_uploads):
                with preview_cols[idx % len(preview_cols)]:
                    img = _resolve_image(item["path"])
                    if img:
                        st.image(img, width=110)
                    tag = "房间图" if item["id"] == st.session_state.render_room_image_id else "参考图"
                    st.caption(f"{tag} · {item['name']}")
        else:
            st.caption("可在此提前上传房间图或参考图，系统会缓存到 Finish 阶段再调用图生图。")

        chat_container = st.container(height=420, border=True)
        with chat_container:
            if not st.session_state.ui_chat_log:
                st.caption("在下方输入需求，Agent 会把合适的家具加入候选池。")
            for role, text in st.session_state.ui_chat_log:
                with st.chat_message(role):
                    st.markdown(text)

        user_input = st.chat_input(
            "描述需求，Agent 检索后将推荐追加到候选池",
            disabled=session.workflow_phase == "finished",
        )
        if session.workflow_phase == "finished":
            st.caption("已确认清单完成；对话已关闭。可在侧栏重置会话。")

        if user_input and DEFAULT_CHAT_API_KEY:
            st.session_state.ui_chat_log.append(("user", user_input))
            with st.spinner("Agent 思考中…"):
                try:
                    reply, tail = run_chat_turn(
                        session,
                        list(st.session_state.lc_messages),
                        user_input,
                    )
                    st.session_state.lc_messages.extend(tail)
                except Exception as e:
                    reply = f"调用失败：{type(e).__name__}: {e}"
            st.session_state.ui_chat_log.append(("assistant", reply or "（无文本回复）"))
            st.rerun()
        elif user_input and not DEFAULT_CHAT_API_KEY:
            st.warning("请先在环境变量中配置 `OPENAI_API_KEY`。")

    with col_lists:
        st.subheader("候选与决策")
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
            finish_disabled = (
                not session.user_list
                or not st.session_state.render_uploads
                or not DEFAULT_ARK_IMAGE_API_KEY
            )
            if not session.user_list:
                st.caption("请先把至少 1 件家具加入 User-List。")
            elif not st.session_state.render_uploads:
                st.caption("请先在对话区上传至少 1 张房间或参考图片。")
            elif not DEFAULT_ARK_IMAGE_API_KEY:
                st.caption("请先在环境变量中配置 `ARK_API_KEY`，才能在 Finish 后进入图生图阶段。")

            if st.button("确认清单完成（Finish）", type="secondary", disabled=finish_disabled):
                session.workflow_phase = "finished"
                with st.spinner("正在根据上传照片与决策池家具生成效果图…"):
                    try:
                        image_url = _run_image_generation(session)
                        msg = (
                            "已记录：你确认了当前决策池清单。后续将不再向候选池写入推荐。"
                            f"\n\n已生成房间效果图：{image_url}"
                        )
                    except Exception as e:
                        st.session_state.generated_room_image_url = None
                        st.session_state.generated_room_image_error = f"{type(e).__name__}: {e}"
                        msg = (
                            "已记录：你确认了当前决策池清单。后续将不再向候选池写入推荐。"
                            f"\n\n图生图失败：{type(e).__name__}: {e}"
                        )
                st.session_state.ui_chat_log.append(("assistant", msg))
                st.rerun()
        else:
            st.success("当前阶段：finished（已确认清单）")
            if st.button(
                "重新生成效果图",
                type="primary",
                disabled=(
                    not st.session_state.render_uploads
                    or not session.user_list
                    or not DEFAULT_ARK_IMAGE_API_KEY
                ),
            ):
                with st.spinner("正在重新生成效果图…"):
                    try:
                        _run_image_generation(session)
                    except Exception as e:
                        st.session_state.generated_room_image_url = None
                        st.session_state.generated_room_image_error = f"{type(e).__name__}: {e}"
                st.rerun()

        st.divider()
        st.subheader("图生图结果")
        if st.session_state.generated_room_image_url:
            st.image(st.session_state.generated_room_image_url, width="stretch")
            st.caption("已使用上传照片与 User-List 家具图生成。")
        elif st.session_state.generated_room_image_error:
            st.error(st.session_state.generated_room_image_error)
        else:
            st.info("确认清单完成后，将在这里展示生成的房间效果图。")


main()
