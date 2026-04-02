"""OpenAI 兼容 Chat + 工具循环（furniture_search / show_list_add）。"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import ValidationError

from agent.config import (
    DEFAULT_ARK_IMAGE_API_KEY,
    DEFAULT_ARK_IMAGE_BASE_URL,
    DEFAULT_ARK_IMAGE_MODEL,
    DEFAULT_ARK_IMAGE_PROMPT,
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOOL_ROUNDS,
    SUPPORTED_ARK_IMAGE_FORMATS,
)
from agent.skill_loader import build_system_prompt
from agent.tools.tool_list import build_tools
from core.state import SessionState

logger = logging.getLogger(__name__)

_XML_INVOKE_RE = re.compile(r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>", re.DOTALL)


def _parse_xml_tool_calls(content: str) -> list[dict]:
    """解析 MiniMax 等模型在 content 里以 XML 格式输出的工具调用。"""
    results = []
    for m in _XML_INVOKE_RE.finditer(content):
        name = m.group(1).strip()
        args: dict = {}
        for pm in _XML_PARAM_RE.finditer(m.group(2)):
            raw = pm.group(2).strip()
            try:
                args[pm.group(1)] = json.loads(raw)
            except json.JSONDecodeError:
                args[pm.group(1)] = raw
        results.append({"id": f"xml_{name}_{len(results)}", "name": name, "args": args})
    return results


def _strip_xml_tool_calls(content: str) -> str:
    """从文本中移除 XML 格式工具调用块及前置标签（minimax:tool_call 等）。"""
    cleaned = _XML_INVOKE_RE.sub("", content)
    cleaned = re.sub(r"minimax:tool_call\s*", "", cleaned)
    return cleaned.strip()


def _dedupe_image_refs(image_refs: List[str]) -> List[str]:
    """去重并清理空字符串，保持原始顺序。"""
    cleaned: List[str] = []
    seen: set[str] = set()
    for ref in image_refs:
        value = ref.strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned


def _encode_local_image_as_data_uri(path: Path) -> str:
    """将本地图片编码为符合 Ark 文档要求的 data URI。"""
    ext = path.suffix.lower()
    image_format = SUPPORTED_ARK_IMAGE_FORMATS.get(ext)
    if not image_format:
        supported = ", ".join(sorted({fmt for fmt in SUPPORTED_ARK_IMAGE_FORMATS.values()}))
        raise ValueError(
            f"不支持的图片格式: {path.suffix or '（无扩展名）'}。"
            f"当前仅支持: {supported}。"
        )

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{image_format};base64,{encoded}"


def _resolve_local_image_path(image_ref: str) -> Path | None:
    """解析本地图片路径，兼容绝对路径和相对于 backend/ 的资源路径。"""
    path = Path(image_ref)
    if path.is_file():
        return path

    backend_relative = Path(__file__).resolve().parents[1] / image_ref
    if backend_relative.is_file():
        return backend_relative

    return None


def _normalize_image_input(image_ref: str) -> str:
    """
    归一化图片输入：
    - http/https/data URL：原样透传
    - 本地文件路径：读取后转为 data:image/<格式>;base64,<编码>
    - 其他字符串：原样透传，便于兼容调用方预先传入完整 data URI
    """
    value = image_ref.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return value

    path = _resolve_local_image_path(value)
    if path is not None:
        return _encode_local_image_as_data_uri(path)

    return value


def build_image_to_image_inputs(
    session: SessionState,
    *,
    uploaded_image_urls: List[str] | None = None,
    room_image_url: str | None = None,
) -> List[str]:
    """
    汇总多图生图输入。

    顺序约定：
    1. 房间图（优先显式传入 room_image_url，否则回退到 session.room_image_url）
    2. User-List 中家具图
    3. 本轮用户额外上传的图片
    """
    image_refs: List[str] = []

    room_ref = (room_image_url or session.room_image_url or "").strip()
    if room_ref:
        image_refs.append(room_ref)

    for item in session.user_list:
        item_ref = (item.image_url or "").strip()
        if item_ref:
            image_refs.append(item_ref)

    for uploaded in uploaded_image_urls or []:
        if isinstance(uploaded, str):
            image_refs.append(uploaded)

    return _dedupe_image_refs(image_refs)


def generate_room_image_from_multiple_inputs(
    session: SessionState,
    *,
    api_key: str | None = None,
    uploaded_image_urls: List[str] | None = None,
    room_image_url: str | None = None,
    base_url: str = DEFAULT_ARK_IMAGE_BASE_URL,
    model: str = DEFAULT_ARK_IMAGE_MODEL,
    size: str = "2K",
    response_format: str = "url",
    watermark: bool = True,
    sequential_image_generation: str = "disabled",
) -> dict:
    """
    调用 Ark 多图生图接口，将房间图、已选家具图和用户上传图合成为一张效果图。

    返回:
        {
            "image_url": str,
            "input_images": list[str],
            "model": str,
            "prompt": str,
        }
    """
    effective_api_key = (api_key or DEFAULT_ARK_IMAGE_API_KEY).strip()
    if not effective_api_key:
        raise ValueError("缺少 Ark API Key。")

    image_refs = build_image_to_image_inputs(
        session,
        uploaded_image_urls=uploaded_image_urls,
        room_image_url=room_image_url,
    )
    if len(image_refs) < 2:
        raise ValueError(
            "至少需要 2 张图片才能进行多图生图：通常应包含 1 张房间图，以及至少 1 张家具图或用户上传图。"
        )
    normalized_images = [_normalize_image_input(ref) for ref in image_refs]

    client = OpenAI(
        base_url=base_url,
        api_key=effective_api_key,
    )
    response = client.images.generate(
        model=model,
        prompt=DEFAULT_ARK_IMAGE_PROMPT,
        size=size,
        response_format=response_format,
        extra_body={
            "image": normalized_images,
            "watermark": watermark,
            "sequential_image_generation": sequential_image_generation,
        },
    )

    data = getattr(response, "data", None) or []
    first = data[0] if data else None
    image_url = getattr(first, "url", None)
    if not image_url:
        raise ValueError("图片生成成功返回，但未拿到结果图片 URL。")

    return {
        "image_url": image_url,
        "input_images": image_refs,
        "model": model,
        "prompt": DEFAULT_ARK_IMAGE_PROMPT,
    }


def run_chat_turn(
    session: SessionState,
    prior_messages: List[BaseMessage],
    user_text: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
) -> Tuple[str, List[BaseMessage]]:
    """
    执行一轮用户消息：system 由 SKILL + 动态会话拼接；返回 (assistant 可见文本, 本轮起需持久化的消息片段)。
    prior_messages 不含 SystemMessage；返回的片段从本轮 HumanMessage 起至最后一轮 AIMessage（含中间 ToolMessage）。
    """
    tools = build_tools(session)
    if session.workflow_phase == "finished":
        tools = [t for t in tools if t.name != "show_list_add"]
    tool_by_name = {t.name: t for t in tools}

    system_content = build_system_prompt(session)
    messages: List[BaseMessage] = [
        SystemMessage(content=system_content),
        *prior_messages,
        HumanMessage(content=user_text),
    ]

    effective_api_key = (api_key or DEFAULT_CHAT_API_KEY).strip()
    if not effective_api_key:
        raise ValueError("缺少对话模型 API Key，请在环境变量 OPENAI_API_KEY 中配置。")

    llm = ChatOpenAI(
        model=(model or DEFAULT_CHAT_MODEL).strip(),
        api_key=effective_api_key,
        base_url=base_url if base_url is not None else DEFAULT_CHAT_BASE_URL,
        temperature=0.2,
    ).bind_tools(tools)

    last_recommendation_text: str | None = None

    rounds = 0
    while rounds < max_tool_rounds:
        rounds += 1
        ai = llm.invoke(messages)
        if not isinstance(ai, AIMessage):
            raise TypeError(f"expected AIMessage, got {type(ai)}")
        messages.append(ai)

        # 标准 OpenAI tool_calls
        tool_call_list = list(ai.tool_calls) if ai.tool_calls else []

        # 若标准格式为空，尝试解析 MiniMax XML 格式
        if not tool_call_list and isinstance(ai.content, str):
            tool_call_list = _parse_xml_tool_calls(ai.content)

        _dbg_sep = "─" * 60
        if tool_call_list:
            print(f"\n{_dbg_sep}")
            print(f"[Round {rounds}] 🔧 工具调用 × {len(tool_call_list)}")
            if ai.content:
                print(f"  raw content: {ai.content[:300]}{'...' if len(str(ai.content)) > 300 else ''}")
        else:
            print(f"\n{_dbg_sep}")
            print(f"[Round {rounds}] 💬 最终回复")
            raw_preview = ai.content if isinstance(ai.content, str) else str(ai.content)
            print(f"  {raw_preview[:500]}{'...' if len(raw_preview) > 500 else ''}")
            print(_dbg_sep)
            break

        for tc in tool_call_list:
            tid = (tc.get("id") or "") if isinstance(tc, dict) else getattr(tc, "id", "") or ""
            name = (tc.get("name") or "") if isinstance(tc, dict) else getattr(tc, "name", "") or ""
            args = (tc.get("args") or {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = dict(args) if hasattr(args, "items") else {}

            print(f"  ▶ {name}({json.dumps(args, ensure_ascii=False, separators=(',', ':'))[:300]})")

            tool = tool_by_name.get(name)
            if tool is None:
                err = {
                    "error": f"未知工具: {name}",
                    "valid_tools": list(tool_by_name),
                }
                print(f"  ✗ 未知工具: {name}")
                messages.append(ToolMessage(content=json.dumps(err, ensure_ascii=False), tool_call_id=tid))
                continue

            try:
                out = tool.invoke(args)
                if not isinstance(out, str):
                    out = str(out)
            except ValidationError as e:
                out = f"参数校验失败，请按工具 schema 修正后重试：{e!s}"
            except json.JSONDecodeError as e:
                out = f"工具入参 JSON 解析问题（若存在）：{e!s}"
            except Exception as e:
                logger.exception("工具执行异常: %s", name)
                err = {"error": type(e).__name__, "message": str(e)}
                out = json.dumps(err, ensure_ascii=False)

            print(f"  ◀ result: {out[:300]}{'...' if len(out) > 300 else ''}")

            # 从 show_list_add 结果中提取推荐理由备用
            if name == "show_list_add":
                try:
                    report = json.loads(out)
                    reason = report.get("recommendation_reason_markdown", "")
                    if reason:
                        last_recommendation_text = reason
                except (json.JSONDecodeError, AttributeError):
                    pass

            messages.append(ToolMessage(content=out, tool_call_id=tid))

        print(_dbg_sep)

    last = messages[-1]
    if isinstance(last, AIMessage):
        raw = last.content
        if isinstance(raw, str):
            # 剥离 MiniMax XML 工具调用块，避免把原始标签渲染给用户
            final_text = _strip_xml_tool_calls(raw)
        elif isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            final_text = _strip_xml_tool_calls("".join(parts))
        else:
            final_text = str(raw) if raw else ""
    else:
        final_text = ""

    # 若最终文本为空（MiniMax 在 XML 工具调用后不再生成文本），
    # 使用 show_list_add 传入的 recommendation_reason_markdown 作为回复
    if not final_text and last_recommendation_text:
        final_text = last_recommendation_text

    start = 1 + len(prior_messages)
    tail = messages[start:]
    return final_text, tail
