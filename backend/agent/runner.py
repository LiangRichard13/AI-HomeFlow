"""OpenAI 兼容 Chat + 工具循环（furniture_search / show_list_add）。"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from agent.skill_loader import build_system_prompt
from agent.tools.tool_list import build_tools
from core.state import SessionState

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ROUNDS = 8

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


def run_chat_turn(
    session: SessionState,
    prior_messages: List[BaseMessage],
    user_text: str,
    *,
    api_key: str,
    model: str,
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

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
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
