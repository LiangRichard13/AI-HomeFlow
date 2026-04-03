"""Agent 可调用的工具：家具向量检索（对接 search_furniture）与 Show-List 候选池写入。"""

from __future__ import annotations

import json
from typing import Annotated, List, Optional, TypedDict

from langchain_core.tools import tool
from pydantic import Field

from core.schema import FurnitureCategory, furniture_category_enum_hint
from core.state import SessionState
from services.furniture_api import load_catalog, search_furniture

_CATEGORY_ENUM_TEXT = furniture_category_enum_hint()

# 供模型构造 description 查询时对齐向量库中的文本形态（与 ingest 脚本一致）
DESCRIPTION_FOR_RETRIEVAL = """
向量库中单条文档的检索文本形态示例（metadata 仅示意，query 里可写自然语言或仿照正文）：
【名称】多功能折叠沙发床
【风格与场景】小户型、多功能、现代
【材质与颜色】材质：……；可选颜色：……
【详情】三人位可展开为双人床，适合书房兼客房。坐垫可拆洗，租房与刚需户型性价比高。

传入本工具的 `description` 时：用简短中文或中英文关键词概括用户需求（风格、场景、材质、功能、预算相关表述等均可）；不必抄全模板，但与「名称/风格/详情」越接近，排序通常越准。无向量需求时可传空或省略。
""".strip()

_FURNITURE_SEARCH_TOOL_DESCRIPTION = (
    "在用户需要找家具、缩小候选范围或按语义挑选时使用。"
    "执行顺序：先用 category 与可选的价格、尺寸（单位见各参数说明）做硬过滤；"
    "若提供 description，则在过滤后的集合内按向量相似度重排，最多返回 20 条 JSON 对象（含 id、category、price、dimensions、description、style_tags、image_url）。"
    "若用户要跨类目，需分多次调用不同 category。\n\n"
    f"category {_CATEGORY_ENUM_TEXT}\n\n"
    "description 与索引正文对齐说明：\n\n"
    + DESCRIPTION_FOR_RETRIEVAL
)

_SHOW_LIST_ADD_TOOL_DESCRIPTION = (
    "在用户已选定要向界面「候选池 / Show-List」展示的一组家具时使用；"
    "须在拿到可信的家具 id 后调用（通常来自同一会话中 furniture_search 的返回字段 `id`，勿编造 id）。"
    "按 furniture_ids 的顺序将尚未在 Show-List 中的条目追加进去；已存在的 id 会跳过。"
    "本工具只负责写入候选池，不负责承载大段推荐理由；推荐说明应写在你给用户的正常回复正文里。"
)

# 与下方 Field、openai_tool_schemas 共用，避免两处文案漂移
P_FURNITURE_CATEGORY = f"必填。{_CATEGORY_ENUM_TEXT}"
P_PRICE_MIN = "可选。价格下限，单位：元（人民币）。不传或省略表示不限制。"
P_PRICE_MAX = "可选。价格上限，单位：元（人民币）。不传或省略表示不限制。"
P_W_MIN = "可选。家具外形宽度 w 的下限，单位：厘米。不传表示不限制。"
P_W_MAX = "可选。家具外形宽度 w 的上限，单位：厘米。不传表示不限制。"
P_D_MIN = "可选。家具深度 d 的下限，单位：厘米。不传表示不限制。"
P_D_MAX = "可选。家具深度 d 的上限，单位：厘米。不传表示不限制。"
P_H_MIN = "可选。家具高度 h 的下限，单位：厘米。不传表示不限制。"
P_H_MAX = "可选。家具高度 h 的上限，单位：厘米。不传表示不限制。"
P_SEMANTIC_QUERY = (
    "可选。用户自然语言需求摘要，用于在硬过滤后的结果里做向量相似度排序；"
    "留空则不做语义排序（结果按 id 排序）。写法参见工具说明中的索引正文示例。"
)
P_SHOW_LIST_IDS = (
    "必填。家具 id 字符串列表，须与目录中一致（如 fur_001、fur_019）。"
    "通常直接使用最近一次 furniture_search 返回中每条记录的 `id` 字段；顺序表示展示优先级。"
    "勿使用名称或模糊描述代替 id。"
)


def build_tools(session: SessionState):
    """绑定会话状态后返回工具列表（LangChain bind_tools / ToolNode 可用）。"""
    catalog_by_id = {it.id: it for it in load_catalog()}

    @tool(description=_FURNITURE_SEARCH_TOOL_DESCRIPTION)
    def furniture_search(
        category: Annotated[FurnitureCategory, Field(description=P_FURNITURE_CATEGORY)],
        price_min: Annotated[Optional[float], Field(default=None, description=P_PRICE_MIN)] = None,
        price_max: Annotated[Optional[float], Field(default=None, description=P_PRICE_MAX)] = None,
        w_min: Annotated[Optional[float], Field(default=None, description=P_W_MIN)] = None,
        w_max: Annotated[Optional[float], Field(default=None, description=P_W_MAX)] = None,
        d_min: Annotated[Optional[float], Field(default=None, description=P_D_MIN)] = None,
        d_max: Annotated[Optional[float], Field(default=None, description=P_D_MAX)] = None,
        h_min: Annotated[Optional[float], Field(default=None, description=P_H_MIN)] = None,
        h_max: Annotated[Optional[float], Field(default=None, description=P_H_MAX)] = None,
        description: Annotated[Optional[str], Field(default=None, description=P_SEMANTIC_QUERY)] = None,
    ) -> str:
        """执行检索并返回 JSON 数组字符串（每项为家具字段）。"""
        items = search_furniture(
            category=category,
            price_min=price_min,
            price_max=price_max,
            w_min=w_min,
            w_max=w_max,
            d_min=d_min,
            d_max=d_max,
            h_min=h_min,
            h_max=h_max,
            description=description,
        )
        return json.dumps([m.model_dump(mode="json") for m in items], ensure_ascii=False)

    @tool(description=_SHOW_LIST_ADD_TOOL_DESCRIPTION)
    def show_list_add(
        furniture_ids: Annotated[List[str], Field(description=P_SHOW_LIST_IDS, min_length=1)],
    ) -> str:
        """写入 Show-List 并返回 JSON 报告（added / unknown_ids / show_list_size）。有实际新增时先清空候选池再写入。"""
        valid_new: List[str] = []
        missing: List[str] = []
        existing_ids = {x.id for x in session.show_list}
        for fid in furniture_ids:
            if catalog_by_id.get(fid) is None:
                missing.append(fid)
            elif fid not in existing_ids:
                valid_new.append(fid)

        # 有新条目时先清空，再写入本批推荐
        if valid_new:
            session.clear_show_list()

        added: List[str] = []
        for fid in valid_new:
            it = catalog_by_id[fid]
            session.show_list.append(it)
            added.append(fid)
        report = {
            "added_furniture_ids": added,
            "unknown_ids": missing,
            "show_list_size": len(session.show_list),
        }
        return json.dumps(report, ensure_ascii=False)

    return [furniture_search, show_list_add]


class FurnitureSearchOpenAIFunction(TypedDict):
    """OpenAI Chat Completions `tools` 中单条 function 的 JSON 形态（可选手写接入时使用）。"""

    type: str
    function: dict


def openai_tool_schemas() -> List[FurnitureSearchOpenAIFunction]:
    """与 `build_tools` 语义一致的 function calling schema（name/description/parameters）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "furniture_search",
                "description": _FURNITURE_SEARCH_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [e.value for e in FurnitureCategory],
                            "description": P_FURNITURE_CATEGORY,
                        },
                        "price_min": {"type": "number", "description": P_PRICE_MIN},
                        "price_max": {"type": "number", "description": P_PRICE_MAX},
                        "w_min": {"type": "number", "description": P_W_MIN},
                        "w_max": {"type": "number", "description": P_W_MAX},
                        "d_min": {"type": "number", "description": P_D_MIN},
                        "d_max": {"type": "number", "description": P_D_MAX},
                        "h_min": {"type": "number", "description": P_H_MIN},
                        "h_max": {"type": "number", "description": P_H_MAX},
                        "description": {"type": "string", "description": P_SEMANTIC_QUERY},
                    },
                    "required": ["category"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_list_add",
                "description": _SHOW_LIST_ADD_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "furniture_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": P_SHOW_LIST_IDS,
                        },
                    },
                    "required": ["furniture_ids"],
                },
            },
        },
    ]


__all__ = [
    "DESCRIPTION_FOR_RETRIEVAL",
    "build_tools",
    "openai_tool_schemas",
]
