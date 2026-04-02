"""README 定义的数据 Schema（逻辑验证与向量检索）。"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FurnitureCategory(str, Enum):
    SOFA = "sofa"
    TABLE = "table"
    BED = "bed"
    CHAIR = "chair"
    CABINET = "cabinet"
    LIGHTING = "lighting"
    OTHER = "other"


# Agent / 工具说明用中文释义；新增 FurnitureCategory 成员时必须同步增删此处键
FURNITURE_CATEGORY_LABELS: Dict[FurnitureCategory, str] = {
    FurnitureCategory.SOFA: "沙发",
    FurnitureCategory.TABLE: "桌子",
    FurnitureCategory.BED: "床",
    FurnitureCategory.CHAIR: "椅子",
    FurnitureCategory.CABINET: "柜类",
    FurnitureCategory.LIGHTING: "灯具",
    FurnitureCategory.OTHER: "其他",
}


def furniture_category_enum_hint() -> str:
    """与 FurnitureCategory 枚举同步的类目说明文案（供工具 description 等使用）。"""
    if set(FURNITURE_CATEGORY_LABELS) != set(FurnitureCategory):
        raise RuntimeError(
            "FURNITURE_CATEGORY_LABELS 与 FurnitureCategory 成员不一致，请同步 core/schema.py"
        )
    pairs = "，".join(f"{m.value}={FURNITURE_CATEGORY_LABELS[m]}" for m in FurnitureCategory)
    return f"必须为以下英文小写之一（勿用中文）：{pairs}。"


class Dimensions(BaseModel):
    w: float = Field(..., description="宽度 cm")
    d: float = Field(..., description="深度 cm")
    h: float = Field(..., description="高度 cm")


class FurnitureItem(BaseModel):
    id: str
    name: str = ""
    category: FurnitureCategory
    price: float
    dimensions: Dimensions
    description: str
    style_tags: List[str] = Field(default_factory=list)
    material: Optional[str] = None
    colors: List[str] = Field(default_factory=list)
    image_url: str = ""


class FurnitureSearchRequest(BaseModel):
    """家具搜索：category 必填；其余可选，用于硬过滤与可选向量排序。"""

    category: FurnitureCategory
    price_min: Optional[float] = Field(None, description="价格下限")
    price_max: Optional[float] = Field(None, description="价格上限")
    w_min: Optional[float] = Field(None, description="宽度 w 下限 cm")
    w_max: Optional[float] = Field(None, description="宽度 w 上限 cm")
    d_min: Optional[float] = Field(None, description="深度 d 下限 cm")
    d_max: Optional[float] = Field(None, description="深度 d 上限 cm")
    h_min: Optional[float] = Field(None, description="高度 h 下限 cm")
    h_max: Optional[float] = Field(None, description="高度 h 上限 cm")
    description: Optional[str] = Field(
        None,
        description="描述/需求文本；非空时在硬过滤结果内按向量相似度排序",
    )
