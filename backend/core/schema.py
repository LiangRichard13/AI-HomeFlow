"""README 定义的数据 Schema（逻辑验证与向量检索）。"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class FurnitureCategory(str, Enum):
    SOFA = "sofa"
    TABLE = "table"
    BED = "bed"
    CHAIR = "chair"
    CABINET = "cabinet"
    LIGHTING = "lighting"
    OTHER = "other"


class Dimensions(BaseModel):
    w: float = Field(..., description="宽度 cm")
    d: float = Field(..., description="深度 cm")
    h: float = Field(..., description="高度 cm")


class FurnitureItem(BaseModel):
    id: str
    category: FurnitureCategory
    price: float
    dimensions: Dimensions
    description: str
    style_tags: List[str]
    image_url: str
