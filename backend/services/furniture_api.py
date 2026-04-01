"""硬性规则过滤：价格、尺寸、类型 — SQL/API 级检索（当前为内存/JSON 模拟）。"""

import json
from pathlib import Path
from typing import List, Optional

from core.schema import FurnitureCategory, FurnitureItem


def _data_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "mock_furniture.json"


def load_catalog() -> List[FurnitureItem]:
    path = _data_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FurnitureItem.model_validate(x) for x in raw]


def filter_hard(
    *,
    category: Optional[FurnitureCategory] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
) -> List[FurnitureItem]:
    items = load_catalog()
    out: List[FurnitureItem] = []
    for it in items:
        if category is not None and it.category != category:
            continue
        if price_min is not None and it.price < price_min:
            continue
        if price_max is not None and it.price > price_max:
            continue
        out.append(it)
    return out
