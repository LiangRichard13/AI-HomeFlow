"""双列表动态交互：Show-List（候选池）与 User-List（决策池）。"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .schema import FurnitureItem


class SessionState(BaseModel):
    """单会话状态：每轮可清空 Show-List，User-List 持久传递。"""

    show_list: List[FurnitureItem] = Field(default_factory=list)
    user_list: List[FurnitureItem] = Field(default_factory=list)
    remaining_budget: Optional[float] = None
    room_image_url: Optional[str] = None
    style_preference: Optional[str] = None

    def clear_show_list(self) -> None:
        self.show_list = []
