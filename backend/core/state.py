"""双列表动态交互：Show-List（候选池）与 User-List（决策池）。"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .schema import FurnitureItem


class SessionState(BaseModel):
    """单会话状态：每轮可清空 Show-List，User-List 持久传递。"""

    show_list: List[FurnitureItem] = Field(default_factory=list)
    user_list: List[FurnitureItem] = Field(default_factory=list)
    room_image_url: Optional[str] = None
    style_preference: Optional[str] = None
    workflow_phase: Literal["browsing", "finished"] = Field(
        default="browsing",
        description="browsing=可推荐并写入 Show-List；finished=用户已确认决策池清单",
    )

    def clear_show_list(self) -> None:
        self.show_list = []

    @property
    def user_list_total(self) -> float:
        return sum(x.price for x in self.user_list)
