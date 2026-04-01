"""语义重排 / RAG：Embedding 向量匹配（占位，接入 OpenAI / Chroma 等后实现）。"""

from typing import List

from core.schema import FurnitureItem


def semantic_rerank(
    query_text: str,
    candidates: List[FurnitureItem],
    top_k: int = 10,
) -> List[FurnitureItem]:
    """占位：当前直接返回前 top_k 条。"""
    return candidates[:top_k]
