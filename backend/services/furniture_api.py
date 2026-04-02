"""硬性规则过滤（类目/价格/尺寸）+ 可选 Chroma 向量排序。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from core.schema import Dimensions, FurnitureCategory, FurnitureItem

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from rag.gte_embeddings import ModelScopeGTEEmbeddings

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 20

_embeddings: "ModelScopeGTEEmbeddings | None" = None
_vectorstore: "Chroma | None" = None
_vectorstore_failed: bool = False


def _data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "mock_furniture.json"


def load_catalog() -> List[FurnitureItem]:
    path = _data_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FurnitureItem.model_validate(x) for x in raw]


def _get_embeddings() -> "ModelScopeGTEEmbeddings":
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    from rag.gte_embeddings import DEFAULT_MODEL_DIR, ModelScopeGTEEmbeddings, ensure_model_weights

    ensure_model_weights(DEFAULT_MODEL_DIR)
    _embeddings = ModelScopeGTEEmbeddings(str(DEFAULT_MODEL_DIR))
    return _embeddings


def _get_vectorstore() -> "Chroma | None":
    global _vectorstore, _vectorstore_failed
    if _vectorstore is not None:
        return _vectorstore
    if _vectorstore_failed:
        return None
    try:
        from langchain_chroma import Chroma
        from rag.gte_embeddings import COLLECTION_NAME, DEFAULT_CHROMA_DIR

        if not DEFAULT_CHROMA_DIR.is_dir():
            _vectorstore_failed = True
            return None
        emb = _get_embeddings()
        _vectorstore = Chroma(
            persist_directory=str(DEFAULT_CHROMA_DIR),
            embedding_function=emb,
            collection_name=COLLECTION_NAME,
        )
        return _vectorstore
    except Exception:
        logger.exception("初始化向量库失败，带 description 的搜索将降级为按 id 排序。")
        _vectorstore_failed = True
        return None


def warmup_rag() -> None:
    """进程启动时预加载 GTE 并连接 Chroma，避免首个带描述的搜索长时间阻塞。"""
    global _vectorstore_failed

    try:
        from rag.gte_embeddings import DEFAULT_CHROMA_DIR
        _get_embeddings().embed_query("warmup")
        logger.info("GTE 嵌入模型已加载并完成一次预热推理。")
    except Exception:
        logger.exception("GTE 预热失败，带 description 的搜索将降级为按 id 排序。")
        _vectorstore_failed = True
        return

    vs = _get_vectorstore()
    if vs is not None:
        logger.info("Chroma 向量库已就绪。")
    elif not DEFAULT_CHROMA_DIR.is_dir():
        logger.warning("未找到 Chroma 目录（%s），向量检索不可用。", DEFAULT_CHROMA_DIR)
    else:
        logger.warning("Chroma 目录存在但未能初始化向量库，带 description 的搜索将降级。")


def _passes_dimensions(
    dim: Dimensions,
    *,
    w_min: Optional[float],
    w_max: Optional[float],
    d_min: Optional[float],
    d_max: Optional[float],
    h_min: Optional[float],
    h_max: Optional[float],
) -> bool:
    if w_min is not None and dim.w < w_min:
        return False
    if w_max is not None and dim.w > w_max:
        return False
    if d_min is not None and dim.d < d_min:
        return False
    if d_max is not None and dim.d > d_max:
        return False
    if h_min is not None and dim.h < h_min:
        return False
    if h_max is not None and dim.h > h_max:
        return False
    return True


def filter_hard(
    *,
    category: Optional[FurnitureCategory] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    w_min: Optional[float] = None,
    w_max: Optional[float] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
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
        if not _passes_dimensions(
            it.dimensions,
            w_min=w_min,
            w_max=w_max,
            d_min=d_min,
            d_max=d_max,
            h_min=h_min,
            h_max=h_max,
        ):
            continue
        out.append(it)
    return out


def search_furniture(
    *,
    category: FurnitureCategory,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    w_min: Optional[float] = None,
    w_max: Optional[float] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    description: Optional[str] = None,
) -> List[FurnitureItem]:
    """先按类目/价格/尺寸硬过滤，再在有描述时对候选集做向量相似度排序，最多 20 条。"""
    candidates = filter_hard(
        category=category,
        price_min=price_min,
        price_max=price_max,
        w_min=w_min,
        w_max=w_max,
        d_min=d_min,
        d_max=d_max,
        h_min=h_min,
        h_max=h_max,
    )
    if not candidates:
        return []
    desc = (description or "").strip()
    if not desc:
        return sorted(candidates, key=lambda x: x.id)[:MAX_SEARCH_RESULTS]

    vs = _get_vectorstore()
    if vs is None:
        return sorted(candidates, key=lambda x: x.id)[:MAX_SEARCH_RESULTS]

    ids = [it.id for it in candidates]
    k = min(MAX_SEARCH_RESULTS, len(ids))
    chroma_filter = {"furniture_id": {"$in": ids}}
    try:
        ranked = vs.similarity_search_with_score(desc, k=k, filter=chroma_filter)
    except Exception:
        return sorted(candidates, key=lambda x: x.id)[:MAX_SEARCH_RESULTS]

    by_id = {it.id: it for it in candidates}
    out: List[FurnitureItem] = []
    seen: set[str] = set()
    for doc, _score in ranked:
        fid = doc.metadata.get("furniture_id")
        if isinstance(fid, str) and fid in by_id and fid not in seen:
            out.append(by_id[fid])
            seen.add(fid)
        if len(out) >= MAX_SEARCH_RESULTS:
            break
    return out[:MAX_SEARCH_RESULTS]
