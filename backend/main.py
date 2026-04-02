"""HTTP API 入口：后续可接 LangGraph 状态机与前端。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.schema import FurnitureSearchRequest
from services.furniture_api import search_furniture, warmup_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_rag()
    yield


app = FastAPI(
    title="AI-HomeFlow API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/furniture/search")
def furniture_search(body: FurnitureSearchRequest):
    """类目必填；价格、三维尺寸可选硬过滤；description 可选，非空时在候选内按向量相似度排序，最多 20 条。"""
    items = search_furniture(
        category=body.category,
        price_min=body.price_min,
        price_max=body.price_max,
        w_min=body.w_min,
        w_max=body.w_max,
        d_min=body.d_min,
        d_max=body.d_max,
        h_min=body.h_min,
        h_max=body.h_max,
        description=body.description,
    )
    return [m.model_dump(mode="json") for m in items]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
