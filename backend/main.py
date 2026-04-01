"""HTTP API 入口：后续可接 LangGraph 状态机与前端。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.schema import FurnitureCategory
from services.furniture_api import filter_hard

app = FastAPI(title="AI-HomeFlow API", version="0.1.0")

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


@app.get("/api/furniture")
def list_furniture(
    category: FurnitureCategory | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
):
    return [
        m.model_dump(mode="json")
        for m in filter_hard(
            category=category,
            price_min=price_min,
            price_max=price_max,
        )
    ]
