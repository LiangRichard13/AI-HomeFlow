"""GTE 中文句向量：供入库脚本与家具搜索 API 共用。"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

RAG_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = RAG_ROOT.parent
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "mock_furniture.json"
DEFAULT_MODEL_DIR = RAG_ROOT / "model" / "nlp_gte_sentence-embedding_chinese-base"
DEFAULT_CHROMA_DIR = RAG_ROOT / "chroma_db"
COLLECTION_NAME = "furniture"
MODELSCOPE_ID = "iic/nlp_gte_sentence-embedding_chinese-base"


def has_local_weight_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    weight_suffixes = {".bin", ".safetensors"}
    for path in model_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in weight_suffixes:
            return True
    return False


def ensure_model_weights(model_dir: Path) -> None:
    if has_local_weight_files(model_dir):
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODELSCOPE_ID, local_dir=str(model_dir))


class ModelScopeGTEEmbeddings(Embeddings):
    def __init__(self, model_path: str, sequence_length: int = 512):
        self._pipe = pipeline(
            Tasks.sentence_embedding,
            model=model_path,
            sequence_length=sequence_length,
            device="cpu",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._pipe(input={"source_sentence": texts})
        mat = out["text_embedding"]
        return [row.astype(float).tolist() for row in mat]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
