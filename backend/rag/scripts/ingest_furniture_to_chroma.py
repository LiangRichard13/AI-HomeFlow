"""
将 mock_furniture.json 转为结构化文本，使用本地 GTE 中文向量模型做 embedding，
并通过 LangChain + Chroma 持久化；每条文档的 Chroma id 与 JSON 中的 id 一致。
默认强制 CPU（device='cpu'）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 避免误用 GPU（ModelScope 部分 pipeline 默认 gpu）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.gte_embeddings import (
    COLLECTION_NAME,
    DEFAULT_CHROMA_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    ModelScopeGTEEmbeddings,
    ensure_model_weights,
    has_local_weight_files,
)


# 本文件在 rag/scripts/ 下
_SCRIPTS_DIR = Path(__file__).resolve().parent


def format_structured_document(item: dict) -> str:
    """与业务约定一致的四段式描述。"""
    styles = "、".join(item.get("style_tags") or [])
    colors = "、".join(item.get("colors") or [])
    material = item.get("material") or ""
    desc = item.get("description") or ""
    name = item.get("name") or ""
    return (
        f"【名称】{name}\n"
        f"【风格与场景】{styles}\n"
        f"【材质与颜色】材质：{material}；可选颜色：{colors}\n"
        f"【详情】{desc}"
    )


def load_furniture_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("mock_furniture.json 应为对象数组")
    return data


def ingest(
    data_path: Path,
    model_dir: Path,
    chroma_dir: Path,
    reset: bool,
) -> None:
    if not has_local_weight_files(model_dir):
        print(f"未检测到权重文件，正在从 ModelScope 下载到: {model_dir}")
    ensure_model_weights(model_dir)

    items = load_furniture_items(data_path)
    documents: list[Document] = []
    ids: list[str] = []

    for item in items:
        fid = item.get("id")
        if not fid:
            raise ValueError("家具条目缺少 id 字段")
        text = format_structured_document(item)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "furniture_id": fid,
                    "name": item.get("name") or "",
                    "category": item.get("category") or "",
                },
            )
        )
        ids.append(str(fid))

    embeddings = ModelScopeGTEEmbeddings(str(model_dir), sequence_length=512)

    if reset and chroma_dir.exists():
        import shutil

        shutil.rmtree(chroma_dir)

    chroma_dir.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_dir),
    )
    print(f"已写入 {len(documents)} 条向量，collection={COLLECTION_NAME}，目录: {chroma_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="家具 Mock 数据写入 Chroma")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="mock_furniture.json 路径",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="nlp_gte_sentence-embedding_chinese-base 本地目录",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空已有 chroma_dir 后重建",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"数据文件不存在: {args.data}", file=sys.stderr)
        return 1

    ingest(args.data, args.model_dir, args.chroma_dir, args.reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
