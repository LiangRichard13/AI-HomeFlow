"""
查看 Chroma 中已入库的文本块（collection: furniture）。
Windows 控制台建议：python ... > ..\chroma_dump.txt 或本脚本 --out file.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb


def _utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


_SCRIPTS_DIR = Path(__file__).resolve().parent
RAG_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CHROMA_DIR = RAG_ROOT / "chroma_db"
COLLECTION_NAME = "furniture"


def main() -> int:
    parser = argparse.ArgumentParser(description="导出/打印 Chroma 中的家具文档块")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="集合名称",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出（UTF-8，含 id / metadata / document）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="写入文件（UTF-8）；不传则打印到标准输出",
    )
    args = parser.parse_args()

    if not args.chroma_dir.is_dir():
        print(f"目录不存在: {args.chroma_dir}", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        coll = client.get_collection(args.collection)
    except Exception as e:
        print(f"无法打开集合 {args.collection!r}: {e}", file=sys.stderr)
        return 1

    row = coll.get(include=["documents", "metadatas"])
    ids = row["ids"]
    docs = row["documents"]
    metas = row["metadatas"]

    if args.json:
        payload = [
            {"id": i, "metadata": m or {}, "document": d}
            for i, d, m in zip(ids, docs, metas)
        ]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        parts = []
        for idx, (doc_id, doc, meta) in enumerate(
            zip(ids, docs, metas), start=1
        ):
            parts.append("=" * 60)
            parts.append(f"[{idx}] id={doc_id}")
            parts.append(f"metadata: {meta}")
            parts.append("--- document ---")
            parts.append(doc)
        text = "\n".join(parts)

    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}（共 {len(ids)} 条）", file=sys.stderr)
    else:
        _utf8_stdio()
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
