# import os
# import json
# import uuid
# from typing import List
# from qdrant_client.http import models

# from src.database import QdrantHandler
# from src.config import MAX_CODE_CHARS_PER_CHUNK


# def _truncate(s: str, n: int) -> str:
#     s = s or ""
#     return s[:n] if len(s) > n else s


# def run_ingest():
#     db = QdrantHandler()
#     db.setup_collections()

#     json_path = "parsed_code.json"
#     if not os.path.exists(json_path):
#         print(f"❌ {json_path} not found! Run 'python -m src.parser' first.")
#         return

#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     points: List[models.PointStruct] = []

#     # README as Doc
#     readme = (data.get("README") or "").strip()
#     if readme:
#         dense_vec = db.dense_model.embed_query(readme)
#         points.append(
#             models.PointStruct(
#                 id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "README")),
#                 vector={"dense": dense_vec},
#                 payload={
#                     "name": "README",
#                     "file": "README.md",
#                     "code": _truncate(readme, MAX_CODE_CHARS_PER_CHUNK),
#                     "type": "Doc",
#                     "docstring": "Project documentation / overview",
#                     "start_line": 1,
#                     "end_line": None,
#                 },
#             )
#         )

#     parsed = data.get("parsed_code", {})
#     print("🚀 Processing Codebase for Ingestion...")

#     for file_path, content in parsed.items():
#         for obj in content.get("functions_classes", []):
#             code_text = (obj.get("code") or "").strip()
#             if not code_text:
#                 continue

#             unique_str = f"{file_path}::{obj.get('name')}::{obj.get('start_line')}"
#             point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

#             dense_vec = db.dense_model.embed_query(code_text)

#             payload = {
#                 "name": obj.get("name"),
#                 "file": file_path,
#                 "code": _truncate(code_text, MAX_CODE_CHARS_PER_CHUNK),
#                 "type": obj.get("type"),
#                 "docstring": obj.get("docstring") or "",
#                 "start_line": obj.get("start_line"),
#                 "end_line": obj.get("end_line"),
#                 "calls": obj.get("calls", []),
#                 "preceding_comments": obj.get("preceding_comments", []),
#             }

#             points.append(
#                 models.PointStruct(
#                     id=point_id,
#                     vector={"dense": dense_vec},
#                     payload=payload,
#                 )
#             )

#     if not points:
#         print("⚠️ No data found to ingest.")
#         return

#     print(f"📤 Uploading {len(points)} chunks to Qdrant...")
#     BATCH = 256
#     for i in range(0, len(points), BATCH):
#         db.upsert_code_points(points[i:i + BATCH])
#         print(f"✅ Uploaded {min(i + BATCH, len(points))}/{len(points)}")

#     print("🎉 Ingestion Complete!")


# if __name__ == "__main__":
#     run_ingest()


# src/ingest.py
import os
import json
import uuid
from typing import Dict, Any, List, Iterable

from qdrant_client.http import models

from src.database import QdrantHandler

PARSED_PATH = os.getenv("PARSED_OUT", "parsed_code.json")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "256"))


def _iter_items(parsed_code: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yields items from:
      - ast_items (Functions/Classes)
      - file_chunks (FileChunk)
    """
    for file_path, info in parsed_code.items():
        ast_items = info.get("ast_items", []) or []
        file_chunks = info.get("file_chunks", []) or []
        for item in ast_items:
            yield item
        for item in file_chunks:
            yield item


def _stable_point_id(file_path: str, item: Dict[str, Any]) -> str:
    """
    Deterministic ID to allow re-ingest without duplicates.
    """
    start = item.get("start_line", 1)
    end = item.get("end_line", start)
    typ = item.get("type", "Unknown")
    symbol = item.get("symbol", "")
    name = item.get("name", "")
    key = f"{file_path}::{typ}::{symbol}::{name}::{start}-{end}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _batched(it: List[models.PointStruct], batch_size: int) -> Iterable[List[models.PointStruct]]:
    for i in range(0, len(it), batch_size):
        yield it[i : i + batch_size]


def run_ingest():
    # 1) Init Qdrant + collections
    db = QdrantHandler()
    db.setup_collections()

    # 2) Load parsed_code.json
    if not os.path.exists(PARSED_PATH):
        print(f"❌ {PARSED_PATH} not found. Run: python -m src.parser")
        return

    with open(PARSED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_code = data.get("parsed_code", {})
    if not parsed_code:
        print("⚠️ parsed_code is empty. Nothing to ingest.")
        return

    repo_root = data.get("repo_root", "")
    print(f"🚀 Ingesting from: {PARSED_PATH}")
    print(f"📌 Repo root: {repo_root if repo_root else '(unknown)'}")

    points: List[models.PointStruct] = []
    total = 0

    # 3) Build points
    for file_path, info in parsed_code.items():
        for item in (info.get("ast_items", []) or []) + (info.get("file_chunks", []) or []):
            code_text = (item.get("code") or "").strip()
            if not code_text:
                continue

            point_id = _stable_point_id(file_path, item)

            # Dense embedding
            vec = db.dense_model.embed_query(code_text)

            payload = {
                # Identity
                "file": file_path,
                "name": item.get("name", ""),
                "symbol": item.get("symbol", ""),
                "type": item.get("type", "Unknown"),
                "language": item.get("language", "python"),

                # Ranges
                "start_line": int(item.get("start_line", 1) or 1),
                "end_line": int(item.get("end_line", 1) or 1),

                # Content
                "docstring": item.get("docstring", "") or "",
                "preceding_comments": item.get("preceding_comments", []) or [],
                "code": code_text,

                # Repo metadata
                "repo_root": repo_root,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": vec},
                    payload=payload,
                )
            )
            total += 1

    if not points:
        print(" No points generated (empty code).")
        return

    # 4) Upsert in batches
    print(f" Uploading {total} points to Qdrant in batches of {BATCH_SIZE}...")
    uploaded = 0
    for batch in _batched(points, BATCH_SIZE):
        db.upsert_code_points(batch)
        uploaded += len(batch)
        print(f" Uploaded {uploaded}/{total}")

    print(" Ingestion Complete!")


if __name__ == "__main__":
    run_ingest()
