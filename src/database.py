import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    QDRANT_URL,
    COLLECTION_CODEBASE,
    COLLECTION_MEMORY,
    DENSE_MODEL_NAME,
    DENSE_VECTOR_SIZE,
    RRF_K,
)


def _tokenize_query(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{1,}", q)
    seen = set()
    out = []
    for t in toks:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
    return out[:8]


class QdrantHandler:
    def __init__(self):
        print(f"🔌 Connecting to Qdrant at {QDRANT_URL}...")
        self.client = QdrantClient(url=QDRANT_URL)

        print(f"🧠 Loading Dense Model ({DENSE_MODEL_NAME})...")
        self.dense_model = HuggingFaceEmbeddings(model_name=DENSE_MODEL_NAME)

    def setup_collections(self):
        if not self.client.collection_exists(COLLECTION_CODEBASE):
            print(f"🔨 Creating Codebase Collection: {COLLECTION_CODEBASE}")
            self.client.create_collection(
                collection_name=COLLECTION_CODEBASE,
                vectors_config={
                    "dense": models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE)
                },
            )
        else:
            print(f"✅ Collection '{COLLECTION_CODEBASE}' exists.")

        if not self.client.collection_exists(COLLECTION_MEMORY):
            print(f"🧠 Creating Memory Collection: {COLLECTION_MEMORY}")
            self.client.create_collection(
                collection_name=COLLECTION_MEMORY,
                vectors_config={
                    "dense": models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE)
                },
            )
        else:
            print(f"✅ Collection '{COLLECTION_MEMORY}' exists.")

    # -------------------------
    # UPSERT
    # -------------------------
    def upsert_code_points(self, points: List[models.PointStruct]):
        self.client.upsert(collection_name=COLLECTION_CODEBASE, points=points)

    def upsert_memory_points(self, points: List[models.PointStruct]):
        self.client.upsert(collection_name=COLLECTION_MEMORY, points=points)

    # -------------------------
    # Qdrant search wrapper (works across client versions)
    # -------------------------
    def _qdrant_dense_search(self, collection_name: str, vector: List[float], vector_name: str, limit: int):
        """
        Robust across qdrant-client versions:
        - Preferred: query_points(query=<raw vector>, using=<vector_name>)
        - Fallback: search_points(query_vector=NamedVector(...))
        """
        # Newer clients: query_points(query=..., using="dense")
        if hasattr(self.client, "query_points"):
            try:
                res = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    using=vector_name,
                    limit=limit,
                    with_payload=True,
                )
                return res.points  # list[ScoredPoint]
            except Exception:
                # If this client has query_points but doesn't like params, fall through
                pass

        # Alternative clients: search_points
        if hasattr(self.client, "search_points"):
            return self.client.search_points(
                collection_name=collection_name,
                query_vector=models.NamedVector(name=vector_name, vector=vector),
                limit=limit,
                with_payload=True,
            )

        raise RuntimeError("qdrant-client too old: missing query_points/search_points. Upgrade qdrant-client.")

    # -------------------------
    # SEARCH: Dense (semantic)
    # -------------------------
    def search_dense(self, query: str, limit: int = 6) -> List[models.ScoredPoint]:
        dense_vec = self.dense_model.embed_query(query)
        return self._qdrant_dense_search(
            collection_name=COLLECTION_CODEBASE,
            vector=dense_vec,
            vector_name="dense",
            limit=limit,
        )

    # -------------------------
    # SEARCH: Keyword (lexical)
    # Uses scroll() with MatchText filters
    # -------------------------
    def search_keyword(self, query: str, limit: int = 6) -> List[models.ScoredPoint]:
        tokens = _tokenize_query(query)
        if not tokens:
            return []

        should: List[models.FieldCondition] = []
        for t in tokens:
            should.append(models.FieldCondition(key="code", match=models.MatchText(text=t)))
            should.append(models.FieldCondition(key="name", match=models.MatchText(text=t)))
            should.append(models.FieldCondition(key="file", match=models.MatchText(text=t)))
            should.append(models.FieldCondition(key="docstring", match=models.MatchText(text=t)))

        flt = models.Filter(should=should)

        points, _ = self.client.scroll(
            collection_name=COLLECTION_CODEBASE,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Create ranked list for RRF fusion
        out: List[models.ScoredPoint] = []
        for i, p in enumerate(points):
            out.append(
                models.ScoredPoint(
                    id=p.id,
                    version=0,
                    score=1.0 / (1 + i),  # dummy score
                    payload=p.payload,
                    vector=None,
                    shard_key=None,
                    order_value=None,
                )
            )
        return out

    # -------------------------
    # FUSION: Reciprocal Rank Fusion (RRF)
    # -------------------------
    def _rrf_fuse(
        self,
        dense: List[models.ScoredPoint],
        keyword: List[models.ScoredPoint],
        k: int = RRF_K,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict[str, Any]] = {}

        def add(results: List[models.ScoredPoint]):
            for rank, sp in enumerate(results, start=1):
                pid = str(sp.id)
                scores[pid] = scores.get(pid, 0.0) + (1.0 / (k + rank))
                if sp.payload is not None:
                    payloads[pid] = sp.payload

        add(dense)
        add(keyword)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(pid, sc, payloads.get(pid, {})) for pid, sc in fused]

    def search_hybrid(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        dense = self.search_dense(query, limit=limit)
        keyword = self.search_keyword(query, limit=limit)

        fused = self._rrf_fuse(dense, keyword)
        out: List[Dict[str, Any]] = []
        for pid, score, payload in fused[:limit]:
            payload = dict(payload or {})
            payload["_hybrid_score"] = score
            payload["_id"] = pid
            out.append(payload)
        return out

    # -------------------------
    # MEMORY
    # -------------------------
    def add_memory(
        self,
        user_text: str,
        assistant_text: str,
        files: List[str] | None = None,
        tags: List[str] | None = None,
    ):
        files = files or []
        tags = tags or []

        text = f"User: {user_text}\nAssistant: {assistant_text}"
        vector = self.dense_model.embed_query(text)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": vector},
            payload={
                "user": user_text,
                "assistant": assistant_text,
                "text": text,
                "files": files,
                "tags": tags,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.upsert_memory_points([point])

    def search_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        vector = self.dense_model.embed_query(query)
        results = self._qdrant_dense_search(
            collection_name=COLLECTION_MEMORY,
            vector=vector,
            vector_name="dense",
            limit=limit,
        )
        return [r.payload for r in results if r.payload is not None]
