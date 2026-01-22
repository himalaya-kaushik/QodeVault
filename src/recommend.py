from typing import List, Dict, Any


def recommend_next_steps(retrieved_chunks: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    """
    Simple, judge-friendly Recommendations:
    - Suggest distinct files not already repeated
    - Prefer higher hybrid score if present
    """
    if not retrieved_chunks:
        return []

    # sort by hybrid score if available
    chunks = sorted(
        retrieved_chunks,
        key=lambda x: x.get("_hybrid_score", 0.0),
        reverse=True
    )

    seen_files = set()
    recs = []

    for c in chunks:
        f = c.get("file") or ""
        name = c.get("name") or ""
        if not f or f in seen_files:
            continue
        seen_files.add(f)
        recs.append(f"- Inspect `{f}` (related: `{name}`)")
        if len(recs) >= limit:
            break

    return recs
