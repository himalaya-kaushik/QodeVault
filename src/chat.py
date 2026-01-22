from src.database import QdrantHandler
from src.llm import LLM
from src.recommend import recommend_next_steps
from src.config import (
    TOP_K_DENSE,
    TOP_K_KEYWORD,
    TOP_K_MEMORY,
    MAX_TOTAL_CONTEXT_CHARS,
    MAX_CODE_CHARS_PER_CHUNK,
)

def _build_context(chunks):
    parts = []
    total = 0

    for c in chunks:
        file = c.get("file", "unknown")
        name = c.get("name", "unknown")
        start = c.get("start_line", "?")
        end = c.get("end_line", "?")
        code = (c.get("code") or "")[:MAX_CODE_CHARS_PER_CHUNK]

        block = f"[{file}:{start}-{end}]  {name}\n```python\n{code}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)

def _build_memory(mem_items):
    if not mem_items:
        return ""
    out = []
    for m in mem_items:
        out.append(f"- {m.get('text', '')[:800]}")
    return "\n".join(out)

def chat_loop():
    db = QdrantHandler()
    db.setup_collections()

    llm = LLM()

    print("\n🧠 OfflineCursor (Qdrant Hybrid Search + Memory + Recommendations)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("📝 You: ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        # 1) Memory retrieval
        mem = db.search_memory(q, limit=TOP_K_MEMORY)
        mem_text = _build_memory(mem)

        # 2) Hybrid retrieval (dense + keyword fused)
        limit = max(TOP_K_DENSE, TOP_K_KEYWORD)
        chunks = db.search_hybrid(q, limit=limit)

        context = _build_context(chunks)

        # 3) Recommendations
        recs = recommend_next_steps(chunks, limit=5)

        # 4) Prompt
        prompt = f"""
You are an expert software engineer helping with a codebase.
You MUST:
- Answer using ONLY the retrieved context and memory (no web browsing).
- If context is insufficient, say what is missing and what file you would need.

# Relevant long-term memory (retrieved from Qdrant)
{mem_text}

# Retrieved code/doc context (hybrid search: semantic + keyword)
{context}

# User question
{q}

# Output format
1) Answer (clear, actionable)
2) Evidence (cite file:line ranges you used)
3) Recommendations (next files/functions to inspect)
"""

        ans = llm.generate(prompt)

        print("\n🤖 Assistant:\n")
        print(ans.strip())

        if recs:
            print("\n⭐ Recommendations:")
            print("\n".join(recs))

        # 5) Store memory (persistent in Qdrant)
        used_files = list({c.get("file") for c in chunks if c.get("file")})
        tags = ["hybrid-search", "memory", "recommendations"]
        db.add_memory(q, ans[:1200], files=used_files, tags=tags)

        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    chat_loop()
