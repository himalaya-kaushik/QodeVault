# QodeVault🔒

### The "Q" nods to Qdrant, and "Vault" implies zero data leakage.

## Stop leaking code. Start debugging privately.

OfflineCursor is a privacy-first AI coding assistant that lives in your terminal. It allows you to chat with your codebase using local LLMs (Ollama) or secure APIs, ensuring your proprietary IP never leaves your machine.

Unlike standard RAG tools that rely on fuzzy semantic matching, OfflineCursor uses Qdrant Hybrid Search to understand both concepts ("function that saves files") and exact symbols (save_to_disk_v2).

# How to use

1. Start Qdrant (Database)
   `docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant  `

2. Install Package
   `pip install -r requirements.txt`

3. Add your REPO link or LOCAL_repo path in env file(important)

```
REPO_URL=https://github.com/himalaya-kaushik/ML-Planner_apple.git
LOCAL_REPO_PATH=Desktop/users/ML-Planner_apple
```

4. Quick Start

```
     #1. Parse your local codebase (builds the AST)
    python3 -m src.parser

    # 2. Ingest into Qdrant (Builds Hybrid Embeddings)
    python3 -m src.ingest

    # 3. Start Chatting
    python3 -m src.chat

```

Sample Output:

The tool provides transparent, evidence-based answers in your terminal:

```
[OfflineCursor] 🟢 Session Started (Memory Active)

📝 You: How does the caching middleware work?

[🔍 Hybrid Search] Retrieving context...
   ├── Found (Dense):  CacheManager class (logic)
   ├── Found (Sparse): middleware_v2.py (exact match)
   └── Found (Memory): You asked about 'redis' 2 mins ago

🤖 Assistant:
The caching middleware is handled in `middleware.py` class `CacheManager`.
It uses the Redis connection we discussed earlier.

Here is the implementation:
def check_cache(request):
    key = generate_key(request)
    return redis.get(key)

```

### Capabilities

OfflineCursor is built on three core pillars using Qdrant:

1. 🔍 Search (Hybrid) Combines Dense Vectors (Semantic meaning) with Sparse Vectors (Exact Keyword matching).

   Benefit: Finds process_tx_logic even if you just ask "where is the transaction logic".

2. 🧠 Memory (Persistent) Maintains a dedicated chat_memory collection in Qdrant.

   Benefit: You can ask "Refactor that function" and it knows which function you mean.

3. 💡 Recommendations (Context-Aware) Uses Qdrant's recommend API to suggest related code.

   Benefit: "Show me other files that use this database pattern."

### Under the Hood

Most RAG tools just chunk text and pray.

OfflineCursor parses code into an AST (Abstract Syntax Tree), indexing Functions and Classes separately.

It leverages Qdrant for:

✔️ Hybrid Retrieval: (SPLADE + MiniLM) for best-in-class accuracy.

✔️ Data Sovereignty: Everything runs on localhost.

✔️ Speed: Sub-millisecond retrieval via Rust-based vector search.

Requirements
Python 3.9+

Docker (for Qdrant)

Ollama (100% offline mode)
LLM api(optional)
