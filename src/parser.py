# src/parser.py
import os
import ast
import json
import subprocess
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG (keep it simple for hackathon)
# -----------------------------
REPO_URL = os.getenv("REPO_URL", "")
LOCAL_REPO_PATH = os.getenv("LOCAL_REPO_PATH", "")  # if set, parse local folder instead of cloning
OUT_JSON = os.getenv("PARSED_OUT", "parsed_code.json")

# Chunking: best balance for RAG
CHUNK_LINES = int(os.getenv("CHUNK_LINES", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(2 * 1024 * 1024)))  # 2MB safety

EXCLUDE_DIRS = set((os.getenv("EXCLUDE_DIRS", ".git,.venv,venv,node_modules,dist,build,__pycache__")
                    ).split(","))

INCLUDE_EXTS = {".py"}  # for now hackathon is Python; extend later to js/ts/go/java etc.

# -----------------------------
# Helpers
# -----------------------------
def safe_read_text(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def iter_files(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            ext = Path(fn).suffix.lower()
            if ext in INCLUDE_EXTS:
                out.append(p)
    return out


def extract_readme(root: str) -> str:
    for candidate in ["README.md", "readme.md", "Readme.md"]:
        p = os.path.join(root, candidate)
        if os.path.exists(p):
            txt = safe_read_text(p)
            return txt or ""
    # try find any readme deeper (optional)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "readme.md":
                txt = safe_read_text(os.path.join(dirpath, fn))
                if txt:
                    return txt
    return ""


def chunk_by_lines(text: str, chunk_lines: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_line, end_line, chunk_text)
    line numbers are 1-based inclusive.
    """
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    chunks = []
    step = max(1, chunk_lines - overlap)
    start = 0
    while start < n:
        end = min(n, start + chunk_lines)
        chunk_text = "\n".join(lines[start:end])
        chunks.append((start + 1, end, chunk_text))
        if end == n:
            break
        start += step
    return chunks


def extract_preceding_comments(raw_lines: List[str], lineno: int) -> List[str]:
    comments = []
    idx = lineno - 2
    while idx >= 0:
        line = raw_lines[idx].strip()
        if line.startswith("#"):
            comments.insert(0, line[1:].strip())
        elif line == "" or line.startswith(('"""', "'''")):
            idx -= 1
            continue
        else:
            break
        idx -= 1
    return comments


# -----------------------------
# AST Parser
# -----------------------------
class CodeParser(ast.NodeVisitor):
    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.raw_lines = content.splitlines()

        self.items: List[Dict[str, Any]] = []
        self.imports: List[str] = []
        self.global_variables: List[str] = []
        self.syntax_error: Optional[str] = None

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        try:
            if node.targets and isinstance(node.targets[0], ast.Name):
                self.global_variables.append(node.targets[0].id)
        except Exception:
            pass
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_callable(node, kind="Function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._add_callable(node, kind="AsyncFunction")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)

        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append(str(type(b)))

        class_data = {
            "type": "Class",
            "name": f"{self.file_path}::{node.name}",
            "symbol": node.name,
            "start_line": start,
            "end_line": end,
            "bases": bases,
            "docstring": ast.get_docstring(node) or "",
            "code": self._safe_unparse(node),
            "preceding_comments": extract_preceding_comments(self.raw_lines, start),
            "language": "python",
        }

        self.items.append(class_data)
        self.generic_visit(node)

    def _add_callable(self, node, kind: str):
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        fn_data = {
            "type": kind,
            "name": f"{self.file_path}::{node.name}",
            "symbol": node.name,
            "start_line": start,
            "end_line": end,
            "docstring": ast.get_docstring(node) or "",
            "code": self._safe_unparse(node),
            "preceding_comments": extract_preceding_comments(self.raw_lines, start),
            "language": "python",
        }
        self.items.append(fn_data)

    def _safe_unparse(self, node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            # fallback: slice from raw file lines using lineno / end_lineno
            start = max(1, getattr(node, "lineno", 1))
            end = max(start, getattr(node, "end_lineno", start))
            return "\n".join(self.raw_lines[start - 1:end])

    def parse(self) -> Dict[str, Any]:
        try:
            tree = ast.parse(self.content)
            self.visit(tree)
        except SyntaxError as e:
            self.syntax_error = f"{e.__class__.__name__}: {e}"
            # Do not fail: return empty items but keep error recorded
            return {
                "items": [],
                "imports": [],
                "global_variables": [],
                "syntax_error": self.syntax_error,
            }

        return {
            "items": self.items,
            "imports": self.imports,
            "global_variables": self.global_variables,
            "syntax_error": None,
        }


# -----------------------------
# Main Parser Runner
# -----------------------------
def resolve_repo_root() -> str:
    if LOCAL_REPO_PATH and os.path.exists(LOCAL_REPO_PATH):
        return LOCAL_REPO_PATH

    clone_dir = REPO_URL.split("/")[-1].replace(".git", "_codebase")
    if not os.path.exists(clone_dir):
        print(f"📦 Cloning {REPO_URL} -> {clone_dir}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, clone_dir], check=False)
    return clone_dir


def run():
    repo_root = resolve_repo_root()
    repo_root = os.path.abspath(repo_root)

    print(f"🔎 Parsing repo at: {repo_root}")
    readme_content = extract_readme(repo_root)
    py_files = iter_files(repo_root)

    parsed: Dict[str, Any] = {}
    syntax_errors: List[Dict[str, str]] = []

    for fp in py_files:
        rel_path = os.path.relpath(fp, repo_root).replace("\\", "/")
        content = safe_read_text(fp)
        if content is None:
            # Too large or unreadable
            parsed[rel_path] = {
                "ast_items": [],
                "file_chunks": [],
                "imports": [],
                "global_variables": [],
                "syntax_error": "SKIPPED: unreadable or too large",
            }
            continue

        # 1) AST extraction (precise)
        parser = CodeParser(rel_path, content)
        ast_result = parser.parse()

        if ast_result.get("syntax_error"):
            syntax_errors.append({"file": rel_path, "error": ast_result["syntax_error"]})

        # 2) Always add file-level chunks (robust retrieval)
        # This fixes missing definitions like `class APIRouter`.
        chunks = chunk_by_lines(content, CHUNK_LINES, CHUNK_OVERLAP)
        file_chunks = []
        for (s, e, chunk_text) in chunks:
            file_chunks.append({
                "type": "FileChunk",
                "name": f"{rel_path}::chunk_{s}_{e}",
                "symbol": "",  # not a single symbol
                "start_line": s,
                "end_line": e,
                "docstring": "",
                "code": chunk_text,
                "preceding_comments": [],  # not needed here
                "language": "python",
            })

        parsed[rel_path] = {
            "ast_items": ast_result["items"],        # Classes/Functions
            "file_chunks": file_chunks,              # Line chunks for robust retrieval
            "imports": ast_result["imports"],
            "global_variables": ast_result["global_variables"],
            "syntax_error": ast_result["syntax_error"],
        }

    final = {
        "repo_root": repo_root,
        "readme": readme_content,
        "parsed_code": parsed,
        "stats": {
            "num_files": len(py_files),
            "num_syntax_errors": len(syntax_errors),
            "chunk_lines": CHUNK_LINES,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "syntax_errors": syntax_errors[:200],  # cap to keep json small
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(f"✅ Parsing complete: wrote {OUT_JSON}")
    if syntax_errors:
        print(f"⚠️ Syntax errors in {len(syntax_errors)} files (still chunked & included).")
        print("   Example:", syntax_errors[0])


if __name__ == "__main__":
    run()
