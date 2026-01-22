import os
from src.config import LLM_TYPE, GEMINI_API_KEY, OLLAMA_URL, OLLAMA_MODEL

class LLM:
    def __init__(self):
        self.llm_type = (LLM_TYPE or "gemini").lower()

        if self.llm_type == "gemini":
            import google.generativeai as genai

            api_key = GEMINI_API_KEY
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY in .env")

            genai.configure(api_key=api_key)

            # Use same style as your working script
            model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
            self.model = genai.GenerativeModel(model_name)

        elif self.llm_type == "ollama":
            self.model = None
        else:
            raise RuntimeError("LLM_TYPE must be 'gemini' or 'ollama'")

    def generate(self, prompt: str) -> str:
        if self.llm_type == "gemini":
            try:
                resp = self.model.generate_content(prompt)
                return (resp.text or "").strip()
            except Exception as e:
                return f"❌ Gemini error: {e}"

        # Ollama fallback
        import requests
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception as e:
            return f"❌ Ollama error: {e}"
