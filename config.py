"""Configuration. All API keys are read from environment variables.

Required environment variables before running:
    OPENAI_API_KEY     OpenAI key (used as fallback judge if Gemini is unavailable)
    GEMINI_API_KEY     Google Gemini key (primary judge model)
    SERPAPI_API_KEY    SerpApi key (for Google + Google Scholar search)

Optional:
    GEMINI_MODEL       Gemini model name (default: gemini-3-flash-preview)
"""
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or ""
