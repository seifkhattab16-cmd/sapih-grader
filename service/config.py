"""Settings, all from environment variables (the host's secrets). Nothing here is ever sent to the browser."""
import os


def env(name: str, default: str = "") -> str:
    for key in (name, name.replace("_", "-"), name.lower()):
        v = (os.getenv(key) or "").strip()
        if v:
            return v
    return default


def lst(v: str) -> list[str]:
    return [s.strip() for s in v.split(",") if s.strip()]


def supabase_url() -> str:
    return env("SUPABASE_URL").rstrip("/")


def supabase_key() -> str:
    return env("SUPABASE_SERVICE_ROLE_KEY")   # secret key: lives on this host only


def webhook_secret() -> str:
    return env("GRADER_WEBHOOK_SECRET")       # protects /tick; the keep-warm ping sends it


def route_order() -> list[str]:
    return lst(env("AI_ROUTE_ORDER", "groq,openrouter,anthropic"))


MAX_ATTEMPTS = 3
MAX_PAGES_STUDENT = 6        # page images sent for the student's work
MAX_PAGES_REFERENCE = 3      # page images sent per reference document (QP / MS / MA) when it is a scan
MIN_TEXT_PER_PAGE = 40       # fewer characters than this per page = a scan, send images instead
