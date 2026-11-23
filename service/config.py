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
MAX_PAGES_STUDENT = 24       # pages of the student's work read, one page per transcription call (a vision route may
                             # take only one image: Groq's does), written pages first when the blank paper is known
MAX_PAGES_REFERENCE = 3      # page images sent per reference document (QP / MS / MA) when it is a short scan
MAX_PAGES_REFERENCE_COPY = 30  # a longer scanned mark scheme / model answer is copied to text, one page per call
WRITTEN_PAGE_MARGIN = 0.3    # a page is "written on" when it differs from the blank paper this much more than a
                             # typical untouched page does (mean grey-level difference, 0-255, at 160 px wide)
RUN_BUDGET_SECONDS = 480     # the Action stops after 12 minutes: take no new job after 8
MIN_TEXT_PER_PAGE = 40       # fewer characters than this per page = a scan, send images instead
MAX_TEACHER_EXAMPLES = 6     # 3.6: the teacher's corrected questions on the same work, shown to the marker as examples
