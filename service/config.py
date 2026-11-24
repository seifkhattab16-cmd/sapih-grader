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
WRITTEN_PAGE_INK = 3         # ...or has at least this many strongly changed pixels (ink) at that size
RUN_BUDGET_SECONDS = 900     # the Action stops after 25 minutes: take no new job after 15, and pause a long paper
                             # (pages read so far are kept in grader_cache) once the run is 18 minutes in; one AI call
JOB_BUDGET_SECONDS = 1080    # takes at most CALL_BUDGET_SECONDS, so a paused run still ends well before the limit
CALL_BUDGET_SECONDS = 150    # one AI call, every route and every 429 retry included (run #214 was cut off mid-call)
REQUEST_TIMEOUT = 60         # one HTTP request to a provider
MARK_PROMPT_CHARS = 9000     # a marking prompt longer than this (a long practice book) is split...
MARK_BATCH_QUESTIONS = 1     # ...into batches of this many questions (Groq's free tier refused a 20-question one: 413,
                             # and cut a 3-question reply at its 1500-token output cap: one question per call fits)
RATE_LIMIT_RETRIES = 2      # a call refused with 429 by every route is tried again this many times...
RATE_LIMIT_WAIT = 30         # ...after waiting the provider's Retry-After, at most this many seconds
MIN_TEXT_PER_PAGE = 40       # fewer characters than this per page = a scan, send images instead
MAX_TEACHER_EXAMPLES = 6     # 3.6: the teacher's corrected questions on the same work, shown to the marker as examples
