"""Drains the grading queue: claim a job, grade it, store the draft. Safe to call often and from anywhere."""
from __future__ import annotations
import logging
from typing import Callable
from . import grader_core
from .materials import fetch_bytes
from .providers import ProviderError, complete as default_complete

log = logging.getLogger("grader")


def process_one(store, fetch: Callable[[str], bytes] = fetch_bytes, complete: Callable = default_complete) -> bool:
    """Returns True if a job was handled (so the caller should try for another)."""
    job = store.claim_next()
    if not job:
        return False
    try:
        sub = store.get_submission(job["kind"], job["submission_id"])
        if not sub:
            store.finish(job["id"], "failed", "submission no longer exists")
            return True
        assignment = (store.get_assignment(sub.get("assignment_id")) or {}).get("data", {}) if job["kind"] == "submission" else {}
        if job["kind"] != "submission":
            store.finish(job["id"], "needs_teacher", "quiz grading arrives with step 2.5")
            return True
        out = grader_core.grade_submission(assignment, sub.get("data") or {}, fetch, complete)
        if out.status == "graded":
            store.save_grade(job, sub, out)
            store.finish(job["id"], "done", (",".join(out.flags) + " | " + out.note)[:500])
        else:
            store.finish(job["id"], "needs_teacher", out.note)
    except ProviderError as e:      # providers busy or rate-limited: put it back, try later (attempt counted)
        store.finish(job["id"], "queued" if job["attempts"] < 3 else "failed", f"providers: {e}")
    except Exception as e:
        log.error("grading job %s failed: %s", job["id"], type(e).__name__)  # no details: public Action logs
        store.finish(job["id"], "queued" if job["attempts"] < 3 else "failed", f"{type(e).__name__}: {str(e)[:200]}")
    return True


def drain(store, limit: int = 5, **kw) -> int:
    store.requeue_stale()
    n = 0
    while n < limit and process_one(store, **kw):
        n += 1
    return n
