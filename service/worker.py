"""Drains the grading queue: claim a job, grade it, store the draft. Safe to call often and from anywhere."""
from __future__ import annotations
import json
import logging
import time
from typing import Callable
from . import config as C
from . import grader_core
from .materials import fetch_bytes
from .providers import ProviderError, complete as default_complete

log = logging.getLogger("grader")


def _text(v) -> str:
    if v in (None, "", [], {}):
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def scheme_from_model_answer(model_answer) -> str:
    """S1 / past-paper bank rows keep their official mark scheme as a picture (mark_scheme is empty) but carry a typed
    worked solution whose parts list the marks (M1 A1 ...), written from and checked against that mark scheme (S3.1).
    Turn it into mark-scheme lines so the grader can mark those questions. Empty when it is not that shape."""
    if not isinstance(model_answer, list):
        return ""
    lines = []
    for st in model_answer:
        if not isinstance(st, dict) or not str(st.get("marks") or "").strip():
            continue
        part = str(st.get("part") or "").strip()
        work = " ".join(str(st.get("working") or "").split())[:700]
        tip = str(st.get("tip") or "").strip()
        lines.append(f"{part or 'whole question'} [{st['marks']}]: final answer {st.get('answer') or '?'}. Method: {work}"
                     + (f" Examiner notes: {tip}" if tip else ""))
    return "\n".join(lines)


def class_question_assignment(cq: dict) -> dict:
    """The grader reads an 'assignment'; for a class question that is the question, its mark scheme and model answer."""
    q, s = cq.get("question") or {}, cq.get("solution") or {}
    return {"title": "Class question", "description": q.get("stem") or "",
            "markingSchemeNotes": _text(s.get("mark_scheme")), "modelAnswerNotes": _text(s.get("model_answer"))}


def quiz_assignment(paper: dict) -> dict | None:
    """The grader reads an 'assignment'; for a bank quiz that is this student's own questions, numbered 1, 2, ...,
    with each one's mark scheme and model answer from the bank. None when no question has a typed mark scheme."""
    desc, ms, ma, have = [], [], [], 0
    for i, q in enumerate(paper.get("questions") or [], 1):
        b = (paper.get("bank") or {}).get(q.get("qbId")) or {}
        marks = q.get("marks") or b.get("marks")
        image_only = bool((b.get("data") or {}).get("image_only"))     # past-paper row: the stem text is unreliable OCR
        desc.append(f"Question {i}" + (f" [{marks} marks]" if marks else "") + ": "
                    + ("(shown as a picture)" if image_only else (q.get("stem") or b.get("stem") or "(shown as a picture)")))
        scheme = _text(b.get("mark_scheme")) or scheme_from_model_answer(b.get("model_answer"))
        if scheme:
            have += 1
            ms.append(f"Question {i}: {scheme}")
        else:
            ms.append(f"Question {i}: (no typed mark scheme: do not mark this question, leave it for the teacher)")
        if _text(b.get("model_answer")):
            ma.append(f"Question {i}: {_text(b.get('model_answer'))}")
    if not have:
        return None
    sep = chr(10)
    return {"title": paper.get("title") or "Quiz", "description": sep.join(desc),
            "markingSchemeNotes": sep.join(ms), "modelAnswerNotes": sep.join(ma)}


def assignment_for(store, kind: str, sub: dict) -> tuple[dict | None, str]:
    """What the grader marks against: (assignment-shaped dict, "") or (None, why it cannot be graded)."""
    if kind == "class_question":
        return class_question_assignment(sub["_cq"]), ""
    if kind == "submission":
        return (store.get_assignment(sub.get("assignment_id")) or {}).get("data", {}), ""
    paper = sub.get("_quiz") or {}
    if paper.get("versioned") and not paper.get("has_version"):
        return None, "no_paper: versioned quiz but this student has no version row"
    a = quiz_assignment(paper) if paper.get("type") == "bank" else None
    return (a, "") if a else (None, "no_scheme: not a bank quiz, or no question has a typed mark scheme")


def guidance_for(store, job: dict, sub: dict) -> dict | None:
    """3.6: the teacher's marking notes and corrections for this work. Never stops a grade: on any error, none."""
    get = getattr(store, "guidance", None)
    if not get:
        return None
    if job["kind"] == "submission":
        work_id = sub.get("assignment_id")
    elif job["kind"] == "quiz" and not (sub.get("_quiz") or {}).get("versioned"):
        work_id = sub.get("quiz_id")
    else:
        return None                     # class questions have no notes; a versioned quiz's numbers differ per student
    try:
        return get(job["kind"], work_id, exclude_submission=job["submission_id"])
    except Exception as e:
        log.warning("guidance not read: %s", type(e).__name__)
        return None


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
        assignment, why = assignment_for(store, job["kind"], sub)
        if not assignment:
            store.finish(job["id"], "needs_teacher", why)
            return True
        out = grader_core.grade_submission(assignment, sub.get("data") or {}, fetch, complete,
                                           guidance=guidance_for(store, job, sub))
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


def drain(store, limit: int = 5, budget: float = C.RUN_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic,
          **kw) -> int:
    """Grades up to `limit` jobs, taking no new job once `budget` seconds have passed (a long practice book takes
    minutes; the Action is stopped at 12 and a job cut off mid-way would sit 'running' until it is requeued)."""
    store.requeue_stale()
    start, n = clock(), 0
    while n < limit and clock() - start < budget and process_one(store, **kw):
        n += 1
    return n
