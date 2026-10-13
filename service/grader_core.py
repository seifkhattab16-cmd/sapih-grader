"""Grades one submission against the assignment's mark scheme / model answer.

Rules (project CLAUDE.md + roadmap 3.2/3.5):
- The AI only grades work it can read: typed answer, photo or PDF. A title or file name is never graded.
- No readable mark scheme or model answer -> no AI call; the job goes to the teacher.
- The result is a DRAFT. It is stored in ai_grades and never written to submissions.grade.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator
from . import config as C
from .materials import Material, read_document
from .providers import complete as default_complete

LOW_CONFIDENCE = 0.6

SYSTEM = (
    "You are a strict but fair maths examiner (Edexcel style: M = method, A = accuracy, B = independent mark). "
    "You grade only from the student's actual work shown to you. If part of it is illegible or missing, "
    "say so and award no marks you cannot justify. Follow-through marks only where the mark scheme allows. "
    "Reply with ONLY one JSON object, no prose, no markdown fences."
)

SCHEMA = (
    '{"questions":[{"ref":"1a","max_marks":<int>,"awarded_marks":<int 0..max_marks>,'
    '"transcription":"<the student\'s working, as you read it>",'
    '"marks":[{"type":"M1"|"A1"|"B1"|...,"awarded":true|false,"criterion":"<mark scheme line>","evidence":"<what the student wrote>"}],'
    '"feedback":"<one or two sentences for the student>","confidence":<0..1>,"reason":"<why these marks>",'
    '"legible":true|false}],"overall_confidence":<0..1>}'
)


class MarkLine(BaseModel):
    type: str = "M1"
    awarded: bool = False
    criterion: str = ""
    evidence: str = ""


class QuestionGrade(BaseModel):
    ref: str = "?"
    max_marks: int = Field(ge=0, le=100)
    awarded_marks: int = 0
    transcription: str = ""
    marks: list[MarkLine] = []
    feedback: str = ""
    confidence: float = 0.5
    reason: str = ""
    legible: bool = True

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class GradeResult(BaseModel):
    questions: list[QuestionGrade]
    overall_confidence: float = 0.5


class Outcome(BaseModel):
    status: str                      # "graded" | "needs_teacher"
    note: str = ""
    total_awarded: Optional[float] = None
    total_max: Optional[float] = None
    confidence: Optional[float] = None
    questions: list[dict] = []
    flags: list[str] = []
    via: str = ""


def parse_reply(text: str) -> GradeResult:
    """Pulls the JSON object out of a model reply and validates it. Raises ValueError if unusable."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    try:
        res = GradeResult.model_validate(json.loads(t[start:end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"bad grade JSON: {str(e)[:150]}") from e
    if not res.questions:
        raise ValueError("no questions in grade")
    for q in res.questions:
        q.awarded_marks = max(0, min(int(q.awarded_marks), q.max_marks))   # clamp, never trust the model
    return res


def _read_url(url: str, fetch: Callable[[str], bytes], **kw) -> Material:
    if not url or not isinstance(url, str) or not url.startswith("https://"):
        return Material()
    try:
        return read_document(fetch(url), url.split("?")[0].rsplit("/", 1)[-1], **kw)
    except Exception:
        return Material()


def _note(v) -> str:
    return v.strip() if isinstance(v, str) and not v.strip().startswith("http") else ""


def grade_submission(assignment: dict, submission: dict, fetch: Callable[[str], bytes],
                     complete: Callable = default_complete) -> Outcome:
    """assignment / submission are the `data` JSON of their rows."""
    # 1. The student's work: typed text + uploaded file. Nothing else counts.
    typed = (submission.get("text") or "").strip()
    work = _read_url(submission.get("fileUrl") or "", fetch, max_images=C.MAX_PAGES_STUDENT, always_images=True)
    if not typed and not work.readable:
        return Outcome(status="needs_teacher", note="unreadable_work: no typed answer, and the file could not be read "
                       "(unsupported type, corrupt, or empty). Not graded.", flags=["unreadable_work"])

    # 2. The reference: marking scheme and/or model answer, as a file or typed notes.
    ms = _read_url(assignment.get("markingSchemeUrl") or "", fetch)
    ma = _read_url(assignment.get("modelAnswerUrl") or "", fetch)
    ms_notes = _note(assignment.get("markingSchemeNotes")) or _note(assignment.get("markingScheme"))
    ma_notes = _note(assignment.get("modelAnswerNotes")) or _note(assignment.get("modelAnswer"))
    if not (ms.readable or ma.readable or ms_notes or ma_notes):
        return Outcome(status="needs_teacher", note="no_scheme: this assignment has no readable marking scheme or "
                       "model answer. Not graded.", flags=["no_scheme"])
    qp = _read_url(assignment.get("fileUrl") or "", fetch)

    def block(title: str, m: Material, notes: str) -> str:
        body = "\n".join(x for x in (notes, m.text) if x) or ("(see attached page images)" if m.images else "")
        return f"=== {title} ===\n{body}\n" if body else ""

    prompt = (
        f"Assignment: {assignment.get('title', '')}\n\n"
        + block("QUESTION PAPER", qp, "")
        + block("MARK SCHEME", ms, ms_notes)
        + block("MODEL ANSWER", ma, ma_notes)
        + "\n=== STUDENT WORK ===\n"
        + (f"Typed answer:\n{typed}\n" if typed else "")
        + ("Attached images are the student's handwritten/scanned pages, first.\n" if work.images else "")
        + (f"Text read from the student's file:\n{work.text}\n" if work.text else "")
        + "\nGrade every question the work attempts, and list unanswered questions with 0 marks. "
          "Use the mark scheme's own marks per question. Respond with JSON of this shape:\n" + SCHEMA
    )
    images = work.images + qp.images + ms.images + ma.images
    # Work that is only pictures (no typed answer, no text layer) must go to a model that can see images.
    handwritten = bool(work.images) and not typed and not work.text.strip()

    # 3. One AI call; ask once more with a stricter reminder if the reply is not valid JSON.
    last_err = ""
    for attempt in range(2):
        text, via = complete(SYSTEM, prompt + (f"\n\nYour last reply was invalid ({last_err}). JSON only." if attempt else ""),
                             images, want_out=3000, require_vision=handwritten)
        try:
            res = parse_reply(text)
            break
        except ValueError as e:
            last_err = str(e)
    else:
        return Outcome(status="needs_teacher", note=f"bad_ai_reply: {last_err}", flags=["bad_ai_reply"])

    if all(not q.legible for q in res.questions):   # the model could not read any of it: a 0 here would be a lie
        return Outcome(status="needs_teacher", note="unreadable_work: the AI could not read the work. Not graded.",
                       flags=["unreadable_work"], via=via)
    flags = []
    if any(not q.legible for q in res.questions):
        flags.append("unreadable_part")
    if res.overall_confidence < LOW_CONFIDENCE or any(q.confidence < LOW_CONFIDENCE for q in res.questions):
        flags.append("low_confidence")
    if any(not q.marks for q in res.questions):
        flags.append("no_mark_breakdown")
    total_max = sum(q.max_marks for q in res.questions)
    return Outcome(status="graded", total_awarded=sum(q.awarded_marks for q in res.questions), total_max=total_max,
                   confidence=max(0.0, min(1.0, res.overall_confidence)),
                   questions=[q.model_dump() for q in res.questions], flags=flags, via=via)
