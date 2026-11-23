"""Grades one submission against the assignment's mark scheme / model answer.

Rules (project CLAUDE.md + roadmap 3.2/3.5):
- The AI only grades work it can read: typed answer, photo or PDF. A title or file name is never graded.
- No readable mark scheme or model answer -> no AI call; the job goes to the teacher.
- The result is a DRAFT. It is stored in ai_grades and never written to submissions.grade.
"""
from __future__ import annotations
import json
from difflib import SequenceMatcher
import re
from typing import Callable, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator
from . import config as C
from .materials import Material, read_document, is_file_ref
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


WORK_TYPES = ("working", "proof", "sketch")
STEP_STATES = ("valid", "unclear", "wrong")
FEATURE_STATES = ("seen", "missing", "unsure")
# Proofs and sketches are never more than "Medium" until 3.5 measures them: the teacher always makes the call.
CHECK_CONFIDENCE = 0.5


def _pick(v, allowed: tuple, default: str) -> str:
    v = str(v or "").strip().lower()
    return v if v in allowed else default


class ProofStep(BaseModel):
    text: str = ""
    state: str = "unclear"
    reason: str = ""

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, v) -> str:
        return _pick(v, STEP_STATES, "unclear")


class SketchFeature(BaseModel):
    name: str = ""
    state: str = "unsure"
    evidence: str = ""

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, v) -> str:
        return _pick(v, FEATURE_STATES, "unsure")


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
    work_type: str = ""                  # working | proof | sketch (3.5 step C); "" until classified
    steps: list[ProofStep] = []          # proofs only
    features: list[SketchFeature] = []   # sketches only
    sketch: str = ""                     # the transcriber's neutral description of a drawn graph, if any

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("steps", "features", mode="before")
    @classmethod
    def _list(cls, v):
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


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


def _weight(mark_type: str) -> int:
    """M1/A1/B1 are worth 1, M2/A2 are worth 2 (the digit in the mark type)."""
    m = re.search(r"\d+", mark_type or "")
    return int(m.group()) if m else 1


def reconcile(q: QuestionGrade) -> bool:
    """The model sometimes lists the marks it awarded and then states a different total (seen in testing:
    two marks ticked, total given as 1). When the listed marks add up to the question's maximum, the ticks
    are the evidence: recompute the total from them. Returns True if it changed the total."""
    if not q.marks or sum(_weight(m.type) for m in q.marks) != q.max_marks:
        return False
    derived = sum(_weight(m.type) for m in q.marks if m.awarded)
    if derived == q.awarded_marks:
        return False
    q.awarded_marks = derived
    return True


def _read_url(url: str, fetch: Callable[[str], bytes], **kw) -> Material:
    if not is_file_ref(url):
        return Material()
    try:
        return read_document(fetch(url), url.split("?")[0].rsplit("/", 1)[-1], **kw)
    except Exception:
        return Material()


def _note(v) -> str:
    return v.strip() if isinstance(v, str) and not v.strip().startswith("http") else ""


TRANSCRIBE_SYSTEM = (
    "You transcribe handwriting. You do NOT mark and you do NOT solve. Copy exactly what the student wrote for each "
    "question, line by line, including mistakes, crossings-out and unfinished lines. Never correct, complete, tidy or "
    "guess. A question number (often circled, e.g. a bare '1') is a label, not an answer. If nothing is written for a "
    "question, its transcription must be an empty string. If the student DREW a graph or sketch, describe only what is "
    "drawn in `sketch`: the shape, where it meets the axes and the values written there, any asymptotes drawn (dashed "
    "lines and their labels), marked turning points, and every label exactly as written. Do not judge whether it is "
    "right, and never add a feature that is not drawn. No drawing: `sketch` is an empty string. "
    "Reply with ONLY one JSON object, no prose."
)
TRANSCRIBE_SCHEMA = ('{"questions":[{"ref":"1","transcription":"<exactly what the student wrote>",'
                     '"sketch":"<what is drawn, or empty>","legible":true|false}]}')

MARK_SYSTEM = (
    "You are a strict maths examiner (Edexcel style: M = method, A = accuracy, B = independent mark). You mark ONLY "
    "from the student's transcribed work you are given, against the mark scheme. Rules: (1) Award a mark only if the "
    "student's own words/lines meet that exact mark scheme line; for every mark you award, `evidence` must be an EXACT "
    "copy of characters from the transcription. If you cannot copy evidence, do not award the mark. (2) No benefit of "
    "the doubt, no 'follow-through', no 'implied' working unless the mark scheme line itself says so. (3) 'Both "
    "required' means both. A wrong sign or a leftover term is wrong. (4) An empty transcription scores 0. "
    "(5) Never invent working the student did not write. (6) A method mark (M) is earned when the correct result of "
    "that method is present in the student's lines (e.g. 12x^2 appearing earns 'reduces a power by one'); do not ask "
    "for an extra visible step unless the scheme line says 'show'. Wrong or missing results still earn nothing. "
    "(7) Set `work_type` per question: \"sketch\" when the question asks for a sketch or graph, \"proof\" when it asks "
    "to prove / show that / deduce, otherwise \"working\". (8) For a proof, list the student's argument in `steps`, in "
    "order: each `text` is an EXACT copy of one line the student wrote; `state` is \"valid\" (follows correctly from "
    "what came before), \"wrong\" (a false or unjustified step) or \"unclear\" (cannot tell); give a short `reason`. "
    "(9) For a sketch, list in `features` each feature the mark scheme or model answer requires (shape, intercepts, "
    "asymptotes, turning points, labels...): `state` is \"seen\" only if the sketch description shows it, with "
    "`evidence` copied from that description; \"missing\" only if the description clearly lacks it; otherwise "
    "\"unsure\". Leave `steps` and `features` empty for other work types. "
    "(10) If TEACHER'S MARKING NOTES are given, they are the teacher's own rulings on this paper: apply them exactly, "
    "and where a note and the mark scheme disagree, the note wins. TEACHER'S CORRECTIONS are questions on other "
    "students' papers where the teacher changed the AI's mark: follow the same judgement on similar work, but mark "
    "this student only from this student's own lines (rule 1 still holds). "
    "Reply with ONLY one JSON object, no prose."
)
MARK_SCHEMA = (
    '{"questions":[{"ref":"1a","max_marks":<int>,"awarded_marks":<int 0..max_marks>,'
    '"work_type":"working"|"proof"|"sketch",'
    '"marks":[{"type":"M1"|"A1"|"B1"|...,"awarded":true|false,"criterion":"<mark scheme line>",'
    '"evidence":"<EXACT copy from the transcription, or empty if not awarded>"}],'
    '"steps":[{"text":"<EXACT copy of one line of the proof>","state":"valid"|"unclear"|"wrong","reason":"<why>"}],'
    '"features":[{"name":"<required feature>","state":"seen"|"missing"|"unsure","evidence":"<copied, or empty>"}],'
    '"feedback":"<one or two sentences for the student>","confidence":<0..1>,"reason":"<why these marks>"}],'
    '"overall_confidence":<0..1>}'
)

SKETCH_RE = re.compile(r"\b(sketch\w*|graph|asymptotes?|correct shape|shape of (the )?(curve|graph))\b", re.I)
PROOF_RE = re.compile(r"\b(prove[sd]?|proof|show that|shows that|hence show|deduce that|by induction|by contradiction)\b",
                      re.I)


def guess_work_type(q: "QuestionGrade", question: str = "") -> str:
    """Same rule as the portal's guess (src/utils/markingTypes.js): from the question's own wording, when it is
    known, and the mark-scheme lines ("Show that ..." is a proof even when no scheme line says so)."""
    text = " \n ".join([question] + [m.criterion for m in q.marks if m.criterion])
    return "sketch" if SKETCH_RE.search(text) else "proof" if PROOF_RE.search(text) else "working"


def question_texts(assignment: dict) -> dict[str, str]:
    """Each question's wording by its number, from the assignment's description when it is written one question per
    line ("Question 3 [3 marks]: Show that ...", as quizzes are; "3) ..." or "Q3: ..." also work)."""
    out: dict[str, str] = {}
    for line in str(assignment.get("description") or "").splitlines():
        m = re.match(r"\s*(?:question\s*|q\s*)?(\d+)\s*[).:\[]", line, re.I)
        if m:
            out.setdefault(m.group(1), line.strip())
    return out


def check_work_type(q: "QuestionGrade", work: str, sketch: str, question: str = "") -> list[str]:
    """3.5 step C: settles the question's work type and checks its proof steps / sketch features against what the
    student actually wrote or drew. Never raises a mark. Returns flags."""
    guess = guess_work_type(q, question)
    t = _pick(q.work_type, WORK_TYPES, "")
    # The marker sometimes calls everything "working"; an explicit sketch / prove wording wins over that.
    q.work_type = guess if (not t or (t == "working" and guess != "working")) else t
    if q.work_type != "proof":
        q.steps = []
    if q.work_type != "sketch":
        q.features = []
    flags: list[str] = []
    if q.work_type == "proof":
        for s in q.steps:
            if not evidence_ok(s.text, work):
                s.state, s.reason = "unclear", "This line was not found in the student's writing. " + s.reason
        q.steps = [s for s in q.steps if s.text.strip()]
        flags.append("proof_check")
    elif q.work_type == "sketch":
        for f in q.features:
            if not q.legible:
                f.state = "unsure"
            elif f.state == "seen" and not evidence_ok(f.evidence, sketch + "\n" + work):
                f.state = "unsure"            # "seen" needs the drawing's description to show it
            elif f.state == "missing" and not sketch.strip():
                f.state = "unsure"            # nothing was described, so nothing can be called missing
        q.features = [f for f in q.features if f.name.strip()]
        flags.append("sketch_check")
    if q.work_type in ("proof", "sketch"):
        q.confidence = min(q.confidence, CHECK_CONFIDENCE)
    return flags


def _norm(s: str) -> str:
    s = (s or "").lower().replace("−", "-").replace("–", "-").replace("×", "x").replace("’", "'")
    return re.sub(r"[\s​]+", "", s)


def evidence_ok(evidence: str, work: str) -> bool:
    """A mark's evidence must really appear in what the student wrote (allowing small OCR-style differences)."""
    ev, w = _norm(evidence), _norm(work)
    if len(ev) < 2 or not w:
        return False
    if ev in w:
        return True
    m = SequenceMatcher(None, ev, w, autojunk=False).find_longest_match(0, len(ev), 0, len(w))
    return m.size >= 0.85 * len(ev) and m.size >= 3


def verify_evidence(q: QuestionGrade, work: str) -> int:
    """Denies every awarded mark whose evidence is not in the student's work. Returns the weight denied."""
    denied_weight = 0
    for m in q.marks:
        if m.awarded and not evidence_ok(m.evidence, work):
            m.awarded = False
            m.evidence = ""
            denied_weight += _weight(m.type)
    if denied_weight:
        if q.marks and sum(_weight(m.type) for m in q.marks) == q.max_marks:
            q.awarded_marks = sum(_weight(m.type) for m in q.marks if m.awarded)
        else:
            q.awarded_marks = max(0, q.awarded_marks - denied_weight)
    return denied_weight


def _json_obj(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    try:
        return json.loads(t[a:b + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON: {str(e)[:100]}") from e


def _ask(complete, system, prompt, images, parse, trace, require_vision=False, skip=None):
    """One AI call, retried once with a stricter reminder when the reply does not parse. Returns (parsed, via)."""
    last = ""
    for attempt in range(2):
        text, via = complete(system, prompt + (f"\n\nYour last reply was invalid ({last}). JSON only." if attempt else ""),
                             images, want_out=3000, require_vision=require_vision, trace=trace, skip=skip or [])
        try:
            return parse(text), via
        except ValueError as e:
            last = str(e)
    raise ValueError(last)


def _parse_transcription(text: str) -> list[dict]:
    qs = _json_obj(text).get("questions")
    if not isinstance(qs, list) or not qs:
        raise ValueError("no questions in transcription")
    return [{"ref": str(q.get("ref", "?")), "transcription": str(q.get("transcription") or "").strip(),
             "sketch": str(q.get("sketch") or "").strip(),
             "legible": bool(q.get("legible", True))} for q in qs if isinstance(q, dict)]


def _lead(ref: str) -> str:
    m = re.match(r"\s*(\d+)", str(ref))
    return m.group(1) if m else str(ref)


def guidance_block(guidance: dict | None) -> str:
    """Roadmap 3.6: the teacher's marking notes and her corrections on other papers of the same work, as prompt text.
    Empty when there are none. Never names a student."""
    g = guidance or {}
    out = ""
    notes = [n for n in g.get("notes") or [] if str(n.get("note") or "").strip()]
    if notes:
        out += "=== TEACHER'S MARKING NOTES (her rulings for this paper) ===\n" + "\n".join(
            f"Q{n['ref']}: {str(n['note']).strip()[:500]}" for n in notes) + "\n"
    ex = g.get("examples") or []
    if ex:
        def m(v):
            return int(v) if float(v).is_integer() else v
        lines = []
        for e in ex:
            work = " ".join(str(e.get("work") or "").split())
            lines.append(f"Q{e['ref']}: the AI gave {m(e['ai'])}" + (f"/{e['max']}" if e.get("max") else "")
                         + f", the teacher gave {m(e['teacher'])}." + (f" Why: {e['why']}" if e.get("why") else "")
                         + (f" (That student wrote: {work})" if work else ""))
        out += ("=== TEACHER'S CORRECTIONS (other students, same paper; learn her judgement, do not copy marks) ===\n"
                + "\n".join(lines) + "\n")
    return out


def grade_submission(assignment: dict, submission: dict, fetch: Callable[[str], bytes],
                     complete: Callable = default_complete, guidance: dict | None = None) -> Outcome:
    """assignment / submission are the `data` JSON of their rows.

    Two stages, so that seeing and judging are separate (a single call kept 'explaining away' wrong working to
    match the mark scheme): (1) a vision model only TRANSCRIBES the handwriting, without the mark scheme in view;
    (2) text models mark from the transcription, twice with different models; every awarded mark must quote the
    transcription, or it is denied."""
    typed = (submission.get("text") or "").strip()
    work = _read_url(submission.get("fileUrl") or "", fetch, max_images=C.MAX_PAGES_STUDENT, always_images=True)
    if not typed and not work.readable:
        return Outcome(status="needs_teacher", note="unreadable_work: no typed answer, and the file could not be read "
                       "(unsupported type, corrupt, or empty). Not graded.", flags=["unreadable_work"])

    ms = _read_url(assignment.get("markingSchemeUrl") or "", fetch)
    ma = _read_url(assignment.get("modelAnswerUrl") or "", fetch)
    ms_notes = _note(assignment.get("markingSchemeNotes")) or _note(assignment.get("markingScheme"))
    ma_notes = _note(assignment.get("modelAnswerNotes")) or _note(assignment.get("modelAnswer"))
    if not (ms.readable or ma.readable or ms_notes or ma_notes):
        return Outcome(status="needs_teacher", note="no_scheme: this assignment has no readable marking scheme or "
                       "model answer. Not graded.", flags=["no_scheme"])
    qp = _read_url(assignment.get("fileUrl") or "", fetch)
    # The questions themselves: a quiz lists each student's own questions in `description`; homework may describe
    # them there too, and a printed question paper adds its text layer. (E test, 2026-11-23: without them a "show
    # that" question was marked as plain working.)
    questions = "\n".join(x for x in (_note(assignment.get("description")), qp.text[:4000]) if x)
    qtexts = question_texts(assignment)

    def block(title: str, m: Material, notes: str) -> str:
        body = "\n".join(x for x in (notes, m.text) if x) or ("(see attached page images)" if m.images else "")
        return f"=== {title} ===\n{body}\n" if body else ""

    trace: list[str] = []
    flags: list[str] = []
    # Noted in the job's trace, not in `flags`: the portal treats any flag as "look closely at this draft".
    g = guidance or {}
    if g.get("notes") or g.get("examples"):
        trace.append(f"teacher guidance: {len(g.get('notes') or [])} note(s), {len(g.get('examples') or [])} correction(s)")
    via_all: list[str] = []
    transcript: list[dict] = []

    # Stage 1: transcribe the uploaded file (vision only). Typed answers need no transcription.
    if work.images:
        p1 = (block("QUESTION PAPER (printed, for the question numbers only)", qp, _note(assignment.get("description")))
              + "\nThe attached images are the student's pages. Transcribe the student's answer to every question "
                "in the paper."
              + (f"\nText layer of the file (may only be the printed paper): {work.text[:1500]}" if work.text else "")
              + "\nRespond with JSON of this shape:\n" + TRANSCRIBE_SCHEMA)
        try:
            transcript, via1 = _ask(complete, TRANSCRIBE_SYSTEM, p1, work.images + qp.images, _parse_transcription,
                                    trace, require_vision=True)
            via_all.append(via1)
        except ValueError as e:
            return Outcome(status="needs_teacher", note=f"bad_ai_reply: transcription {e}. Tried: " + "; ".join(trace)[:250],
                           flags=["bad_ai_reply"])
        if all(not t["legible"] for t in transcript):
            return Outcome(status="needs_teacher", note="unreadable_work: the AI could not read the work. Not graded. "
                           "Tried: " + "; ".join(trace)[:250], flags=["unreadable_work"], via=via1)
        if any(not t["legible"] for t in transcript):
            flags.append("unreadable_part")
    work_lines = "\n".join(f"Q{t['ref']}: {t['transcription'] or '(nothing written)'}"
                           + (f"\n  [sketch drawn: {t['sketch']}]" if t.get("sketch") else "") for t in transcript)
    student = (f"Typed answer:\n{typed}\n" if typed else "") + (f"Transcribed handwriting:\n{work_lines}\n" if transcript else "")
    # A sketch's description counts as the student's work: marks for a drawn feature quote it as evidence.
    work_text = (typed + "\n" + "\n".join("\n".join(x for x in (t["transcription"], t.get("sketch", "")) if x)
                                          for t in transcript)).strip()

    # Stage 2: mark from the transcription, with two different models.
    p2 = (f"Assignment: {assignment.get('title', '')}\n\n"
          + (f"=== QUESTIONS ===\n{questions}\n" if questions else "") + block("MARK SCHEME", ms, ms_notes)
          + block("MODEL ANSWER", ma, ma_notes) + guidance_block(guidance)
          + "\n=== STUDENT WORK (transcribed) ===\n" + student
          + "\nMark every question in the mark scheme (0 for anything not written). Use the mark scheme's own marks "
            "per question. Respond with JSON of this shape:\n" + MARK_SCHEMA)
    mark_images = ms.images + ma.images          # only non-empty if the scheme itself is a scan
    try:
        res, via2 = _ask(complete, MARK_SYSTEM, p2, mark_images, parse_reply, trace, require_vision=bool(mark_images))
    except ValueError as e:
        return Outcome(status="needs_teacher", note=f"bad_ai_reply: {e}. Tried: " + "; ".join(trace)[:250],
                       flags=["bad_ai_reply"])
    via_all.append(via2)
    try:
        res2, via3 = _ask(complete, MARK_SYSTEM, p2, mark_images, parse_reply, trace, require_vision=bool(mark_images),
                          skip=[via2])
        via_all.append(via3)
    except Exception:
        res2, via3 = None, ""                    # no second model available: single marker, flagged
        flags.append("single_marker")

    changed = False
    for q in res.questions:
        if verify_evidence(q, work_text):
            flags.append("evidence_not_found")
        changed |= reconcile(q)
    if changed:
        flags.append("marks_recomputed")
    if res2:
        for q2 in res2.questions:
            verify_evidence(q2, work_text)
            reconcile(q2)
        alt = {_lead(q.ref): q for q in res2.questions}
        disagree = [q for q in res.questions
                    if alt.get(_lead(q.ref)) is not None and alt[_lead(q.ref)].awarded_marks != q.awarded_marks]
        if disagree:
            flags.append("markers_disagree")
            alt3, via4 = {}, ""
            try:                                  # a third, different model breaks the tie
                res3, via4 = _ask(complete, MARK_SYSTEM, p2, mark_images, parse_reply, trace,
                                  require_vision=bool(mark_images), skip=[via2, via3])
                for q3 in res3.questions:
                    verify_evidence(q3, work_text)
                    reconcile(q3)
                alt3 = {_lead(q.ref): q for q in res3.questions}
                via_all.append(via4)
            except Exception:
                pass
            for q in disagree:
                o, t = alt[_lead(q.ref)], alt3.get(_lead(q.ref))
                vals = [q.awarded_marks, o.awarded_marks] + ([t.awarded_marks] if t else [])
                new = sorted(vals)[1] if t else min(vals)       # median of three, else the lower mark
                q.reason = (q.reason + f" | Markers gave {vals} ({via2}, {via3}" + (f", {via4}" if t else "")
                            + f"); used {new}.").strip(" |")
                q.awarded_marks = new
                q.confidence = min(q.confidence, 0.5)
    for q in res.questions:
        t = next((x for x in transcript if _lead(x["ref"]) == _lead(q.ref)), None)
        q.transcription = (t["transcription"] if t else typed)[:1500]
        q.legible = t["legible"] if t else True
        q.sketch = (t.get("sketch", "") if t else "")[:800]
        flags.extend(check_work_type(q, work_text, q.sketch, qtexts.get(_lead(q.ref), "")))
    if res.overall_confidence < LOW_CONFIDENCE or any(q.confidence < LOW_CONFIDENCE for q in res.questions):
        flags.append("low_confidence")
    total_max = sum(q.max_marks for q in res.questions)
    note = "; ".join(trace)[:400]              # which models were tried and which failed (no student content)
    conf = min([res.overall_confidence] + [q.confidence for q in res.questions])
    return Outcome(status="graded", total_awarded=sum(q.awarded_marks for q in res.questions), total_max=total_max,
                   confidence=max(0.0, min(1.0, conf)), questions=[q.model_dump() for q in res.questions], note=note,
                   flags=list(dict.fromkeys(flags)), via=" + ".join(via_all))
