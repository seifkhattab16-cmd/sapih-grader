import json
import pymupdf
import pytest
from service import grader_core, worker
from service.grader_core import QuestionGrade, MarkLine, evidence_ok, reconcile, verify_evidence
from service.providers import ProviderError, Route, complete


def pdf(text: str) -> bytes:
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), text)
    return d.tobytes()


FILES = {"https://f/ms.pdf": pdf("M1 for factorising. A1 for x=2. " * 5), "https://f/work.pdf": pdf("x^2-4=0 so x=2 " * 5)}
fetch = lambda u: FILES[u]
ASSIGN = {"title": "T", "markingSchemeUrl": "https://f/ms.pdf"}
SUB = {"fileUrl": "https://f/work.pdf"}

TRANSCRIPT = {"questions": [{"ref": "1", "transcription": "x^2-4=0 (x-2)(x+2) x=2 x=-2", "legible": True}]}


def marking(awarded=True, evidence="(x-2)(x+2)", total=2, conf=0.9):
    return {"questions": [{"ref": "1", "max_marks": 2, "awarded_marks": total, "feedback": "ok", "confidence": conf,
                           "reason": "r", "marks": [
                               {"type": "M1", "awarded": awarded, "criterion": "factorise", "evidence": evidence},
                               {"type": "A1", "awarded": awarded, "criterion": "roots", "evidence": "x=2 x=-2"}]}],
            "overall_confidence": conf}


class Fake:
    """Routes by system prompt: transcription (vision), first marker m1, second marker m2."""
    def __init__(self, transcript=TRANSCRIPT, first=None, second="same", no_second=False, third=None):
        self.transcript, self.first, self.second, self.no_second = transcript, first or marking(), second, no_second
        self.third = third
        self.calls = []

    def __call__(self, system, text, images, want_out=3000, require_vision=False, trace=None, skip=None, **kw):
        self.calls.append({"system": system[:20], "images": len(images or []), "rv": require_vision, "skip": skip})
        if system.startswith("You transcribe"):
            return json.dumps(self.transcript), "vision"
        if skip and "m2" in skip:
            if self.third is None:
                raise ProviderError("no third model")
            return json.dumps(self.third), "m3"
        if skip and "m1" in skip:
            if self.no_second:
                raise ProviderError("only one model")
            return json.dumps(self.first if self.second == "same" else self.second), "m2"
        return json.dumps(self.first), "m1"


def test_two_stage_grades_and_requires_vision_only_for_transcription():
    f = Fake()
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.status == "graded" and out.total_awarded == 2 and out.total_max == 2
    assert out.via == "vision + m1 + m2" and out.flags == []
    assert f.calls[0]["rv"] is True and f.calls[0]["images"] >= 1          # handwriting is read by a vision model
    assert f.calls[1]["rv"] is False and f.calls[1]["images"] == 0          # marking sees text only, no images
    assert out.questions[0]["transcription"].startswith("x^2-4=0")


def test_invented_evidence_is_denied():
    # the model claims x = 4 but the student never wrote it
    f = Fake(first=marking(evidence="x = 4"))
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.total_awarded == 1 and "evidence_not_found" in out.flags    # M1 denied, A1 kept (its evidence is real)


def test_blank_answer_with_only_question_label_scores_zero():
    blank = {"questions": [{"ref": "1", "transcription": "", "legible": True}]}
    f = Fake(transcript=blank, first=marking(evidence="1", total=2))
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.total_awarded == 0 and "evidence_not_found" in out.flags


def test_markers_disagree_without_third_uses_lower_and_flags():
    f = Fake(first=marking(total=2), second=marking(awarded=False, evidence="", total=0))
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.total_awarded == 0 and "markers_disagree" in out.flags and out.confidence <= 0.5


def test_third_marker_breaks_the_tie_by_majority():
    # one strict marker must not drag a correct answer down: two of three agree on full marks
    f = Fake(first=marking(total=2), second=marking(awarded=False, evidence="", total=0), third=marking(total=2))
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.total_awarded == 2 and "markers_disagree" in out.flags and out.via.endswith("m3")


def test_single_marker_flagged_when_no_second_model():
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(no_second=True))
    assert out.status == "graded" and "single_marker" in out.flags


def test_unreadable_work_never_calls_ai():
    def boom(*a, **k): raise AssertionError("AI must not be called")
    out = grader_core.grade_submission(ASSIGN, {"fileUrl": "https://f/work.docx", "text": ""},
                                       lambda u: b"PK\x03\x04notapdf", boom)
    assert out.status == "needs_teacher" and "unreadable_work" in out.flags


def test_title_or_filename_is_not_work():
    def boom(*a, **k): raise AssertionError("AI must not be called")
    out = grader_core.grade_submission(ASSIGN, {"fileName": "my great answer.pdf", "text": "  "}, fetch, boom)
    assert out.status == "needs_teacher"


def test_no_scheme_never_calls_ai():
    def boom(*a, **k): raise AssertionError("AI must not be called")
    out = grader_core.grade_submission({"title": "T"}, SUB, fetch, boom)
    assert out.status == "needs_teacher" and "no_scheme" in out.flags


def test_all_illegible_goes_to_teacher_not_zero():
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(
        transcript={"questions": [{"ref": "1", "transcription": "", "legible": False}]}))
    assert out.status == "needs_teacher" and "unreadable_work" in out.flags


def test_bad_json_from_marker_goes_to_teacher():
    def f(system, text, images, **kw):
        if system.startswith("You transcribe"):
            return json.dumps(TRANSCRIPT), "v"
        return "sorry, 3/3", "m"
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert out.status == "needs_teacher" and "bad_ai_reply" in out.flags


def test_low_confidence_flagged():
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(first=marking(conf=0.3)))
    assert "low_confidence" in out.flags


def test_evidence_matching_is_forgiving_about_spacing_and_minus_signs():
    assert evidence_ok("x = −2", "x=-2 x=-3")
    assert evidence_ok("(x - 2)(x - 3)", "(x-2)(x-3)=0")
    assert not evidence_ok("x = 4", "x=-2 x=-3")
    assert not evidence_ok("1", "1")               # a bare label is never evidence
    assert not evidence_ok("", "anything")


def test_total_recomputed_from_ticked_marks():
    q = QuestionGrade(ref="3", max_marks=3, awarded_marks=1,
                      marks=[MarkLine(type="M1", awarded=True), MarkLine(type="A1", awarded=True), MarkLine(type="A1", awarded=False)])
    assert reconcile(q) and q.awarded_marks == 2


def test_no_recompute_when_marks_do_not_add_to_max():
    q = QuestionGrade(ref="3", max_marks=3, awarded_marks=2, marks=[MarkLine(type="M1", awarded=True)])
    assert not reconcile(q) and q.awarded_marks == 2


def test_verify_evidence_recomputes_total():
    q = QuestionGrade(ref="2", max_marks=2, awarded_marks=2,
                      marks=[MarkLine(type="M1", awarded=True, evidence="no such text"), MarkLine(type="A1", awarded=True, evidence="x=2")])
    assert verify_evidence(q, "x = 2") == 1 and q.awarded_marks == 1


def test_provider_fallback_skip_and_vision_rules():
    class R:
        def __init__(self, fail_first=True): self.n = 0; self.fail_first = fail_first
        def post(self, url, **kw):
            self.n += 1
            class Resp:
                status_code = 200 if (self.n > 1 or not self.fail_first) else 429
                def raise_for_status(s):
                    if s.status_code != 200: raise RuntimeError("429")
                def json(s): return {"choices": [{"message": {"content": "hi"}}]}
            return Resp()
    mk = lambda n, imgs=0, u="u": Route(n, "openai", u, "k", "m", imgs, 100)
    assert complete("s", "t", routes=[mk("a"), mk("b")], client=R())[1] == "b"           # first rate-limited
    assert complete("s", "t", routes=[mk("a"), mk("b")], client=R(fail_first=False), skip=["a"])[1] == "b"
    with pytest.raises(ProviderError):
        complete("s", "t", routes=[], client=R())
    with pytest.raises(ProviderError):
        complete("s", "t", ["img"], routes=[mk("text-only")], client=R(), require_vision=True)
    assert complete("s", "t", ["img"], routes=[mk("text-only"), mk("v", 1)], client=R(fail_first=False), require_vision=True)[1] == "v"


class FakeStore:
    def __init__(self, kind="submission"):
        self.job = {"id": "j1", "kind": kind, "submission_id": "s1", "attempts": 1}
        self.status = None; self.saved = None
    def requeue_stale(self): pass
    def claim_next(self):
        j, self.job = self.job, None; return j
    def get_submission(self, k, i): return {"course_id": "c", "assignment_id": "a", "student_uid": "u", "data": SUB}
    def get_assignment(self, i): return {"data": ASSIGN}
    def save_grade(self, job, sub, out): self.saved = out
    def finish(self, id, status, note=""): self.status = (status, note)


def test_worker_saves_draft_and_never_touches_final_grade():
    st = FakeStore()
    assert worker.drain(st, fetch=fetch, complete=Fake()) == 1
    assert st.status[0] == "done" and st.saved.total_awarded == 2


def test_worker_provider_outage_requeues():
    st = FakeStore()
    def down(*a, **k): raise ProviderError("all down")
    worker.drain(st, fetch=fetch, complete=down)
    assert st.status[0] == "queued"


def test_eval_harness_summarises_agreement_per_type(tmp_path):
    from service import eval_harness as eh
    (tmp_path / "w01.png").write_bytes(pdf("x"))                 # any readable bytes; the Fake never looks at them
    it = {"id": "w01", "type": "working", "page": "w01.png", "scheme": "Q1 (2 marks): M1, A1", "teacher_marks": {"1": 2}}
    r = eh.run_item(it, tmp_path, Fake())
    assert r["status"] == "graded" and r["rows"] == [{"ref": "1", "teacher": 2, "ai": 2, "max": 2}]
    r2 = {**r, "type": "proof", "rows": [{"ref": "1", "teacher": 1, "ai": 2, "max": 2}, {"ref": "2", "teacher": 2, "ai": None, "max": None}]}
    s = eh.summarise([r, r2])
    assert s["working"]["exact_pct"] == 100 and s["working"]["passes_bar"]
    assert s["proof"]["over"] == 1 and s["proof"]["under"] == 1 and not s["proof"]["passes_bar"]


def test_class_question_photo_is_graded_against_the_solution():
    st = FakeStore(kind="class_question")
    st.get_submission = lambda k, i: {"course_id": "c", "assignment_id": None, "student_uid": "u", "data": SUB,
                                      "_cq": {"question": {"stem": "Solve x^2-4=0"},
                                              "solution": {"mark_scheme": [{"ref": "1", "marks": "M1 A1"}], "model_answer": []}}}
    f = Fake()
    assert worker.drain(st, fetch=fetch, complete=f) == 1
    assert st.status[0] == "done" and st.saved.total_awarded == 2


def test_class_question_assignment_carries_scheme_and_answer():
    a = worker.class_question_assignment({"question": {"stem": "Q"}, "solution": {"mark_scheme": ["M1"], "model_answer": []}})
    assert a["markingSchemeNotes"] == '["M1"]' and a["modelAnswerNotes"] == "" and a["description"] == "Q"


def _paper(**kw):
    p = {"title": "Mock", "type": "bank", "versioned": True, "has_version": True,
         "questions": [{"qbId": "a", "marks": 2, "stem": "Solve x^2-4=0"}, {"qbId": "b", "marks": 1, "stem": "Expand"}],
         "bank": {"a": {"mark_scheme": [{"ref": "1", "marks": "M1 A1"}], "model_answer": ["x=2"]},
                  "b": {"mark_scheme": [], "model_answer": []}}}
    p.update(kw)
    return p


def test_quiz_assignment_uses_the_students_own_questions_and_skips_missing_schemes():
    a = worker.quiz_assignment(_paper())
    assert "Question 1 [2 marks]: Solve x^2-4=0" in a["description"]
    assert "Question 2: (no typed mark scheme" in a["markingSchemeNotes"] and "Question 1: " in a["modelAnswerNotes"]
    assert worker.quiz_assignment(_paper(bank={})) is None


def test_quiz_job_is_graded_from_the_students_version():
    st = FakeStore(kind="quiz")
    st.get_submission = lambda k, i: {"course_id": "c", "quiz_id": "q", "student_uid": "u", "data": SUB, "_quiz": _paper()}
    assert worker.drain(st, fetch=fetch, complete=Fake()) == 1
    assert st.status[0] == "done" and st.saved.total_awarded == 2


def test_quiz_job_without_a_version_or_scheme_goes_to_the_teacher():
    for paper in (_paper(has_version=False), _paper(type="pdf"), _paper(bank={})):
        st = FakeStore(kind="quiz")
        st.get_submission = lambda k, i, p=paper: {"course_id": "c", "quiz_id": "q", "student_uid": "u", "data": SUB, "_quiz": p}
        assert worker.drain(st, fetch=fetch, complete=Fake()) == 1
        assert st.status[0] == "needs_teacher" and st.saved is None
