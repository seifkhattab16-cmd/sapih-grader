import json
import pymupdf
import pytest
from service import grader_core, worker
from service.providers import ProviderError, Route, complete


def pdf(text: str) -> bytes:
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), text)
    return d.tobytes()


GOOD = {"questions": [{"ref": "1", "max_marks": 3, "awarded_marks": 9, "transcription": "x=2",
                       "marks": [{"type": "M1", "awarded": True, "criterion": "c", "evidence": "e"}],
                       "feedback": "ok", "confidence": 0.9, "reason": "r", "legible": True}],
        "overall_confidence": 0.9}
FILES = {"https://f/ms.pdf": pdf("M1 for factorising. A1 for x=2. " * 5), "https://f/work.pdf": pdf("x^2-4=0 so x=2 " * 5)}
fetch = lambda u: FILES[u]
ASSIGN = {"title": "T", "markingSchemeUrl": "https://f/ms.pdf"}
SUB = {"fileUrl": "https://f/work.pdf"}


def test_clamps_marks_and_grades():
    seen = {}
    def fake(system, text, images, want_out=3000):
        seen["images"], seen["text"] = images, text
        return "```json\n" + json.dumps(GOOD) + "\n```", "fake"
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, fake)
    assert out.status == "graded" and out.total_awarded == 3 and out.total_max == 3  # 9 clamped to 3
    assert out.flags == [] and out.via == "fake"
    assert "M1 for factorising" in seen["text"] and len(seen["images"]) >= 1   # student pages sent as images


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


def test_bad_json_retries_once_then_teacher():
    calls = []
    def fake(system, text, images, want_out=3000):
        calls.append(1); return "sorry, here are the marks: 3/3", "fake"
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, fake)
    assert len(calls) == 2 and out.status == "needs_teacher" and "bad_ai_reply" in out.flags


def test_low_confidence_flagged():
    low = json.loads(json.dumps(GOOD)); low["questions"][0]["confidence"] = 0.3
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, lambda s, t, i, want_out=3000: (json.dumps(low), "f"))
    assert "low_confidence" in out.flags


def test_provider_fallback_and_total_failure():
    class R:  # minimal client: first route 429s, second works
        def __init__(self): self.n = 0
        def post(self, url, **kw):
            self.n += 1
            class Resp:
                status_code = 200 if self.n > 1 else 429
                def raise_for_status(s):
                    if s.status_code != 200: raise RuntimeError("429")
                def json(s): return {"choices": [{"message": {"content": "hi"}}]}
            return Resp()
    mk = lambda n: Route(n, "openai", "u", "k", "m", 0, 100)
    assert complete("s", "t", routes=[mk("a"), mk("b")], client=R())[1] == "b"
    with pytest.raises(ProviderError):
        complete("s", "t", routes=[], client=R())


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
    assert worker.drain(st, fetch=fetch, complete=lambda s, t, i, want_out=3000: (json.dumps(GOOD), "f")) == 1
    assert st.status[0] == "done" and st.saved.total_awarded == 3


def test_worker_provider_outage_requeues():
    st = FakeStore()
    def down(*a, **k): raise ProviderError("all down")
    worker.drain(st, fetch=fetch, complete=down)
    assert st.status[0] == "queued"
