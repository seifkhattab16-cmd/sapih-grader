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
        self.calls.append({"system": system[:20], "images": len(images or []), "rv": require_vision, "skip": skip, "text": text})
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


# ── 3.5 step C: work type, proof steps, sketch features ──

def test_plain_working_is_typed_working_with_no_steps_or_features():
    m = marking()
    m["questions"][0]["steps"] = [{"text": "x=2", "state": "valid"}]          # stray steps on working are dropped
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(first=m))
    q = out.questions[0]
    assert q["work_type"] == "working" and q["steps"] == [] and q["features"] == []
    assert "proof_check" not in out.flags and "sketch_check" not in out.flags and q["confidence"] == 0.9


PROOF_T = {"questions": [{"ref": "1", "transcription": "n^2+n = n(n+1) one of n, n+1 is even so n(n+1) is even",
                          "legible": True}]}


def proof_marking(steps, work_type="proof"):
    return {"questions": [{"ref": "1", "max_marks": 2, "awarded_marks": 2, "work_type": work_type, "feedback": "ok",
                           "confidence": 0.95, "reason": "r", "steps": steps, "marks": [
                               {"type": "M1", "awarded": True, "criterion": "factorises", "evidence": "n(n+1)"},
                               {"type": "A1", "awarded": True, "criterion": "complete argument", "evidence": "is even"}]}],
            "overall_confidence": 0.95}


def test_proof_steps_are_checked_against_the_writing_and_always_flagged():
    steps = [{"text": "n^2+n = n(n+1)", "state": "valid", "reason": "factorised"},
             {"text": "n(n+1) is a product of consecutive integers", "state": "valid", "reason": "invented"},
             {"text": "so n(n+1) is even", "state": "maybe", "reason": ""}]
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(transcript=PROOF_T, first=proof_marking(steps)))
    q = out.questions[0]
    assert q["work_type"] == "proof" and "proof_check" in out.flags
    assert [s["state"] for s in q["steps"]] == ["valid", "unclear", "unclear"]   # invented line, unknown state
    assert "not found in the student's writing" in q["steps"][1]["reason"]
    assert q["confidence"] <= 0.5 and out.confidence <= 0.5 and out.total_awarded == 2   # never raises or drops marks


def test_scheme_that_says_show_that_overrides_a_working_label():
    m = proof_marking([], work_type="working")
    m["questions"][0]["marks"][1]["criterion"] = "Shows that n^2+n is even"
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(transcript=PROOF_T, first=m))
    assert out.questions[0]["work_type"] == "proof" and out.questions[0]["steps"] == []
    assert "proof_check" in out.flags


SKETCH_T = {"questions": [{"ref": "1", "transcription": "y = 1/(x-2)",
                           "sketch": "two curves either side of a dashed vertical line labelled x=2; crosses y axis at -1/2",
                           "legible": True}]}


def sketch_marking(features):
    return {"questions": [{"ref": "1", "max_marks": 3, "awarded_marks": 2, "work_type": "sketch", "feedback": "ok",
                           "confidence": 0.9, "reason": "r", "features": features, "marks": [
                               {"type": "B1", "awarded": True, "criterion": "correct shape of the graph",
                                "evidence": "two curves either side"},
                               {"type": "B1", "awarded": True, "criterion": "asymptote x=2", "evidence": "dashed vertical line labelled x=2"},
                               {"type": "B1", "awarded": False, "criterion": "horizontal asymptote y=0", "evidence": ""}]}],
            "overall_confidence": 0.9}


def test_sketch_features_need_the_drawing_to_show_them():
    feats = [{"name": "Asymptotes", "state": "seen", "evidence": "dashed vertical line labelled x=2"},
             {"name": "Intercepts with the axes", "state": "seen", "evidence": "crosses x axis at 3"},   # not drawn
             {"name": "Labels", "state": "missing"},
             {"name": "", "state": "seen"}]
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(transcript=SKETCH_T, first=sketch_marking(feats)))
    q = out.questions[0]
    assert q["work_type"] == "sketch" and "sketch_check" in out.flags and q["sketch"].startswith("two curves")
    assert [(f["name"], f["state"]) for f in q["features"]] == [
        ("Asymptotes", "seen"), ("Intercepts with the axes", "unsure"), ("Labels", "missing")]
    assert out.total_awarded == 2 and "evidence_not_found" not in out.flags   # marks may quote the sketch description
    assert q["confidence"] <= 0.5


def test_sketch_with_nothing_described_or_illegible_is_all_unsure():
    no_drawing = {"questions": [{"ref": "1", "transcription": "y = 1/(x-2)", "sketch": "", "legible": True}]}
    feats = [{"name": "Asymptotes", "state": "missing"}, {"name": "Shape", "state": "seen", "evidence": "y = 1/(x-2)"}]
    m = sketch_marking(feats)
    for mk in m["questions"][0]["marks"]:
        mk["awarded"], mk["evidence"] = False, ""
    m["questions"][0]["awarded_marks"] = 0
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(transcript=no_drawing, first=m))
    assert [f["state"] for f in out.questions[0]["features"]] == ["unsure", "seen"]   # "missing" needs a description

    q = QuestionGrade(max_marks=1, work_type="sketch", legible=False,
                      features=[{"name": "Asymptotes", "state": "seen", "evidence": "x=2"}])
    assert grader_core.check_work_type(q, "x=2", "x=2") == ["sketch_check"] and q.features[0].state == "unsure"


def test_transcriber_is_asked_to_describe_sketches_without_the_scheme():
    f = Fake(transcript=SKETCH_T, first=sketch_marking([]))
    grader_core.grade_submission(ASSIGN, SUB, fetch, f)
    assert "sketch" in grader_core.TRANSCRIBE_SCHEMA and "describe only what is drawn" in grader_core.TRANSCRIBE_SYSTEM
    assert "work_type" in grader_core.MARK_SCHEMA and '"steps"' in grader_core.MARK_SCHEMA


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


# ---- S1 / past-paper bank rows: the mark scheme is a picture, the typed worked solution carries the marks ----
S1_ANSWER = [{"part": "(a)", "working": "Let $X\sim N(40,2.4^2)$. $P(X<37)=P(Z<-1.25)=0.1056$", "answer": "0.106", "marks": "M1 M1 A1",
              "tip": "Tables give 0.1056."},
             {"part": "(b)", "working": "z = 0.2533 so m = 31.4", "answer": "m=31.4", "marks": "M1 B1 A1"}]


def test_scheme_from_model_answer_lists_parts_marks_answers_and_tips():
    t = worker.scheme_from_model_answer(S1_ANSWER)
    assert "(a) [M1 M1 A1]: final answer 0.106" in t and "(b) [M1 B1 A1]: final answer m=31.4" in t and "Tables give 0.1056" in t
    assert worker.scheme_from_model_answer([]) == "" and worker.scheme_from_model_answer(None) == ""
    assert worker.scheme_from_model_answer([{"working": "words only", "answer": "x"}]) == ""      # no marks listed: not a scheme


def test_image_only_past_paper_question_is_markable_and_its_ocr_stem_is_not_shown():
    paper = _paper(questions=[{"qbId": "pp", "marks": 6, "stem": "garbled \x01 OCR text"}],
                   bank={"pp": {"stem": "garbled \x01 OCR text", "marks": 6, "mark_scheme": [], "model_answer": S1_ANSWER,
                                "data": {"image_only": True}}})
    a = worker.quiz_assignment(paper)
    assert a is not None and "(shown as a picture)" in a["description"] and "garbled" not in a["description"]
    assert "[M1 M1 A1]" in a["markingSchemeNotes"] and "no typed mark scheme" not in a["markingSchemeNotes"]


def test_typed_mark_scheme_still_wins_and_a_row_with_neither_is_left_to_the_teacher():
    paper = _paper(questions=[{"qbId": "ai", "marks": 2}, {"qbId": "none", "marks": 3}],
                   bank={"ai": {"stem": "AI question", "marks": 2, "mark_scheme": [{"ref": "1", "marks": "B1 B1"}], "model_answer": S1_ANSWER, "data": {"ai_generated": True}},
                         "none": {"stem": "x", "marks": 3, "mark_scheme": [], "model_answer": [], "data": {"image_only": True}}})
    a = worker.quiz_assignment(paper)
    assert "AI question" in a["description"] and '"marks": "B1 B1"' in a["markingSchemeNotes"].replace("'", '"')
    assert "Question 2: (no typed mark scheme" in a["markingSchemeNotes"]


# ── Private uploads (portal P0.3): "storage://uploads/<path>" is read through the Storage API with the service key ──
def test_private_upload_is_read_like_a_link():
    ref = "storage://uploads/c/aaaa/s/bbbb/work.pdf"
    FILES[ref] = FILES["https://f/work.pdf"]
    try:
        m = grader_core._read_url(ref, fetch, always_images=True)
        assert m.readable and m.images
    finally:
        del FILES[ref]


def test_fetch_bytes_signs_storage_requests_with_the_service_key(monkeypatch):
    import httpx
    from service import materials
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sekret")
    seen = {}

    def handler(req):
        seen["url"], seen["auth"], seen["apikey"] = str(req.url), req.headers.get("authorization"), req.headers.get("apikey")
        return httpx.Response(200, content=b"%PDF-1.4 x")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert materials.fetch_bytes("storage://uploads/c/a b/s/u/1-work.pdf", client) == b"%PDF-1.4 x"
    assert seen["url"] == "https://proj.supabase.co/storage/v1/object/authenticated/uploads/c/a%20b/s/u/1-work.pdf"
    assert seen["auth"] == "Bearer sekret" and seen["apikey"] == "sekret"


def test_fetch_bytes_sends_no_key_to_web_links(monkeypatch):
    import httpx
    from service import materials
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sekret")
    seen = {}
    with httpx.Client(transport=httpx.MockTransport(lambda r: (seen.setdefault("auth", r.headers.get("authorization")), httpx.Response(200, content=b"x"))[1])) as client:
        materials.fetch_bytes("https://res.cloudinary.com/x/work.pdf", client)
    assert seen["auth"] is None


@pytest.mark.parametrize("bad", ["storage://uploads", "storage://uploads/../secret", "http://plain/work.pdf", "ftp://x/y"])
def test_fetch_bytes_refuses_bad_references(monkeypatch, bad):
    from service import materials
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sekret")
    with pytest.raises(ValueError):
        materials.fetch_bytes(bad)


# ---- 3.6: the teacher's marking notes and corrections reach the marking prompt ----
from service.store import teacher_examples

GUIDE = {"notes": [{"ref": "1", "note": "Accept answers in terms of pi"}],
         "examples": [{"ref": "1", "max": 2, "ai": 2.0, "teacher": 1.0, "why": "no check of x=-2", "work": "x = 2"}]}


def _capturing():
    seen = []
    base = Fake()
    def c(system, text, images, **kw):
        seen.append((system, text))
        return base(system, text, images, **kw)
    return c, seen


def test_marking_prompt_carries_the_teachers_notes_and_corrections():
    c, seen = _capturing()
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, c, guidance=GUIDE)
    marks = [t for s, t in seen if s.startswith("You are a strict")]
    assert marks and all("Q1: Accept answers in terms of pi" in t for t in marks)
    assert all("the AI gave 2/2, the teacher gave 1. Why: no check of x=-2 (That student wrote: x = 2)" in t for t in marks)
    assert not any("TEACHER'S" in t for s, t in seen if s.startswith("You transcribe"))   # the reader never sees them
    assert out.flags == [] and "teacher guidance: 1 note(s), 1 correction(s)" in out.note  # info, not a warning flag


def test_no_guidance_leaves_the_prompt_unchanged():
    c, seen = _capturing()
    grader_core.grade_submission(ASSIGN, SUB, fetch, c, guidance={"notes": [], "examples": []})
    assert not any("TEACHER'S" in t for s, t in seen)


def test_teacher_examples_keep_only_her_own_changes_and_reasons():
    q = lambda ref, ai, t: {"ref": ref, "max_marks": 3, "awarded_marks": ai, "transcription": "w", "teacher": t}
    drafts = [
        {"submission_id": "me", "questions": [q("1", 3, {"awarded": 0})]},                       # this paper: skipped
        {"submission_id": "a", "questions": [q("1", 3, {"awarded": 3}),                          # agreed, no why
                                             q("2", 3, {"awarded": 1, "why": "sign error"}),
                                             q("3", 2, {"awarded": 0, "bulk": True}),            # Approve all
                                             q("4", 2, {"awarded": 0, "unmarked": True})]},      # not a verdict
        {"submission_id": "b", "questions": [q("2", 0, {"awarded": 2}),                          # same Q again: skipped
                                             q("5", 1, {"awarded": 1, "why": "ok to leave as 3/4"})]},
    ]
    ex = teacher_examples(drafts, exclude_submission="me")
    assert [(e["ref"], e["ai"], e["teacher"]) for e in ex] == [("2", 3, 1), ("5", 1, 1)]
    assert teacher_examples(drafts, "me", limit=1)[0]["why"] == "sign error"


def test_worker_reads_guidance_for_assignments_but_not_versioned_quizzes():
    calls = []
    st = FakeStore()
    st.guidance = lambda kind, wid, exclude_submission=None: calls.append((kind, wid, exclude_submission)) or GUIDE
    assert worker.drain(st, fetch=fetch, complete=Fake()) == 1
    assert calls == [("submission", "a", "s1")] and st.status[0] == "done"
    st = FakeStore(kind="quiz"); calls.clear()
    st.guidance = lambda *a, **k: calls.append(a) or GUIDE
    st.get_submission = lambda k, i: {"course_id": "c", "quiz_id": "q", "student_uid": "u", "data": SUB,
                                      "_quiz": {**_paper(), "versioned": True, "has_version": True}}
    worker.drain(st, fetch=fetch, complete=Fake())
    assert calls == [] and st.status[0] == "done"


def test_worker_still_grades_when_guidance_cannot_be_read():
    st = FakeStore()
    def boom(*a, **k): raise RuntimeError("db down")
    st.guidance = boom
    worker.drain(st, fetch=fetch, complete=Fake())
    assert st.status[0] == "done"


def test_store_guidance_queries_notes_and_approved_drafts_of_the_same_work(monkeypatch):
    import httpx
    from service.store import Store
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    seen = []
    def handle(req):
        seen.append((req.url.path, dict(req.url.params)))
        if req.url.path.endswith("/marking_notes"):
            return httpx.Response(200, json=[{"question_ref": "1", "note": "pi ok"}])
        if req.url.path.endswith("/quiz_submissions"):
            return httpx.Response(200, json=[{"id": "s1"}, {"id": "s2"}])
        return httpx.Response(200, json=[{"submission_id": "s2", "questions": [
            {"ref": "1", "max_marks": 2, "awarded_marks": 2, "teacher": {"awarded": 0, "why": "wrong root"}}]}])
    st = Store(httpx.Client(transport=httpx.MockTransport(handle)))
    g = st.guidance("submission", "A1", exclude_submission="s1")
    assert g["notes"] == [{"ref": "1", "note": "pi ok"}] and g["examples"][0]["why"] == "wrong root"
    assert seen[0][1]["work_id"] == "eq.A1"
    assert seen[1][1]["assignment_id"] == "eq.A1" and seen[1][1]["status"] == "eq.approved"
    seen.clear()
    st.guidance("quiz", "Q1", exclude_submission="s1")
    assert seen[1][1]["quiz_id"] == "eq.Q1" and seen[2][1]["submission_id"] == "in.(s1,s2)"
    assert st.guidance("class_question", "x") == {"notes": [], "examples": []}


def test_verdict_eval_compares_the_new_grade_with_the_teachers_marks():
    from service import verdict_eval
    draft = {"id": "d1", "questions": [
        {"ref": "1", "awarded_marks": 2, "teacher": {"awarded": 1, "type": "working"}},
        {"ref": "2", "awarded_marks": 0, "teacher": {"awarded": 0, "bulk": True}},
        {"ref": "3", "awarded_marks": 1, "teacher": {"awarded": 0, "unmarked": True}}]}
    assert verdict_eval.checked_questions(draft) == {"1": (1.0, "working")}
    out = grader_core.grade_submission(ASSIGN, SUB, fetch, Fake(first=marking(total=1)))
    rows = verdict_eval.compare(draft, out)
    assert rows[0]["rows"][0]["teacher"] == 1 and rows[0]["rows"][0]["before"] == 2
    s = verdict_eval.summarise(rows)["working"]
    assert s["questions"] == 1


# ── 3.5 E fixes (test quiz, 2026-11-23) ──

def test_the_marker_and_transcriber_see_the_question_text():
    a = dict(ASSIGN, description="Question 1 [2 marks]: Solve x^2 - 4 = 0")
    f = Fake()
    grader_core.grade_submission(a, SUB, fetch, f)
    assert "Solve x^2 - 4 = 0" in f.calls[0]["text"]                       # transcriber: for the numbering
    assert "=== QUESTIONS ===" in f.calls[1]["text"] and "Solve x^2 - 4 = 0" in f.calls[1]["text"]


def test_a_show_that_question_is_a_proof_even_when_the_marker_says_working():
    a = dict(ASSIGN, description="Question 1 [2 marks]: Given n is an integer, show that n^2+n is even")
    m = proof_marking([], work_type="working")             # no scheme line says "show that"
    out = grader_core.grade_submission(a, SUB, fetch, Fake(transcript=PROOF_T, first=m))
    assert out.questions[0]["work_type"] == "proof" and "proof_check" in out.flags


def test_question_texts_by_number():
    t = grader_core.question_texts({"description": "Question 1 [2 marks]: A\nQuestion 2: B\n3) C\nQ4: D\nnot a question"})
    assert sorted(t) == ["1", "2", "3", "4"] and t["2"] == "Question 2: B"
    assert grader_core.question_texts({}) == {}


def test_openrouter_tie_break_models_are_current_and_end_with_the_free_router(monkeypatch):
    from service.providers import build_routes
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_MODELS"):
        monkeypatch.delenv(k, raising=False)
    names = [r.model for r in build_routes(need_vision=False)]
    assert "inclusionai/ling-3.0-flash:free" not in names and names[-1] == "openrouter/free" and len(names) >= 3


# ── Long files: a practice book the student wrote on, a long scanned model answer (2026-11-23) ──
# Seen live: a 50-page practice book with answers on 18 pages got no draft, because only page 1 reached the
# transcriber (Groq's vision route takes one image) and the model answer's 20 scanned pages were cut to 3.
from service.materials import written_pages, read_document


def book(n: int, ink: tuple = (), scan: bool = False) -> bytes:
    """n A4 pages; `ink` pages (1-based) carry a large mark, as a student's writing would. scan: no text layer."""
    d = pymupdf.open()
    for i in range(1, n + 1):
        p = d.new_page()
        if scan:
            p.draw_rect(pymupdf.Rect(60, 60 + i * 5, 300, 120 + i * 5), color=(0, 0, 0), width=2)
        else:
            p.insert_text((72, 72), f"Question {i}. Printed question text for page {i}. " * 2)
        if i in ink:
            p.draw_rect(pymupdf.Rect(80, 300, 420, 460), color=(0.8, 0, 0), fill=(0.8, 0, 0))
    return d.tobytes()


def test_written_pages_are_the_ones_that_differ_from_the_blank_paper():
    blank = book(8)
    assert written_pages(book(8, ink=(3, 7)), blank) == [2, 6]
    assert written_pages(blank, blank) == []                       # handed in the blank book
    assert written_pages(book(7, ink=(3,)), blank) is None          # a different file: read in order instead
    assert written_pages(pdf("x"), pdf("x")) is None                 # one page: nothing to choose


def test_read_document_reads_the_chosen_pages():
    m = read_document(book(8, ink=(3, 7)), "w.pdf", max_images=24, always_images=True, pages=[2, 6])
    assert len(m.images) == 2 and m.page_numbers == [3, 7] and m.pages == 8


class BookFake:
    """Transcriber: answers per page ('page N'); copier: one model-answer question per page; marker: marks Q3 and Q7."""
    def __init__(self, answers=None):
        self.answers = answers if answers is not None else {3: "3", 7: "7"}
        self.calls = []

    def __call__(self, system, text, images, want_out=3000, require_vision=False, trace=None, skip=None, **kw):
        self.calls.append({"system": system[:8], "images": len(images or []), "rv": require_vision, "text": text})
        hit = re.search(r"[Pp]age (\d+) of the", text)      # not "[page 1]" from the printed paper's text layer
        page = int(hit.group(1)) if hit else 0
        if system.startswith("You transcribe"):
            ref = self.answers.get(page)
            return json.dumps({"questions": [{"ref": ref, "transcription": f"working for q{ref} x=2", "legible": True}]
                               if ref else []}), "vision"
        if system.startswith("You copy"):
            return json.dumps({"questions": [{"ref": str(page), "text": f"Q{page} M1 method A1 x=2"}]}), "vision"
        qs = [{"ref": r, "max_marks": 2, "awarded_marks": 2, "feedback": "ok", "confidence": 0.9, "reason": "r",
               "marks": [{"type": "M1", "awarded": True, "criterion": "m", "evidence": f"working for q{r}"},
                         {"type": "A1", "awarded": True, "criterion": "a", "evidence": "x=2"}]} for r in ("3", "7")]
        return json.dumps({"questions": qs, "overall_confidence": 0.9}), ("m2" if skip else "m1")


import re
BOOK = {"https://f/book.pdf": book(8), "https://f/omar.pdf": book(8, ink=(3, 7)), "https://f/ma.pdf": book(8, scan=True)}
BOOK_A = {"title": "Practice book", "fileUrl": "https://f/book.pdf", "modelAnswerUrl": "https://f/ma.pdf"}


def test_a_practice_book_is_read_on_its_written_pages_and_marked_on_the_questions_answered():
    grader_core._COPIES.clear()
    f = BookFake()
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, f)
    assert out.status == "graded" and out.total_awarded == 4 and out.total_max == 4
    reads = [c for c in f.calls if c["system"].startswith("You tran")]
    assert [re.search(r"page (\d+) of the", c["text"]).group(1) for c in reads] == ["3", "7"]   # only the written pages
    assert all(c["images"] == 1 and c["rv"] for c in reads)                          # one page per call
    copies = [c for c in f.calls if c["system"].startswith("You copy")]
    assert len(copies) == 8 and all(c["images"] == 1 for c in copies)                 # the whole model answer
    marker = next(c for c in f.calls if c["system"].startswith("You are"))
    assert marker["images"] == 0 and "Question 3:" in marker["text"] and "Question 7:" in marker["text"]
    assert "Question 5:" not in marker["text"]                    # not answered: not sent, not marked 0
    assert "read pages 3,7" in out.note and "marked the questions answered: 3,7" in out.note


def test_the_long_model_answer_is_copied_once_per_run():
    grader_core._COPIES.clear()
    f1, f2 = BookFake(), BookFake()
    grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, f1)
    grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, f2)
    assert not [c for c in f2.calls if c["system"].startswith("You copy")]


def test_a_blank_book_is_not_sent_to_the_ai():
    def boom(*a, **k): raise AssertionError("AI must not be called")
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/book.pdf"}, BOOK.__getitem__, boom)
    assert out.status == "needs_teacher" and out.note.startswith("no_writing")


def test_written_pages_with_no_answers_found_go_to_the_teacher():
    grader_core._COPIES.clear()
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, BookFake(answers={}))
    assert out.status == "needs_teacher" and out.note.startswith("no_answers_found")


def test_a_long_photo_file_without_the_blank_paper_is_read_page_by_page_in_order():
    grader_core._COPIES.clear()
    files = {"https://f/w.pdf": book(9, scan=True), "https://f/ms.pdf": FILES["https://f/ms.pdf"]}
    f = BookFake(answers={3: "3", 7: "7"})
    out = grader_core.grade_submission({"title": "T", "markingSchemeUrl": "https://f/ms.pdf"},
                                       {"fileUrl": "https://f/w.pdf"}, files.__getitem__, f)
    reads = [c for c in f.calls if c["system"].startswith("You tran")]
    assert len(reads) == 9 and out.status == "graded"            # every page, not just the first


def test_merge_joins_a_question_that_runs_over_two_pages():
    m = grader_core.merge_transcripts([[{"ref": "4", "transcription": "a", "sketch": "", "legible": True}],
                                       [{"ref": "4 ", "transcription": "b", "sketch": "", "legible": False},
                                        {"ref": "5", "transcription": "c", "sketch": "", "legible": True}]])
    assert [(t["ref"], t["transcription"], t["legible"]) for t in m] == [("4", "a\nb", False), ("5", "c", True)]


def test_drain_takes_no_new_job_after_the_time_budget():
    class S:
        def __init__(self): self.n = 0
        def requeue_stale(self): pass
    ticks = iter([0, 100, 300, 500, 900])
    st = S()

    def one(store, **kw):
        store.n += 1
        return True
    import service.worker as w
    orig = w.process_one
    w.process_one = one
    try:
        assert w.drain(st, limit=25, budget=480, clock=lambda: next(ticks)) == 2
    finally:
        w.process_one = orig


# ── Free-tier rate limits on long papers: pause, keep the pages read, carry on next run (2026-11-23) ──
# Seen live: the first long-book run hit 429 on every provider part-way and the job used up its 3 attempts at once.

class LimitedFake(BookFake):
    """Like BookFake, but every provider is rate-limited after `budget` calls."""
    def __init__(self, budget, **kw):
        super().__init__(**kw)
        self.budget = budget

    def __call__(self, system, text, images, **kw):
        if self.budget <= 0:
            raise ProviderError("all providers failed: 429", rate_limited=True)
        self.budget -= 1
        return super().__call__(system, text, images, **kw)


def test_a_rate_limited_long_paper_pauses_and_the_next_run_reuses_the_pages_read():
    grader_core._COPIES.clear()
    cache = {}
    with pytest.raises(grader_core.Paused):
        grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, LimitedFake(5),
                                     cache=cache)
    assert len(cache) == 5                                          # the 5 pages read are kept
    grader_core._COPIES.clear()                                     # a new run: only the lasting cache survives
    f = BookFake()
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, f, cache=cache)
    assert out.status == "graded" and out.total_awarded == 4
    vision = [c for c in f.calls if c["system"].startswith(("You tran", "You copy"))]
    assert len(vision) == 2 + 8 - 5                                 # only the pages not read before


def test_a_long_paper_pauses_when_the_run_is_out_of_time():
    grader_core._COPIES.clear()
    t = iter(range(0, 10_000, 100))
    with pytest.raises(grader_core.Paused):
        grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/omar.pdf"}, BOOK.__getitem__, BookFake(),
                                     cache={}, deadline=250, clock=lambda: next(t))


def test_complete_waits_and_retries_when_every_route_is_rate_limited():
    import httpx
    calls, slept = [], []

    def post(url, headers=None, json=None):
        calls.append(url)
        req = httpx.Request("POST", url)
        if len(calls) <= 2:
            return httpx.Response(429, headers={"retry-after": "7"}, request=req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=req)

    class C:
        def post(self, *a, **k): return post(*a, **k)
    r = Route("a", "openai", "https://x/a", "k", "m", 0, 100)
    out, via = complete("s", "t", routes=[r, Route("b", "openai", "https://x/b", "k", "m", 0, 100)], client=C(),
                        sleep=slept.append)
    assert out == "ok" and slept == [7] and len(calls) == 3


def test_complete_marks_a_rate_limit_it_could_not_get_past():
    import httpx

    class C:
        def post(self, url, **k): return httpx.Response(429, request=httpx.Request("POST", url))
    with pytest.raises(ProviderError) as e:
        complete("s", "t", routes=[Route("a", "openai", "https://x/a", "k", "m", 0, 100)], client=C(), sleep=lambda s: 0)
    assert e.value.rate_limited


def test_worker_pauses_a_rate_limited_job_without_counting_the_attempt_and_stops_the_run():
    class S:
        def __init__(self):
            self.jobs = [{"id": "j1", "kind": "submission", "submission_id": "s1", "attempts": 1},
                         {"id": "j2", "kind": "submission", "submission_id": "s2", "attempts": 1}]
            self.paused_jobs, self.finished = [], []
        def requeue_stale(self): pass
        def claim_next(self): return self.jobs.pop(0) if self.jobs else None
        def get_submission(self, kind, sid): return {"assignment_id": "a", "data": {"fileUrl": "https://f/omar.pdf"}}
        def get_assignment(self, aid): return {"data": BOOK_A}
        def pause(self, job, note): self.paused_jobs.append((job["id"], note))
        def finish(self, *a): self.finished.append(a)
        def cache(self): return {}
    grader_core._COPIES.clear()
    st = S()
    n = worker.drain(st, limit=5, fetch=BOOK.__getitem__, complete=LimitedFake(3))
    assert n == 1 and st.paused_jobs and st.paused_jobs[0][0] == "j1" and not st.finished
    assert st.jobs and st.paused is True                            # j2 waits for the next run


def test_run_once_continues_only_after_progress():
    from service.run_once import should_continue

    class S: pass
    s = S()
    assert not should_continue(s)
    s.paused = True
    assert not should_continue(s)                                   # paused with nothing read: do not loop
    s.cache_writes = 3
    assert should_continue(s)


# ── Long papers are marked a few questions at a time (Groq refused a 20-question marking request: 413) ──

class BatchFake(BookFake):
    """Marker marks exactly the questions whose student work is in its prompt."""
    def __call__(self, system, text, images, want_out=3000, require_vision=False, trace=None, skip=None, **kw):
        if not system.startswith("You are"):
            return super().__call__(system, text, images, want_out, require_vision, trace, skip, **kw)
        self.calls.append({"system": system[:8], "images": 0, "rv": require_vision, "text": text})
        work = text.split("=== STUDENT WORK")[1]
        refs = re.findall(r"^Q(\d+):", work, re.M)
        qs = [{"ref": r, "max_marks": 2, "awarded_marks": 2, "feedback": "ok", "confidence": 0.9, "reason": "r",
               "marks": [{"type": "M1", "awarded": True, "criterion": "m", "evidence": f"working for q{r}"},
                         {"type": "A1", "awarded": True, "criterion": "a", "evidence": "x=2"}]} for r in refs]
        return json.dumps({"questions": qs, "overall_confidence": 0.9}), ("m2" if skip else "m1")


def test_a_long_paper_is_marked_in_batches_and_the_batches_are_cached(monkeypatch):
    monkeypatch.setattr(grader_core.C, "MARK_PROMPT_CHARS", 10)
    monkeypatch.setattr(grader_core.C, "MARK_BATCH_QUESTIONS", 2)
    files = dict(BOOK, **{"https://f/many.pdf": book(8, ink=(1, 2, 3, 5, 7))})
    answers = {1: "1", 2: "2", 3: "3", 5: "5", 7: "7"}
    grader_core._COPIES.clear()
    cache = {}
    f = BatchFake(answers=answers)
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/many.pdf"}, files.__getitem__, f, cache=cache)
    assert out.status == "graded" and out.total_awarded == 10 and out.total_max == 10
    markers = [c for c in f.calls if c["system"].startswith("You are") and "STUDENT WORK" in c["text"]]
    assert len(markers) == 6                                        # 3 batches (1,2 / 3,5 / 7) x 2 markers
    first = markers[0]["text"]
    assert "Question 1:" in first and "Question 2:" in first and "Question 3:" not in first   # only its questions
    assert [q["ref"] for q in out.questions] == ["1", "2", "3", "5", "7"]
    grader_core._COPIES.clear()
    f2 = BatchFake(answers=answers)                                 # a resumed run: nothing is marked twice
    grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/many.pdf"}, files.__getitem__, f2, cache=cache)
    assert not f2.calls


def test_one_ai_call_never_runs_past_its_time_budget():
    import httpx
    t = iter([0, 10, 200, 210, 220, 230])          # the first route alone took 190 s

    class C:
        def post(self, url, **k): return httpx.Response(500, request=httpx.Request("POST", url))
    routes = [Route(n, "openai", f"https://x/{n}", "k", "m", 0, 100) for n in "abc"]
    with pytest.raises(ProviderError) as e:
        complete("s", "t", routes=routes, client=C(), sleep=lambda s: 0, clock=lambda: next(t))
    assert e.value.rate_limited and "too long" in str(e.value)     # the job pauses instead of the run being killed


# ── Labels on a long paper (live draft 2026-11-24: every question came back as "Q1"; parts "a", "b)" stood alone) ──

def test_parts_without_a_number_join_the_question_on_the_page_before():
    m = grader_core.merge_transcripts([
        [{"ref": "4", "transcription": "a) x=1", "sketch": "", "legible": True}],
        [{"ref": "b)", "transcription": "y=2", "sketch": "", "legible": True},
         {"ref": "c", "transcription": "z=3", "sketch": "", "legible": True}],
        [{"ref": "5a", "transcription": "w", "sketch": "", "legible": True}]])
    assert [t["ref"] for t in m] == ["4", "4b", "4c", "5a"]


class OneLabelFake(BatchFake):
    """A marker that numbers every question's parts '1a', '1b' whatever it was given."""
    def __call__(self, system, text, images, want_out=3000, require_vision=False, trace=None, skip=None, **kw):
        if not system.startswith("You are"):
            return super().__call__(system, text, images, want_out, require_vision, trace, skip, **kw)
        self.calls.append({"system": system[:8], "text": text})
        refs = re.findall(r"^Q(\d+)", text.split("=== STUDENT WORK")[1], re.M)
        qs = [{"ref": "1a", "max_marks": 2, "awarded_marks": 2, "feedback": "ok", "confidence": 0.9, "reason": "r",
               "marks": [{"type": "M1", "awarded": True, "criterion": "m", "evidence": f"working for q{r}"},
                         {"type": "A1", "awarded": True, "criterion": "a", "evidence": "x=2"}]} for r in refs[:1]]
        return json.dumps({"questions": qs, "overall_confidence": 0.9}), ("m2" if skip else "m1")


def test_a_one_question_batch_keeps_its_own_number_and_working(monkeypatch):
    monkeypatch.setattr(grader_core.C, "MARK_PROMPT_CHARS", 10)
    files = dict(BOOK, **{"https://f/two.pdf": book(8, ink=(3, 7))})
    grader_core._COPIES.clear()
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/two.pdf"}, files.__getitem__,
                                       OneLabelFake(answers={3: "3", 7: "7"}), cache={})
    assert [q["ref"] for q in out.questions] == ["3a", "7a"]
    assert "q3" in out.questions[0]["transcription"] and "q7" in out.questions[1]["transcription"]


def test_a_question_with_no_model_answer_is_not_marked_against_nothing(monkeypatch):
    monkeypatch.setattr(grader_core.C, "MARK_PROMPT_CHARS", 10)

    class NoQ7(BatchFake):
        def __call__(self, system, text, images, *a, **kw):
            if system.startswith("You copy") and "Page 7 " in text:
                return json.dumps({"questions": []}), "vision"
            return super().__call__(system, text, images, *a, **kw)
    files = dict(BOOK, **{"https://f/two.pdf": book(8, ink=(3, 7))})
    grader_core._COPIES.clear()
    out = grader_core.grade_submission(BOOK_A, {"fileUrl": "https://f/two.pdf"}, files.__getitem__,
                                       NoQ7(answers={3: "3", 7: "7"}), cache={})
    assert [q["ref"] for q in out.questions] == ["3"] and "no model answer found): 7" in out.note


# --- Marked-work R1.1: a generated booklet lists its bank questions and pages in the PDF Subject ---
from service.materials import booklet_manifest, questions_on_page, MANIFEST_PREFIX


def _pdf_with_subject(subject: str, pages: int = 3) -> bytes:
    d = pymupdf.open()
    for _ in range(pages):
        d.new_page()
    d.set_metadata({"subject": subject})
    return d.tobytes()


def test_booklet_manifest_is_read_from_the_question_paper():
    m = {"v": 1, "unit": "P1", "pages": 6, "q": [{"n": 1, "id": "a", "marks": 6, "page": 2, "solve": []},
                                                 {"n": 2, "id": "b", "marks": 12, "page": 2, "solve": [3]}]}
    got = booklet_manifest(_pdf_with_subject(MANIFEST_PREFIX + json.dumps(m)))
    assert got == m
    assert [q["id"] for q in questions_on_page(got, 1)] == ["a", "b"]   # 0-based page 1 = PDF page 2
    assert [q["id"] for q in questions_on_page(got, 2)] == ["b"]        # its answer sheet
    assert questions_on_page(got, 0) == []                              # the cover


def test_other_files_have_no_booklet_manifest():
    assert booklet_manifest(_pdf_with_subject("pure mathematics 1 (Questions)")) is None
    assert booklet_manifest(_pdf_with_subject(MANIFEST_PREFIX + "{broken")) is None
    assert booklet_manifest(_pdf_with_subject(MANIFEST_PREFIX + json.dumps({"v": 2, "q": []}))) is None
    assert booklet_manifest(b"not a pdf") is None
    assert booklet_manifest(None) is None
    assert questions_on_page(None, 0) == []


# --- Marked-work R1.2: a booklet is marked against each question's own bank mark scheme ---
from service.booklet import booklet_references, scheme_lines, answer_lines


def _booklet(n: int, manifest: dict, ink: tuple = ()) -> bytes:
    d = pymupdf.open(stream=book(n, ink=ink), filetype="pdf")
    d.set_metadata({"subject": MANIFEST_PREFIX + json.dumps(manifest)})
    return d.tobytes()


MANI = {"v": 1, "unit": "P1", "pages": 4, "q": [{"n": 1, "id": "id1", "marks": 3, "page": 2, "solve": []},
                                                {"n": 2, "id": "id2", "marks": 4, "page": 3, "solve": [4]}]}
ROWS = {"id1": {"id": "id1", "marks": 3, "stem": "Find sin x.",
                "mark_scheme": [{"part": "(a)", "markType": "M1", "answer": "sine rule", "notes": ""},
                                {"part": "(a)", "markType": "A1", "answer": "0.548", "notes": "awrt"},
                                {"part": "(b)", "markType": "B1", "answer": "2.56", "notes": ""}],
                "model_answer": [{"part": "(a)", "marks": "M1 A1", "answer": "0.548", "working": "8 sin 0.5 / 7"}]},
        "id2": {"id": "id2", "marks": 4, "stem": "A question with nothing stored", "mark_scheme": [], "model_answer": None}}
BK = {"https://f/bk.pdf": _booklet(4, MANI), "https://f/bk-omar.pdf": _booklet(4, MANI, ink=(2, 4))}


class BookletFake:
    """Transcriber: page 2 gives bare parts, page 4 a wrong number; marker: full marks on what it is given."""
    def __init__(self):
        self.calls = []

    def __call__(self, system, text, images, want_out=3000, require_vision=False, trace=None, skip=None, **kw):
        self.calls.append({"system": system[:8], "text": text})
        page = int((re.search(r"[Pp]age (\d+) of the", text) or [0, 0])[1])
        if system.startswith("You transcribe"):
            refs = {2: ["a)", "b)"], 4: ["7"]}.get(page, [])
            return json.dumps({"questions": [{"ref": r, "transcription": f"x=2 for {r}", "legible": True} for r in refs]}), "vision"
        leads = sorted(set(re.findall(r"Question (\d+):", text)))   # named in both the scheme and the answer
        qs = [{"ref": f"{n}a", "max_marks": 2, "awarded_marks": 2, "feedback": "ok", "confidence": 0.9, "reason": "r",
               "marks": [{"type": "M1", "awarded": True, "criterion": "m", "evidence": "x=2"},
                         {"type": "A1", "awarded": True, "criterion": "a", "evidence": "x=2"}]} for n in leads]
        return json.dumps({"questions": qs, "overall_confidence": 0.9}), ("m2" if skip else "m1")


def test_scheme_and_answer_lines_read_the_bank_format():
    assert scheme_lines(ROWS["id1"]["mark_scheme"]).splitlines() == ["(a) M1: sine rule", "(a) A1: 0.548 (awrt)", "(b) B1: 2.56"]
    assert answer_lines(ROWS["id1"]["model_answer"]) == "(a) [M1 A1] Answer: 0.548\n8 sin 0.5 / 7"
    refs = booklet_references(MANI, ROWS)
    assert refs["1"]["marks"] == 3 and refs["1"]["ms"].startswith("[3 marks] Find sin x.\nMark scheme:")
    assert "Find sin x." not in refs["1"]["ma"]                  # the question text goes once
    assert refs["2"]["ms"] == "" and refs["2"]["ma"] == ""        # nothing stored: never marked against nothing


def test_a_booklet_is_marked_against_its_own_bank_scheme():
    grader_core._COPIES.clear()
    f = BookletFake()
    asked = []
    bank = lambda ids: (asked.append(ids), ROWS)[1]
    out = grader_core.grade_submission({"title": "Booklet", "fileUrl": "https://f/bk.pdf"},   # no MS / MA files at all
                                       {"fileUrl": "https://f/bk-omar.pdf"}, BK.__getitem__, f, bank=bank)
    assert asked == [["id1", "id2"]]
    reads = [c for c in f.calls if c["system"].startswith("You tran")]
    assert "belongs to question 1 of the booklet" in reads[0]["text"]
    assert "belongs to question 2 of the booklet" in reads[1]["text"]          # its answer sheet (page 4)
    markers = [c for c in f.calls if c["system"].startswith("You are")]
    assert markers and all("Question 1:" in c["text"] for c in markers)        # only Q1 has a stored scheme
    assert "(a) A1: 0.548 (awrt)" in markers[0]["text"] and "8 sin 0.5 / 7" in markers[0]["text"]
    assert "Q1a: x=2 for a)" in markers[0]["text"]                             # bare "a)" filed under question 1
    assert out.status == "graded" and out.total_max == 2 and {q["ref"] for q in out.questions} == {"1a"}
    assert "no_answer_for_some" in out.flags
    assert "not marked (no stored mark scheme in the bank): 2" in out.note     # the wrong "7" became question 2
    assert "booklet: 2 bank questions on known pages, 1 with a stored mark scheme" in out.note


def test_a_booklet_without_bank_access_falls_back_to_the_assignment_files():
    grader_core._COPIES.clear()
    out = grader_core.grade_submission({"title": "Booklet", "fileUrl": "https://f/bk.pdf"},
                                       {"fileUrl": "https://f/bk-omar.pdf"}, BK.__getitem__, BookletFake(), bank=None)
    assert out.status == "needs_teacher" and out.note.startswith("no_scheme")


def test_a_question_left_out_of_the_booklet_list_is_not_marked_and_not_flagged():
    grader_core._COPIES.clear()
    m = {**MANI, "q": [MANI["q"][0]]}                       # question 2 taken out of the list (not in the syllabus)
    files = {"https://f/bk.pdf": _booklet(4, m), "https://f/bk-omar.pdf": _booklet(4, m, ink=(2, 4))}
    out = grader_core.grade_submission({"title": "Booklet", "fileUrl": "https://f/bk.pdf"},
                                       {"fileUrl": "https://f/bk-omar.pdf"}, files.__getitem__, BookletFake(),
                                       bank=lambda ids: ROWS)
    assert out.status == "graded" and "no_answer_for_some" not in out.flags
    assert "not marked (not in the booklet's question list): 7" in out.note   # page 4 is no longer a listed page


def test_questions_the_teacher_left_out_are_not_marked_or_counted():
    grader_core._COPIES.clear()
    f = BookFake()                                            # the student answered 3 and 7; the marker marks both
    out = grader_core.grade_submission({**BOOK_A, "leaveOutQuestions": ["7"]}, {"fileUrl": "https://f/omar.pdf"},
                                       BOOK.__getitem__, f)
    assert out.status == "graded" and out.total_awarded == 2 and out.total_max == 2
    assert {q["ref"] for q in out.questions} == {"3"}
    assert "left out by the teacher: 7" in out.note
    markers = [c for c in f.calls if c["system"].startswith("You are")]
    assert all("Qa7" not in c["text"] and "Q7:" not in c["text"] for c in markers)   # its work is not sent either


def test_a_paper_whose_every_answer_was_left_out_goes_to_the_teacher():
    grader_core._COPIES.clear()
    out = grader_core.grade_submission({**BOOK_A, "leaveOutQuestions": ["3", "7"]}, {"fileUrl": "https://f/omar.pdf"},
                                       BOOK.__getitem__, BookFake())
    assert out.status == "needs_teacher"


def test_the_marker_can_give_the_correct_working_for_a_lost_mark():
    reply = json.dumps({"questions": [
        {"ref": "1b", "max_marks": 3, "awarded_marks": 0, "feedback": "Turn 0.548 into an angle first.",
         "correct_working": ["$x = \sin^{-1}(0.548) = 0.580$", "$\pi - 0.580 = 2.56$", "", "a", "b", "c"],
         "confidence": 0.9, "reason": "r", "marks": [{"type": "B1", "awarded": False, "criterion": "0.58"}]},
        {"ref": "2", "max_marks": 1, "awarded_marks": 0, "correct_working": "$y = 3$\n$x = 930$", "confidence": 0.9}],
        "overall_confidence": 0.9})
    res = grader_core.parse_reply(reply)
    assert res.questions[0].correct_working[:2] == ["$x = \sin^{-1}(0.548) = 0.580$", "$\pi - 0.580 = 2.56$"]
    assert len(res.questions[0].correct_working) == 4                  # at most 4 lines, blanks dropped
    assert res.questions[1].correct_working == ["$y = 3$", "$x = 930$"]  # a string is split into lines
    assert "correct_working" in grader_core.MARK_SCHEMA and "16-year-old" in grader_core.MARK_SYSTEM


def test_the_transcriber_says_where_each_answer_starts():
    got = grader_core._parse_page(json.dumps({"questions": [
        {"ref": "1a", "transcription": "x", "top": 0.42}, {"ref": "1b", "transcription": "y", "top": "1.7"},
        {"ref": "1c", "transcription": "z"}]}))
    assert [t["top"] for t in got] == [0.42, None, None]
    assert '"top"' in grader_core.TRANSCRIBE_SCHEMA
