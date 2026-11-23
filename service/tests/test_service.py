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
