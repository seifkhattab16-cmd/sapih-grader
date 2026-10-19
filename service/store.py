"""Supabase access through PostgREST with the service-role key (this host only). Uses the tables from
supabase_grading.sql in the portal repo."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import httpx
from . import config as C


class Store:
    def __init__(self, client: httpx.Client | None = None):
        self.c = client or httpx.Client(timeout=30)

    def _h(self, extra: dict | None = None) -> dict:
        k = C.supabase_key()
        return {"apikey": k, "Authorization": f"Bearer {k}", "Content-Type": "application/json", **(extra or {})}

    def _u(self, table: str) -> str:
        return f"{C.supabase_url()}/rest/v1/{table}"

    def requeue_stale(self, minutes: int = 10) -> None:
        """A host that died mid-grade leaves jobs stuck on 'running'."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        self.c.patch(self._u("grading_jobs"), headers=self._h(),
                     params={"status": "eq.running", "updated_at": f"lt.{cutoff}"}, json={"status": "queued"}).raise_for_status()

    def requeue_all(self, statuses=("needs_teacher", "failed", "done")) -> None:
        """Teacher asked for a re-grade (workflow checkbox). Approved grades are never overwritten (save_grade checks)."""
        self.c.patch(self._u("grading_jobs"), headers=self._h(),
                     params={"status": f"in.({','.join(statuses)})"},
                     json={"status": "queued", "attempts": 0, "note": None, "updated_at": _now()}).raise_for_status()

    def claim_next(self) -> dict | None:
        """Takes the oldest queued job. The PATCH only matches a row still 'queued', so two workers never get the same job."""
        r = self.c.get(self._u("grading_jobs"), headers=self._h(), params={
            "status": "eq.queued", "attempts": f"lt.{C.MAX_ATTEMPTS}", "order": "created_at.asc", "limit": "1"})
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        job = rows[0]
        r = self.c.patch(self._u("grading_jobs"), headers=self._h({"Prefer": "return=representation"}),
                         params={"id": f"eq.{job['id']}", "status": "eq.queued"},
                         json={"status": "running", "attempts": job["attempts"] + 1, "updated_at": _now()})
        r.raise_for_status()
        got = r.json()
        return got[0] if got else None

    def finish(self, job_id: str, status: str, note: str = "") -> None:
        self.c.patch(self._u("grading_jobs"), headers=self._h(), params={"id": f"eq.{job_id}"},
                     json={"status": status, "note": note[:500], "updated_at": _now()}).raise_for_status()

    def get_submission(self, kind: str, sid: str) -> dict | None:
        if kind == "class_question":
            return self._class_question_answer(sid)
        table = "submissions" if kind == "submission" else "quiz_submissions"
        r = self.c.get(self._u(table), headers=self._h(), params={"id": f"eq.{sid}", "limit": "1"})
        r.raise_for_status()
        row = (r.json() or [None])[0]
        if row and kind == "quiz":
            row["_quiz"] = self._quiz_paper(row)
        return row

    def _one(self, table: str, **eq) -> dict | None:
        r = self.c.get(self._u(table), headers=self._h(), params={**{k: f"eq.{v}" for k, v in eq.items()}, "limit": "1"})
        r.raise_for_status()
        return (r.json() or [None])[0]

    def _quiz_paper(self, sub: dict) -> dict:
        """The paper this student sat, with the bank's mark scheme and model answer for each question.
        A versioned quiz keeps each student's own questions in exam_versions (the quiz row has none)."""
        quiz = self._one("quizzes", id=sub.get("quiz_id")) or {}
        qd = quiz.get("data") or {}
        ver = self._one("exam_versions", quiz_id=sub.get("quiz_id"), student_uid=sub.get("student_uid"))
        questions = (ver or {}).get("questions") or qd.get("questions") or []
        ids = [q["qbId"] for q in questions if q.get("qbId")]
        bank = {}
        if ids:
            r = self.c.get(self._u("question_bank"), headers=self._h(), params={
                "id": f"in.({','.join(ids)})", "select": "id,stem,marks,mark_scheme,model_answer,data"})
            r.raise_for_status()
            bank = {b["id"]: b for b in r.json()}
        return {"title": qd.get("title") or "Quiz", "type": qd.get("type"), "versioned": bool(ver) or bool(qd.get("versioned")),
                "has_version": bool(ver), "questions": questions, "bank": bank}

    def _class_question_answer(self, sid: str) -> dict | None:
        """A class-question answer shaped like a submission row (data.fileUrl / data.text), plus the question
        and its solution under `_cq` so the worker can build the mark scheme."""
        r = self.c.get(self._u("class_question_answers"), headers=self._h(), params={"id": f"eq.{sid}", "limit": "1"})
        r.raise_for_status()
        a = (r.json() or [None])[0]
        if not a:
            return None
        q = self.c.get(self._u("class_questions"), headers=self._h(), params={"id": f"eq.{a['class_question_id']}", "limit": "1"})
        q.raise_for_status()
        q = (q.json() or [None])[0] or {}
        s = self.c.get(self._u("class_question_solutions"), headers=self._h(),
                       params={"class_question_id": f"eq.{a['class_question_id']}", "limit": "1"})
        s.raise_for_status()
        s = (s.json() or [{}])[0]
        return {"id": a["id"], "course_id": q.get("course_id"), "assignment_id": None, "student_uid": a["student_uid"],
                "data": {"fileUrl": a.get("photo_url") or "", "text": a.get("answer_text") or ""},
                "_cq": {"question": q, "solution": s}}

    def get_assignment(self, aid: str | None) -> dict | None:
        if not aid:
            return None
        r = self.c.get(self._u("assignments"), headers=self._h(), params={"id": f"eq.{aid}", "limit": "1"})
        r.raise_for_status()
        return (r.json() or [None])[0]

    def save_grade(self, job: dict, sub: dict, out) -> None:
        row = {"kind": job["kind"], "submission_id": job["submission_id"], "course_id": sub.get("course_id"),
               "assignment_id": sub.get("assignment_id"), "student_uid": sub.get("student_uid"),
               "total_awarded": out.total_awarded, "total_max": out.total_max, "confidence": out.confidence,
               "questions": out.questions, "flags": out.flags, "via": out.via, "status": "draft"}
        # A teacher-approved grade is never overwritten by a re-grade.
        r = self.c.get(self._u("ai_grades"), headers=self._h(), params={
            "kind": f"eq.{job['kind']}", "submission_id": f"eq.{job['submission_id']}", "select": "status"})
        r.raise_for_status()
        if r.json() and r.json()[0]["status"] == "approved":
            return
        self.c.post(self._u("ai_grades"), params={"on_conflict": "kind,submission_id"},
                    headers=self._h({"Prefer": "resolution=merge-duplicates"}), json=row).raise_for_status()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
