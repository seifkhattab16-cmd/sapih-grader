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
        table = "submissions" if kind == "submission" else "quiz_submissions"
        r = self.c.get(self._u(table), headers=self._h(), params={"id": f"eq.{sid}", "limit": "1"})
        r.raise_for_status()
        return (r.json() or [None])[0]

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
